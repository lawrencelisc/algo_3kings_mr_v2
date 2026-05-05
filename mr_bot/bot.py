"""
mr_bot/bot.py — 均值回歸 Bot 主體

整合所有模組：
  - Kalman Filter Z-Score 入場
  - Microprice 取代 mid-price
  - Rule-based Regime Detection（Green/Yellow/Red）
  - Dynamic SL（分段 time-decay）
  - Quarter-Kelly sizing（Hurst adjusted）
  - Adverse Flow Exit
  - 完整手續費計算

設計遵從 Opus 4.6 建議：
  - 保留 principled 模組（Kalman、Microprice、VPIN、Adverse Flow）
  - 簡化易 overfit 模組（幣池只用 Hurst + half-life 兩關）
  - Regime 用 rule-based voting，唔係 weighted sum
  - Sizing 用 quarter-Kelly + shrinkage，唔係 rolling optimization

手續費：
  - 開倉即時從 equity 扣 maker fee（0.0384%）
  - 平倉按 reason 扣 maker/taker fee
  - CSV 記錄 fee_entry / fee_exit / fee_total / net_pnl（來回費用後）
"""
from __future__ import annotations

import csv
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .core.indicators import (
    KalmanZScore,
    RollingHurst,
    VPIN,
    VolBurstDetector,
    OBIAcceleration,
    microprice,
)
from .core.regime import (
    RegimeZone,
    RegimeSignals,
    compute_regime_zone,
    screen_coin,
    DynamicSLConfig,
    SizingConfig,
    compute_sl_distance,
    compute_position_size,
)
from .core.execution import (
    FeeModel,
    Account,
    AdverseFlowMonitor,
    LeeReadyClassifier,
    TradeRecord,
    simulate_limit_fill,
)
from .core.risk import CorrelationGuard, DrawdownThrottle

logger = logging.getLogger("mr_bot")


# ────────────────────────────────────────────────────────────────────────────
# Per-Symbol State
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolState:
    """每個幣種嘅獨立狀態。"""
    symbol: str
    kalman: KalmanZScore = field(default_factory=lambda: KalmanZScore(warm_up=100, mle_window=300))
    hurst_tracker: RollingHurst = field(default_factory=lambda: RollingHurst(window=500, compute_every=50))
    vpin: VPIN = field(default_factory=lambda: VPIN(bucket_size=1000.0, n_buckets=50))
    vol_burst: VolBurstDetector = field(default_factory=lambda: VolBurstDetector(short_window=10, long_window=100))
    obi_accel: OBIAcceleration = field(default_factory=lambda: OBIAcceleration(window=10))
    adverse_flow: AdverseFlowMonitor = field(default_factory=lambda: AdverseFlowMonitor(adverse_bars_threshold=3))
    lee_ready: LeeReadyClassifier = field(default_factory=LeeReadyClassifier)

    # 幣種適性（定期更新）
    hurst_val: Optional[float] = None
    half_life_bars: Optional[float] = None
    eligible: bool = False
    eligibility_checked_at: float = 0.0

    # 持倉
    in_position: bool = False
    position_side: Optional[str] = None
    entry_price: float = 0.0
    entry_time: float = 0.0
    entry_bar: int = 0
    base_sl_distance: float = 0.0    # 入場時嘅初始 SL 距離
    tp_price: float = 0.0
    sl_price: float = 0.0
    entry_fee: float = 0.0           # 已扣嘅開倉費（記錄用）
    z_at_entry: float = 0.0
    vpin_pct_at_entry: float = 0.0
    size: float = 0.0               # contract size
    notional: float = 0.0
    size_mult: float = 1.0
    bar_count: int = 0

    # SL 冷卻期（防止接刀：SL 出場後同一幣唔可以即刻重入）
    sl_cooldown_until_bar: int = 0   # bar_count 超過此值才可再入場

    # Rolling P&L statistics（for Kelly sizing）
    _pnl_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def update_pnl_history(self, net_pnl: float) -> None:
        self._pnl_history.append(net_pnl)

    @property
    def roll_mu(self) -> float:
        if not self._pnl_history:
            return 0.0
        return float(sum(self._pnl_history) / len(self._pnl_history))

    @property
    def roll_sigma2(self) -> float:
        if len(self._pnl_history) < 2:
            return 1e-4
        arr = list(self._pnl_history)
        mu = sum(arr) / len(arr)
        return float(sum((x - mu) ** 2 for x in arr) / (len(arr) - 1))


# ────────────────────────────────────────────────────────────────────────────
# Bot Config
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    """
    所有 parameter 都有 economic rationale。
    避免過多 tunable parameter（Opus 4.6 overfit 警告）。
    """
    # 資金
    initial_equity: float = 1000.0

    # 入場 Z-score 閾值
    z_entry_long: float = -2.0    # Z < -2 → long（價格低估）
    z_entry_short: float = 2.0    # Z > +2 → short（價格高估）
    z_tp: float = 0.0             # Z 回到 0 → TP（mean reversion 完成）

    # 幣池篩選
    hurst_threshold: float = 0.45
    hl_slack: float = 1.5
    coin_rescreen_interval_bars: int = 30   # 每 30 bars 重新 screen（120s bar → 60min）

    # ROUND5: 入場時嘅嚴格 Hurst 閾值（與 universe filter 解耦）
    # universe_relax_hurst（傳入 hurst_threshold）可放寬到 0.55 揀池，
    # 但實際入場必須通過呢個更嚴格嘅閘，避免喺隨機區（h>=0.45）入場。
    hurst_strict_threshold: float = 0.45

    # ROUND5: 極端 z 入場保護
    # |z| 越大，越可能係 trending breakout 而非 MR 機會：
    # 要求 Hurst 嚴格 < extreme_z_hurst_max 才能入場，否則 skip。
    extreme_z_threshold: float = 3.0
    extreme_z_hurst_max: float = 0.40

    # Timeout（分 bar 計）
    timeout_bars: int = 30

    # SL 冷卻期：止損後同一幣種至少等幾多 bars 才可再入場（防接刀）
    sl_cooldown_bars: int = 3       # default 3 bars（3m bar = 9 分鐘冷靜期）

    # DECEL exit 最少持倉 bars：防止「入場後 1-2 bar 就 z 回歸 → 強制 exit」
    # 嘅微利出場（fee × 2 已食晒），俾 mean reversion 至少 N bars 走完。
    decel_min_hold_bars: int = 3

    # SL config
    atr_mult: float = 1.0
    # SL bps 上下限（自適應跨幣種，唔需要 WFA）：
    #   sl_bps = clamp(atr_mult × ATR / price × 1e4, min_sl_bps, max_sl_bps)
    #   BTC: raw≈12bps → floor to 15bps；SOL: raw≈40bps → 直接用；
    #   超波動幣: cap at 80bps 防止 SL 太大虧大錢
    min_sl_bps: float = 15.0   # 最窄 SL = 15 bps（> breakeven 7.84 bps）
    max_sl_bps: float = 80.0   # 最闊 SL = 80 bps（細幣保護上限）
    sl_config: DynamicSLConfig = field(default_factory=DynamicSLConfig)

    # Sizing
    sizing_config: SizingConfig = field(default_factory=SizingConfig)

    # ROUND3: size_mult 硬上限（防止 Kelly 估計誤差在樣本少時放大虧損）
    # 建議初期用 2.0-3.0，跑夠 100 筆後可放寬到 5.0
    max_size_mult: float = 3.0

    # Bar duration（秒）
    bar_duration_sec: float = 60.0

    # VPIN bucket size（合約數）
    vpin_bucket_size: float = 1000.0

    # TP 距離最低要求（bps）：必須 > breakeven
    min_tp_bps: float = 10.0   # 最少 10 bps TP 距離（> 7.84 bps breakeven）

    # Correlation cap（portfolio risk control）
    corr_threshold: float = 0.6   # |pairwise corr| 上限
    corr_window: int = 200        # rolling window（bars）

    # Drawdown throttle（intraday risk control）
    dd_window_sec: float = 3600.0  # rolling window（秒），default 1h
    dd_limit: float = 0.005        # drawdown 上限（0.5% of equity）

    # Output
    csv_path: str = "mr_trades.csv"
    log_diag_every_bars: int = 50

    # Gate funnel diagnostic：每 N 個 global bar（= sum of on_bar across symbols）
    # print 一次入場漏斗統計，方便睇邊道閘 reject 最多。
    # 1000：46 幣 ×  ~22 polls/symbol ≈ 22 分鐘（poll=60s）一次，
    # 足夠 sample 計 |z| 嘅 p95/p99 + 收集 screen 失敗 breakdown。
    gate_log_every_bars: int = 1000


# ────────────────────────────────────────────────────────────────────────────
# CSV Output
# ────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "trade_id", "symbol", "side",
    "entry_price", "exit_price", "size", "notional",
    "gross_pnl", "fee_entry", "fee_exit", "fee_total", "net_pnl",
    "hold_bars", "hold_sec", "reason", "regime_at_exit",
    "hurst_at_entry", "half_life_at_entry",
    "z_at_entry", "vpin_pct_at_entry",
    "size_mult", "entry_is_maker", "exit_is_maker",
    "closed_at_ts",
]


def append_csv(path: str, rec: TradeRecord) -> None:
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        d = rec.to_dict()
        w.writerow({k: d.get(k, "") for k in CSV_FIELDS})


# ────────────────────────────────────────────────────────────────────────────
# Main Bot
# ────────────────────────────────────────────────────────────────────────────

class MeanReversionBot:
    """
    均值回歸 Bot 主體（Paper Trade 版）。

    每個 tick（bar）的處理流程：
      1. 更新 microprice
      2. 更新 Kalman Z-score
      3. 更新 VPIN / Vol Burst / OBI Accel
      4. 計算 Regime Zone
      5. 幣池適性 check（定期）
      6. 持倉管理（TP / SL / DECEL / ADVERSE FLOW / TIMEOUT / REGIME RED）
      7. 入場 check（若未持倉 + GREEN/YELLOW + eligible）
      8. 輸出 CSV + log

    Live 版：將 step 6-7 嘅 account.close/open 換成 ex.create_order()。
    """

    # Gate funnel keys（按入場路徑順序）：
    #   bars         on_bar 進入次數
    #   considered   未持倉，會考慮入場嘅 bar 數
    #   g1_z         Kalman z 已 warm up
    #   g2_eligible  screen_coin 通過（VR + HL）
    #   g3_regime    regime != RED
    #   g4_z_thresh  |z| 達到 z_entry
    #   g5_no_cd     冇 SL cooldown
    #   g6_corr      side-aware CorrelationGuard 通過
    #   g7_fill      simulate_limit_fill 成功
    #   g8_notional  sizing > 0
    #   opens        實際開倉
    _GATE_KEYS = (
        "bars", "considered", "g1_z", "g2_eligible", "g3_regime",
        "g4_z_thresh", "g5_no_cd", "g6_corr", "g7_fill", "g8_notional",
        "opens",
    )

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.fee_model = FeeModel()
        self.account = Account(
            initial_equity=config.initial_equity,
            fee_model=self.fee_model,
        )
        self._states: Dict[str, SymbolState] = {}
        self._global_bar = 0

        self.corr_guard  = CorrelationGuard(
            window=config.corr_window, threshold=config.corr_threshold
        )
        self.dd_throttle = DrawdownThrottle(
            window_sec=config.dd_window_sec, dd_limit=config.dd_limit
        )

        # Gate funnel counters（reset 每次 _log_gate_diag 之後）
        self._gate_stats: Dict[str, int] = {k: 0 for k in self._GATE_KEYS}

        # |z| 分布 sample buffer（只 collect G3 通過嘅 bar，即真正有資格觸發 z 嘅）
        # 用嚟 print max / p95 / p99 / mean，分辨：
        #   p99 << z_entry  → z 結構性偏細（可能 Kalman R inflation），需修 scale
        #   p99 >> z_entry  → z 偶然 cross threshold，純粹樣本不足，等多陣
        self._z_abs_window: List[float] = []

        # screen_coin 失敗原因 breakdown（每次 re-screen 累計一次）
        # data：資料不足（warm-up 中）；vr：VR 太高（趨勢市）；
        # hl：HL 太慢；ok：通過
        self._screen_reasons: Dict[str, int] = {"data": 0, "vr": 0, "hl": 0, "ok": 0}

        # Warm-up flag：歷史 bar replay 期間 pause gate diagnostic
        # （否則 46×600=27,600 個 replay bar 會 print 27 個無意義 GATE_DIAG，
        # 而且 state.eligible 仲未設 → G2 永遠 0%，數字誤導）
        self._in_warmup: bool = False

        logger.info(
            "MeanReversionBot initialized  equity=%.2f  "
            "fee: maker=%.4f%%  taker=%.4f%%  breakeven=%.2f bps",
            config.initial_equity,
            self.fee_model.maker_rate * 100,
            self.fee_model.taker_rate * 100,
            self.fee_model.breakeven_bps(exit_is_maker=False),
        )

    def get_state(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState(symbol=symbol)
        return self._states[symbol]

    # ── Bar 更新入口 ────────────────────────────────────────────────────────

    def on_bar(
        self,
        symbol: str,
        close: float,
        high: float,
        low: float,
        volume: float,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        bid_levels: List[Tuple[float, float]],
        ask_levels: List[Tuple[float, float]],
        atr: float,
        buy_flow: float = 0.0,
        sell_flow: float = 0.0,
        liq_spike: bool = False,
        prices_for_screen: Optional[List[float]] = None,
    ) -> Optional[TradeRecord]:
        """
        每根 bar 呼叫一次。
        返回 TradeRecord 如果有平倉，否則 None。
        """
        self._global_bar += 1
        state = self.get_state(symbol)
        state.bar_count += 1

        # 1. Microprice
        mp = microprice(bid, bid_size, ask, ask_size)

        # 2. Kalman Z-score
        z = state.kalman.update(mp)

        # Risk controls update（每 bar 都要更新，唔係只係入場時）
        self.corr_guard.update(symbol, close)
        self.dd_throttle.update(self.account.equity)

        # 3. Market Structure Indicators
        vpin_raw = state.vpin.update(close, volume)
        vpin_pct = state.vpin.percentile_rank(vpin_raw) if vpin_raw is not None else 50.0
        vol_ratio = state.vol_burst.update(close)
        obi_acc = state.obi_accel.update(bid_size, ask_size)
        hurst_val = state.hurst_tracker.update(close)
        if hurst_val is not None:
            state.hurst_val = hurst_val

        # 4. Regime Zone
        sig = RegimeSignals(
            vpin_pct=vpin_pct,
            liq_spike=liq_spike,
            vol_burst_ratio=vol_ratio,
            obi_accel=obi_acc,
        )
        regime_zone, regime_reason = compute_regime_zone(sig)

        # 5. 幣池適性 check（每 N bars）
        if (
            prices_for_screen is not None
            and (state.bar_count - state.eligibility_checked_at) >= self.config.coin_rescreen_interval_bars
        ):
            # prices_for_screen 包含 warmup 期的 1m-bar closes（最多 600）加上 live closes。
            # half_life_ou 計算結果以「prices 的時間間隔數」為單位（主要是 1m-bar）。
            # warmup_symbol 已把 timeout_bars 轉換為 1m-bar 等價值；
            # 這裡也要做同樣換算，否則 HL cap 比 warmup 嚴 bar_duration/60 倍。
            # e.g. poll_sec=180 → timeout_1m = 20 × 3 = 60 bars（比 20 寬鬆 3 倍）。
            _timeout_1m = max(1, int(
                self.config.timeout_bars * (self.config.bar_duration_sec / 60.0)
            ))
            result = screen_coin(
                prices=prices_for_screen,
                timeout_bars=_timeout_1m,
                hurst_threshold=self.config.hurst_threshold,
                hl_slack=self.config.hl_slack,
            )
            state.eligible = result.eligible
            state.hurst_val = result.hurst or state.hurst_val
            state.half_life_bars = result.half_life_bars
            state.eligibility_checked_at = state.bar_count

            # 累計 screen 結果 breakdown（reason_code: data / vr / hl / ok）
            rc = result.reason_code or "unknown"
            self._screen_reasons[rc] = self._screen_reasons.get(rc, 0) + 1

            if not result.eligible:
                logger.info("SCREEN  %s  INELIGIBLE  %s", symbol, result.reason)

        # 6. 持倉管理
        closed_rec: Optional[TradeRecord] = None
        if state.in_position:
            closed_rec = self._manage_position(
                state=state,
                symbol=symbol,
                close=close,
                atr=atr,
                z=z,
                regime_zone=regime_zone,
                buy_flow=buy_flow,
                sell_flow=sell_flow,
            )

        # 7. 入場（funnel 同時 count 每道閘；warm-up 期間唔計）
        if not self._in_warmup:
            self._gate_stats["bars"] += 1
        if not state.in_position:
            if not self._in_warmup:
                self._gate_stats["considered"] += 1
            if z is not None:
                if not self._in_warmup:
                    self._gate_stats["g1_z"] += 1
                if state.eligible:
                    if not self._in_warmup:
                        self._gate_stats["g2_eligible"] += 1
                    if regime_zone != RegimeZone.RED:
                        if not self._in_warmup:
                            self._gate_stats["g3_regime"] += 1
                            # Sample |z|：G3 通過 = 真正有資格觸發 z 嘅 bar
                            self._z_abs_window.append(abs(float(z)))
                        self._try_entry(
                            state=state,
                            symbol=symbol,
                            close=close,
                            mp=mp,
                            bid=bid,
                            ask=ask,
                            bid_levels=bid_levels,
                            ask_levels=ask_levels,
                            z=float(z),
                            atr=atr,
                            vpin_pct=vpin_pct,
                            regime_zone=regime_zone,
                        )

        # 8. Diag log（warm-up 期間唔 print，避免被 replay 噪音淹沒）
        if not self._in_warmup and self._global_bar % self.config.log_diag_every_bars == 0:
            self._log_diag(symbol, state, z, regime_zone, regime_reason, vpin_pct, vol_ratio)

        # 8b. Gate funnel diag（warm-up 唔 print；reset window 之後再起跳）
        if not self._in_warmup and self._global_bar % self.config.gate_log_every_bars == 0:
            self._log_gate_diag()

        return closed_rec

    # ── 入場邏輯 ────────────────────────────────────────────────────────────

    def _try_entry(
        self,
        state: SymbolState,
        symbol: str,
        close: float,
        mp: float,
        bid: float,
        ask: float,
        bid_levels: List[Tuple[float, float]],
        ask_levels: List[Tuple[float, float]],
        z: float,
        atr: float,
        vpin_pct: float,
        regime_zone: RegimeZone,
    ) -> None:
        side: Optional[str] = None
        if z <= self.config.z_entry_long:
            side = "long"
        elif z >= self.config.z_entry_short:
            side = "short"
        if side is None:
            return
        self._gate_stats["g4_z_thresh"] += 1

        # ── ROUND5: Strict Hurst gate（防 stale eligibility）─────────────
        # universe scan 用 hurst_threshold（可放寬至 0.55 揀池），但 live entry
        # 必須通過 hurst_strict_threshold（預設 0.45）— 兩者解耦，避免喺
        # 隨機區（h_equiv ≥ 0.45）入場。state.hurst_val 由 30 bars 一次嘅
        # re-screen 更新，可能 stale，呢度作為最後一道閘。
        _hurst = state.hurst_val
        _strict = self.config.hurst_strict_threshold
        if _hurst is not None and _hurst >= _strict:
            logger.info(
                "HURST_STRICT  %s %s  H_eq=%.3f >= %.2f → skip entry (stale/weak MR)",
                side.upper(), symbol, _hurst, _strict,
            )
            return

        # ── ROUND5: Extreme z gate ───────────────────────────────────────
        # |z| 極大（≥ extreme_z_threshold，預設 3.0σ）通常係 trending
        # breakout 而非 MR 機會：在 H 邊緣嘅幣，呢類入場係「接刀」。
        # 只有 H 嚴格細於 extreme_z_hurst_max（預設 0.40）才允許入場。
        _z_thr = self.config.extreme_z_threshold
        _z_h_max = self.config.extreme_z_hurst_max
        if abs(z) >= _z_thr:
            if _hurst is None or _hurst >= _z_h_max:
                logger.info(
                    "EXTREME_Z  %s %s  |z|=%.2f >= %.2f  H=%s >= %.2f → skip entry",
                    side.upper(), symbol, abs(z), _z_thr,
                    f"{_hurst:.3f}" if _hurst is not None else "None",
                    _z_h_max,
                )
                return

        # SL 冷卻期：止損後唔可以即刻重入（防接刀 / 連輸）
        if state.bar_count < state.sl_cooldown_until_bar:
            logger.info(
                "SL_COOLDOWN  %s  bars_left=%d  (re-entry blocked)",
                symbol, state.sl_cooldown_until_bar - state.bar_count,
            )
            return
        self._gate_stats["g5_no_cd"] += 1

        # Correlation cap：side-aware（反方向高 ρ 視為對沖，唔 block）
        held_positions = {
            s: st.position_side for s, st in self._states.items()
            if st.in_position and st.position_side
        }
        corr_blocked, corr_reason = self.corr_guard.check_entry(
            symbol, side, held_positions
        )
        if corr_blocked:
            logger.info("CORR_BLOCK  %s %s  %s", side.upper(), symbol, corr_reason)
            return
        self._gate_stats["g6_corr"] += 1

        # Drawdown throttle：連虧保護
        if self.dd_throttle.is_throttled:
            logger.info("DD_THROTTLE  %s  %s", symbol, self.dd_throttle.status_str())

        # Limit order fill simulation（用 microprice 附近掛單）
        limit_price = mp
        fill_price, is_maker = simulate_limit_fill(side, limit_price, bid_levels, ask_levels)
        if fill_price is None:
            return
        self._gate_stats["g7_fill"] += 1

        # Base SL distance（bps-clamped ATR，自適應跨幣種）
        # raw_sl_bps = atr_mult × ATR / price × 1e4
        # clamp 到 [min_sl_bps, max_sl_bps]，防止大幣太緊、細幣太闊
        raw_sl_bps = self.config.atr_mult * atr / max(fill_price, 1e-9) * 1e4
        sl_bps = max(self.config.min_sl_bps, min(raw_sl_bps, self.config.max_sl_bps))
        base_sl = sl_bps / 1e4 * fill_price
        if base_sl <= 0:
            return
        logger.debug(
            "SL  %s  raw=%.1f bps → clamped=%.1f bps  dist=%.4f",
            symbol, raw_sl_bps, sl_bps, base_sl,
        )

        # TP price（Kalman fair value，即 mean reversion target）
        tp_price = state.kalman.fair_value or mp
        if side == "long":
            sl_price = fill_price - base_sl
            tp_price = max(tp_price, fill_price + base_sl * 0.5)
        else:
            sl_price = fill_price + base_sl
            tp_price = min(tp_price, fill_price - base_sl * 0.5)

        # Position size（quarter-Kelly + Hurst adjusted + drawdown throttle）
        notional = compute_position_size(
            equity=self.account.equity,
            entry_price=fill_price,
            sl_distance=base_sl,
            hurst=state.hurst_val or 0.45,
            roll_mu=state.roll_mu,
            roll_sigma2=state.roll_sigma2,
            config=self.config.sizing_config,
            regime_zone=regime_zone,
        ) * self.dd_throttle.size_multiplier
        if notional <= 0:
            return
        self._gate_stats["g8_notional"] += 1

        size = notional / max(fill_price, 1e-9)
        # size_mult 記錄
        base_notional = self.account.equity * self.config.sizing_config.min_fraction
        size_mult = notional / max(base_notional, 1e-9)

        # ── ROUND3: size_mult 硬上限 ──────────────────────────────────────
        # Quarter-Kelly 在樣本少時估計誤差極大，初期大倉會嚴重放大虧損
        # 硬 cap = 3.0×（允許 Kelly 放大，但防止極端情況）
        # 可用 MR_MAX_SIZE_MULT 環境變數調整
        _max_mult = getattr(self.config, "max_size_mult", 3.0)
        if size_mult > _max_mult:
            notional = base_notional * _max_mult
            size = notional / max(fill_price, 1e-9)
            size_mult = _max_mult
            logger.debug(
                "SIZE_MULT CAP  %s  capped to %.1f×  notional=%.2f",
                symbol, _max_mult, notional,
            )

        # ── ROUND4: half-life based size cap ─────────────────────────────
        # 慢速 MR（HL 長）持倉期間 regime 惡化風險高，縮倉保護資本：
        #   HL ≤ 30 bars → 保持 _max_mult（快速 MR，Kelly 可信）
        #   30 < HL ≤ 60 bars → 上限 1.5×
        #   HL > 60 bars → 上限 1.0×（slow MR，regime 視窗太短）
        _hl = state.half_life_bars
        if _hl is not None:
            if _hl > 60:
                _hl_max = 1.0
            elif _hl > 30:
                _hl_max = 1.5
            else:
                _hl_max = _max_mult
            if size_mult > _hl_max:
                _orig_mult = size_mult
                notional = base_notional * _hl_max
                size = notional / max(fill_price, 1e-9)
                size_mult = _hl_max
                logger.info(
                    "HL_SIZE_CAP  %s  HL=%.1f bars → size_mult %.2f× → %.2f×  notional=%.2f",
                    symbol, _hl, _orig_mult, size_mult, notional,
                )

        # ── ROUND5: VPIN-based size cap ──────────────────────────────────
        # VPIN_pct 高 = toxic flow 重，距離 regime RED（>80）只一步之遙；
        # 即使 H/HL 靚，呢個時候大倉等於賭 regime 唔轉紅，期望值差。
        #   VPIN ≥ 80 → 上限 1.0×（regime RED 邊界，唔好放大）
        #   75 ≤ VPIN < 80 → 上限 1.5×
        if vpin_pct >= 80:
            _vpin_max = 1.0
        elif vpin_pct >= 75:
            _vpin_max = 1.5
        else:
            _vpin_max = _max_mult
        if size_mult > _vpin_max:
            _orig_mult = size_mult
            notional = base_notional * _vpin_max
            size = notional / max(fill_price, 1e-9)
            size_mult = _vpin_max
            logger.info(
                "VPIN_SIZE_CAP  %s  VPIN=%.0f%% → size_mult %.2f× → %.2f×  notional=%.2f",
                symbol, vpin_pct, _orig_mult, size_mult, notional,
            )

        # 扣開倉手續費
        entry_fee = self.account.deduct_entry_fee(notional, is_maker=is_maker)

        # 更新 state
        state.in_position = True
        state.position_side = side
        state.entry_price = fill_price
        state.entry_time = time.time()
        state.entry_bar = state.bar_count
        state.base_sl_distance = base_sl
        state.tp_price = tp_price
        state.sl_price = sl_price
        state.entry_fee = entry_fee
        state.z_at_entry = z
        state.vpin_pct_at_entry = vpin_pct
        state.size = size
        state.notional = notional
        state.size_mult = size_mult
        state.adverse_flow.reset()
        self._gate_stats["opens"] += 1

        logger.info(
            "OPEN  %s %s  fill=%.4f  TP=%.4f  SL=%.4f  "
            "Z=%.3f  VPIN_pct=%.0f  H=%.3f  HL=%.1f bars  "
            "notional=%.2f  fee_entry=%.4f  is_maker=%s",
            side.upper(), symbol, fill_price, tp_price, sl_price,
            z, vpin_pct, state.hurst_val or 0, state.half_life_bars or 0,
            notional, entry_fee, is_maker,
        )

    # ── 持倉管理 ────────────────────────────────────────────────────────────

    def _manage_position(
        self,
        state: SymbolState,
        symbol: str,
        close: float,
        atr: float,
        z: Optional[float],
        regime_zone: RegimeZone,
        buy_flow: float,
        sell_flow: float,
    ) -> Optional[TradeRecord]:
        side = state.position_side
        assert side is not None

        # 動態 SL 距離（trailing-only：唔隨時間收緊）
        current_sl_dist = compute_sl_distance(
            base_sl=state.base_sl_distance,
            entry_time=state.entry_time,
            half_life_bars=state.half_life_bars or self.config.timeout_bars,
            bar_duration_sec=self.config.bar_duration_sec,
            config=self.config.sl_config,
            regime_zone=regime_zone,
        )

        # ── Trailing SL：賺到 trigger × ATR 後鎖部分利潤 ──────────────────────
        # base trailing 從 entry ± current_sl_dist 計起，再睇有冇符合 trail 條件
        sl_cfg = self.config.sl_config
        trail_trigger = sl_cfg.trail_trigger_atr * atr
        trail_lock    = sl_cfg.trail_lock_atr * atr

        if side == "long":
            base_new_sl = state.entry_price - current_sl_dist
            # 若已賺超過 trigger → SL trail 上 entry + trail_lock
            if (close - state.entry_price) >= trail_trigger:
                trail_sl = state.entry_price + trail_lock
                base_new_sl = max(base_new_sl, trail_sl)
            # 只 move 向有利方向（向上）；trending follow 唔 shrink
            state.sl_price = max(state.sl_price, base_new_sl)
        else:
            base_new_sl = state.entry_price + current_sl_dist
            if (state.entry_price - close) >= trail_trigger:
                trail_sl = state.entry_price - trail_lock
                base_new_sl = min(base_new_sl, trail_sl)
            state.sl_price = min(state.sl_price, base_new_sl)

        # 檢查出場條件（優先順序：TP > REGIME_RED > SL > ADVERSE_FLOW > TIMEOUT > DECEL）
        reason: Optional[str] = None

        # TP
        if side == "long" and close >= state.tp_price:
            reason = "TP"
        elif side == "short" and close <= state.tp_price:
            reason = "TP"

        # Regime RED：aggressive exit
        if reason is None and regime_zone == RegimeZone.RED:
            reason = "REGIME_RED"

        # SL
        if reason is None:
            if side == "long" and close <= state.sl_price:
                reason = "SL"
            elif side == "short" and close >= state.sl_price:
                reason = "SL"

        # Adverse Flow Exit
        if reason is None:
            adverse = state.adverse_flow.update(buy_flow, sell_flow, side)
            if adverse:
                reason = "ADVERSE_FLOW"

        # Timeout
        if reason is None:
            bars_held = state.bar_count - state.entry_bar
            if bars_held >= self.config.timeout_bars:
                reason = "TIMEOUT"

        # DECEL（Z-score 回歸長期 mean，mean reversion 大致完成）
        # OU-anchored Kalman 修正後：z 反映「price 偏離長期 mean 多少 σ」，
        # 入場 z=±2 → DECEL z=±0.5 = 確實已完成 75% reversion。
        # min_hold_bars guard：防止「入場後 1-2 bar 就 DECEL」嘅冇邊際 trade
        # （fee × 2 已食晒 0.5σ 嘅 P&L）。
        if reason is None and z is not None:
            bars_held_check = state.bar_count - state.entry_bar
            if bars_held_check >= self.config.decel_min_hold_bars:
                if side == "long" and z >= -0.5:
                    reason = "DECEL"
                elif side == "short" and z <= 0.5:
                    reason = "DECEL"

        if reason is None:
            return None

        # 平倉
        exit_price = close
        bars_held = state.bar_count - state.entry_bar
        hold_sec = time.time() - state.entry_time

        rec = self.account.close_trade(
            symbol=symbol,
            side=side,
            entry_price=state.entry_price,
            exit_price=exit_price,
            notional=state.notional,
            size=state.size,
            fee_entry=state.entry_fee,
            reason=reason,
            hold_bars=bars_held,
            hold_sec=hold_sec,
            regime_at_exit=regime_zone.value,
            hurst_at_entry=state.hurst_val,
            half_life_at_entry=state.half_life_bars,
            z_at_entry=state.z_at_entry,
            vpin_pct_at_entry=state.vpin_pct_at_entry,
            size_mult=state.size_mult,
        )

        # 更新 per-symbol P&L history（for Kelly sizing）
        state.update_pnl_history(rec.net_pnl)

        # SL 出場：設定冷卻期，防止接刀連輸
        if reason in ("SL", "REGIME_RED", "ADVERSE_FLOW"):
            state.sl_cooldown_until_bar = state.bar_count + self.config.sl_cooldown_bars
            logger.info(
                "SL_COOLDOWN set  %s  no re-entry for %d bars (~%d min)",
                symbol, self.config.sl_cooldown_bars,
                int(self.config.sl_cooldown_bars * self.config.bar_duration_sec / 60),
            )

        # 清倉 state
        state.in_position = False
        state.position_side = None

        # CSV 記錄
        append_csv(self.config.csv_path, rec)

        logger.info(
            "CLOSE #%d  %s %s  exit=%.4f  gross=%+.4f  "
            "fee_entry=%.4f  fee_exit=%.4f  net=%+.4f  reason=%s  equity=%.2f",
            rec.trade_id, side.upper(), symbol,
            exit_price, rec.gross_pnl,
            rec.fee_entry, rec.fee_exit, rec.net_pnl,
            reason, self.account.equity,
        )
        return rec

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def _log_diag(
        self,
        symbol: str,
        state: SymbolState,
        z: Optional[float],
        zone: RegimeZone,
        zone_reason: str,
        vpin_pct: float,
        vol_ratio: Optional[float],
    ) -> None:
        logger.info(
            "DIAG  %-20s  Z=%s  zone=%s  VPIN_pct=%.0f  vol_ratio=%s  "
            "H=%s  HL=%s  eligible=%s  dd=%.2f%%  throttle=%s  equity=%.2f  fee_paid=%.4f",
            symbol,
            f"{z:.3f}" if z is not None else "None",
            zone.value.upper(),
            vpin_pct,
            f"{vol_ratio:.2f}x" if vol_ratio else "None",
            f"{state.hurst_val:.3f}" if state.hurst_val else "None",
            f"{state.half_life_bars:.1f}" if state.half_life_bars else "None",
            state.eligible,
            self.dd_throttle.current_drawdown * 100,
            self.dd_throttle.is_throttled,
            self.account.equity,
            self.account.total_fee_paid,
        )

    def gate_stats_snapshot(self) -> Dict[str, int]:
        """返回當前 gate funnel counter 嘅 copy（唔 reset）。"""
        return dict(self._gate_stats)

    def start_warmup(self) -> None:
        """
        Warm-up 歷史 replay 開始前呼叫。
        期間 _global_bar 仍會增加（Kalman/VPIN 等狀態正常 update），
        但 gate funnel 唔 count、GATE_DIAG/DIAG 唔 print，
        避免 replay bar 產生無意義 log。
        Pair with end_warmup() afterwards.
        """
        self._in_warmup = True

    def end_warmup(self) -> None:
        """
        Warm-up 完成後呼叫。只切返 live 模式，唔 reset 已累計嘅 live 統計。
        Universe rescan 期間嘅 incremental warm-up 都用呢個（避免 reset live data）。
        如果係 initial warm-up，外面跟住可以呼叫 reset_gate_diag()。
        """
        self._in_warmup = False

    def rollback_entry(self, symbol: str) -> None:
        """
        撤銷 _try_entry() 剛剛建立嘅虛擬倉位。

        適用場景：bot.on_bar() 內部已開倉（state.in_position=True），
        但外部邏輯（cooldown / jail / regime_pause）決定唔落真實訂單。
        若不 rollback，下一 bar _manage_position() 會對一個交易所不存在的幻象倉位
        進行管理，最終 _live_place_exit() 會遇到 "would increase position" 錯誤，
        且 bot 內部 equity 會因虛假出場而失真。

        Rollback 步驟：
          1. 清 in_position / position_side
          2. 退還 entry fee（已從 equity 扣）
          3. reset SL cooldown（唔 penalize 冇落到單）
          4. gate stats 的 opens -= 1（唔計呢次為真實開倉）
        """
        state = self.get_state(symbol)
        if not state.in_position:
            return
        # 退還開倉費（加返去 equity）
        self.account.equity += state.entry_fee
        self.account.total_fee_paid -= state.entry_fee
        # 清倉狀態
        state.in_position = False
        state.position_side = None
        state.entry_fee = 0.0
        state.base_sl_distance = 0.0
        state.sl_price = 0.0
        state.tp_price = 0.0
        state.size = 0.0
        state.notional = 0.0
        state.size_mult = 1.0
        state.z_at_entry = 0.0
        state.vpin_pct_at_entry = 0.0
        # SL cooldown 唔設（未入場，不應有 cooldown 懲罰）
        state.sl_cooldown_until_bar = 0
        # gate stats 修正
        if self._gate_stats.get("opens", 0) > 0:
            self._gate_stats["opens"] -= 1
        logger.info("ENTRY_ROLLBACK  %s  virtual position cleared (live order was blocked)", symbol)

    def reset_gate_diag(self) -> None:
        """
        手動 reset gate funnel + |z| buffer + screen breakdown。
        典型用法：initial warm-up 完成後一次性 clean slate，
        令 live 數字唔被任何前置事件污染。
        """
        self._gate_stats = {k: 0 for k in self._GATE_KEYS}
        self._z_abs_window = []
        self._screen_reasons = {"data": 0, "vr": 0, "hl": 0, "ok": 0}
        logger.info("Gate diagnostic counters reset — fresh live window starting")

    def _log_gate_diag(self) -> None:
        """
        Print 入場漏斗統計 + |z| 分布 + screen breakdown，然後 reset window。

        Funnel：bars → considered → G1 → G2 → G3 → G4 → G5 → G6 → G7 → G8 → opens
        每個 stage 嘅 % 係 conditional on 上一 stage 通過。

        Diagnostic 解讀：
          ZDIST  p99 << z_entry  → z 結構性偏細（Kalman R inflation 等），
                                   降 threshold 都冇用，要修 Kalman scale
          ZDIST  p99 >> z_entry  → z 偶然 cross threshold，純粹 sample 不足
          SCREEN vr 佔大多數      → 趨勢市，universe 太多 trending 幣
          SCREEN hl 佔大多數      → 反轉太慢，hl_slack 偏緊
          SCREEN data 佔大多數    → warm-up 仲未夠，等多陣
        """
        s = self._gate_stats

        def pct(num: int, den: int) -> str:
            return f"{num / den * 100:5.1f}%" if den > 0 else "  ---"

        logger.info(
            "GATE_DIAG  bars=%d  considered=%d (%s)  "
            "G1_z=%d (%s)  G2_eligible=%d (%s)  G3_regime=%d (%s)  "
            "G4_|z|>=%.2f=%d (%s)  G5_no_cd=%d (%s)  G6_corr=%d (%s)  "
            "G7_fill=%d (%s)  G8_notional=%d (%s)  → OPEN=%d",
            s["bars"], s["considered"], pct(s["considered"], s["bars"]),
            s["g1_z"], pct(s["g1_z"], s["considered"]),
            s["g2_eligible"], pct(s["g2_eligible"], s["g1_z"]),
            s["g3_regime"], pct(s["g3_regime"], s["g2_eligible"]),
            self.config.z_entry_short,
            s["g4_z_thresh"], pct(s["g4_z_thresh"], s["g3_regime"]),
            s["g5_no_cd"], pct(s["g5_no_cd"], s["g4_z_thresh"]),
            s["g6_corr"], pct(s["g6_corr"], s["g5_no_cd"]),
            s["g7_fill"], pct(s["g7_fill"], s["g6_corr"]),
            s["g8_notional"], pct(s["g8_notional"], s["g7_fill"]),
            s["opens"],
        )

        # ── |z| 分布（G3 通過嘅 bar）────────────────────────────────────────
        zw = self._z_abs_window
        if zw:
            arr = np.asarray(zw, dtype=float)
            zmax = float(arr.max())
            zmean = float(arr.mean())
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            n_ge_thresh = int((arr >= self.config.z_entry_short).sum())
            logger.info(
                "ZDIST  n=%d  mean|z|=%.2f  p95=%.2f  p99=%.2f  max=%.2f  "
                "n(|z|>=%.2f)=%d (%.1f%%)",
                len(arr), zmean, p95, p99, zmax,
                self.config.z_entry_short, n_ge_thresh,
                n_ge_thresh / len(arr) * 100,
            )
        else:
            logger.info("ZDIST  n=0 (no G3-pass bars in this window)")

        # ── screen_coin breakdown ──────────────────────────────────────────
        sr = self._screen_reasons
        total_screens = sum(sr.values())
        if total_screens > 0:
            logger.info(
                "SCREEN_BREAKDOWN  total=%d  ok=%d (%s)  vr=%d (%s)  "
                "hl=%d (%s)  data=%d (%s)",
                total_screens,
                sr.get("ok", 0), pct(sr.get("ok", 0), total_screens),
                sr.get("vr", 0), pct(sr.get("vr", 0), total_screens),
                sr.get("hl", 0), pct(sr.get("hl", 0), total_screens),
                sr.get("data", 0), pct(sr.get("data", 0), total_screens),
            )
        else:
            logger.info("SCREEN_BREAKDOWN  no re-screen events in this window")

        # Reset window
        self._gate_stats = {k: 0 for k in self._GATE_KEYS}
        self._z_abs_window = []
        self._screen_reasons = {"data": 0, "vr": 0, "hl": 0, "ok": 0}

    def summary(self) -> dict:
        return self.account.summary()
