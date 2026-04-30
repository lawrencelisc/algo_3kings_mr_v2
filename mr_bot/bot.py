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

    # Timeout（分 bar 計）
    timeout_bars: int = 30

    # SL config
    atr_mult: float = 1.0
    sl_config: DynamicSLConfig = field(default_factory=DynamicSLConfig)

    # Sizing
    sizing_config: SizingConfig = field(default_factory=SizingConfig)

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
            result = screen_coin(
                prices=prices_for_screen,
                timeout_bars=self.config.timeout_bars,
                hurst_threshold=self.config.hurst_threshold,
                hl_slack=self.config.hl_slack,
            )
            state.eligible = result.eligible
            state.hurst_val = result.hurst or state.hurst_val
            state.half_life_bars = result.half_life_bars
            state.eligibility_checked_at = state.bar_count
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

        # 7. 入場
        if (
            not state.in_position
            and z is not None
            and state.eligible
            and regime_zone != RegimeZone.RED
        ):
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

        # 8. Diag log
        if self._global_bar % self.config.log_diag_every_bars == 0:
            self._log_diag(symbol, state, z, regime_zone, regime_reason, vpin_pct, vol_ratio)

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

        # Correlation cap：防止隱性集中風險
        held_symbols = [s for s, st in self._states.items() if st.in_position]
        corr_blocked, corr_reason = self.corr_guard.check_entry(symbol, held_symbols)
        if corr_blocked:
            logger.info("CORR_BLOCK  %s %s  %s", side.upper(), symbol, corr_reason)
            return

        # Drawdown throttle：連虧保護
        if self.dd_throttle.is_throttled:
            logger.info("DD_THROTTLE  %s  %s", symbol, self.dd_throttle.status_str())

        # Limit order fill simulation（用 microprice 附近掛單）
        limit_price = mp
        fill_price, is_maker = simulate_limit_fill(side, limit_price, bid_levels, ask_levels)
        if fill_price is None:
            return

        # Base SL distance（ATR-normalized）
        base_sl = self.config.atr_mult * atr
        if base_sl <= 0:
            return

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

        size = notional / max(fill_price, 1e-9)
        # size_mult 記錄
        base_notional = self.account.equity * self.config.sizing_config.min_fraction
        size_mult = notional / max(base_notional, 1e-9)

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

        # 動態 SL 距離
        current_sl_dist = compute_sl_distance(
            base_sl=state.base_sl_distance,
            entry_time=state.entry_time,
            half_life_bars=state.half_life_bars or self.config.timeout_bars,
            bar_duration_sec=self.config.bar_duration_sec,
            config=self.config.sl_config,
            regime_zone=regime_zone,
        )

        # 更新 SL price（只收緊，唔放鬆）
        if side == "long":
            new_sl = state.entry_price - current_sl_dist
            state.sl_price = max(state.sl_price, new_sl)
        else:
            new_sl = state.entry_price + current_sl_dist
            state.sl_price = min(state.sl_price, new_sl)

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

        # DECEL（Z-score 回歸，即 mean reversion 大致完成）
        if reason is None and z is not None:
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

    def summary(self) -> dict:
        return self.account.summary()
