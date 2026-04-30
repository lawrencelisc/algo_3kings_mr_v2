#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_mr_live.py — 均值回歸 Bot 真實/紙上交易入口

架構遵從 Opus 4.6 建議：
  Layer 1  幣池篩選：Hurst + Half-life（只兩關，避免 overfit）
  Layer 2  信號：Kalman Filter Z-Score + Microprice（model-based）
  Layer 3  Regime：VPIN + Vol Burst + OBI Accel（rule-based voting）
  Layer 4  SL：分段 time-decay（heuristic，唔係 curve fitting）
  Layer 5  Sizing：Quarter-Kelly + Hurst adj + shrinkage

手續費（HL 截圖確認）：
  Maker 0.0384%，Taker 0.0400%
  來回 breakeven ≈ 7.84 bps

環境變數（靜態模式）：
  HYPERLIQUID_WALLET          錢包地址（live 模式必填）
  HYPERLIQUID_PRIVATE_KEY     私鑰（live 模式必填）
  MR_PAPER                    "1" = 紙上交易（default），"0" = live
  MR_EQUITY_USDT              初始資金，default 1000
  MR_SYMBOLS                  逗號分隔候選池（MR_UNIVERSE_SCAN=0 時用）
  MR_POLL_SEC                 輪詢間隔，default 60（1 分鐘 bar）
  MR_Z_ENTRY                  入場 Z 閾值，default 2.0
  MR_TIMEOUT_BARS             Timeout bar 數，default 30
  MR_ATR_MULT                 SL ATR 倍數，default 1.0
  MR_LOG_FILE                 CSV 路徑，default mr_trades.csv

環境變數（動態 Universe 模式）：
  MR_UNIVERSE_SCAN            "1" = 動態選幣（default "0" = 用 MR_SYMBOLS）
  MR_UNIVERSE_TOP_K           最多同時跑幾多隻，default 8
  MR_UNIVERSE_MIN_VOL         最低 24h quoteVolume (USDC)，default 3_000_000
  MR_UNIVERSE_SCAN_TOP_N      先按 volume 取幾多隻做 Hurst 計算，default 30
  MR_UNIVERSE_RESCAN_H        每幾小時重新掃描，default 4
  MR_UNIVERSE_OHLCV_BARS      Universe 計 Hurst 用幾多根 1m bar，default 500
  MR_UNIVERSE_RELAX_HURST     Relaxed fallback：優先 H 低於此值，default 0.55
  MR_UNIVERSE_MIN_ATR_BPS     ATR(14)/價最低 bps（排除死市），default 8
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── sys.path 修正 ──────────────────────────────────────────────────────────
# main_mr_live.py 喺 project root（algo_3kings_mr_v2/）
# mr_bot/ 係 project root 嘅子目錄
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ── .env → os.environ ─────────────────────────────────────────────────────
# Python 本身唔讀 .env；PyCharm Run Configuration 有時會留下舊嘅環境變數，
# 導致你改咗 .env 但 log 仍然 universe_scan=False。
# load_dotenv(override=True) 令 project root 嘅 .env 覆蓋已存在嘅同名變數。
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

_ENV_FILE = os.path.join(_THIS_DIR, ".env")
if load_dotenv and os.path.isfile(_ENV_FILE):
    load_dotenv(_ENV_FILE, override=True)
# ──────────────────────────────────────────────────────────────────────────

import ccxt  # type: ignore
import numpy as np

from mr_bot.bot import MeanReversionBot, BotConfig
from mr_bot.core.indicators import microprice, variance_ratio, half_life_ou
from mr_bot.core.regime import screen_coin

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_mr")

# ── Default Symbol List（靜態模式 fallback）──────────────────────────────────
DEFAULT_SYMBOLS = (
    "HYPE/USDC:USDC,"
    "TAO/USDC:USDC,"
    "TRUMP/USDC:USDC,"
    "AXS/USDC:USDC,"
    "AAVE/USDC:USDC,"
    "XRP/USDC:USDC"
)


# ── Exchange Helper ──────────────────────────────────────────────────────────

def _make_exchange() -> ccxt.Exchange:
    wallet = os.environ.get("HYPERLIQUID_WALLET", "").strip()
    key    = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "").strip()
    paper  = os.environ.get("MR_PAPER", "1").strip() == "1"

    params = {
        "enableRateLimit": True,
        "timeout": 15000,
    }
    if paper:
        logger.warning("⚡ PAPER TRADE MODE — 無真實下單")
        params["walletAddress"] = wallet or "0x0000000000000000000000000000000000000000"
        params["privateKey"]    = key or "0" * 64
    else:
        if not wallet or not key:
            raise RuntimeError("Live mode 需要 HYPERLIQUID_WALLET + HYPERLIQUID_PRIVATE_KEY")
        logger.warning("🔴 LIVE TRADE MODE — 真實資金")
        params["walletAddress"] = wallet
        params["privateKey"]    = key
    return ccxt.hyperliquid(params)


# ── ATR ──────────────────────────────────────────────────────────────────────

def compute_atr(ohlcv: List[List[float]], period: int = 14) -> float:
    if len(ohlcv) < period + 1:
        return ohlcv[-1][2] - ohlcv[-1][3] if ohlcv else 1.0
    trs = []
    for i in range(1, len(ohlcv)):
        h, l, pc = ohlcv[i][2], ohlcv[i][3], ohlcv[i - 1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs[-period:]))


# ── Order Book Parse ─────────────────────────────────────────────────────────

def parse_ob(ob: Dict) -> Tuple[float, float, float, float, List, List]:
    bids = ob.get("bids", [])[:5]
    asks = ob.get("asks", [])[:5]
    bid      = float(bids[0][0]) if bids else 0.0
    bid_size = sum(float(b[1]) for b in bids)
    ask      = float(asks[0][0]) if asks else 0.0
    ask_size = sum(float(a[1]) for a in asks)
    return bid, bid_size, ask, ask_size, \
           [(float(b[0]), float(b[1])) for b in bids], \
           [(float(a[0]), float(a[1])) for a in asks]


# ── Liquidation Spike Detection ──────────────────────────────────────────────

class LiquidationTracker:
    def __init__(self, window: int = 10, spike_threshold: float = 3.0) -> None:
        from collections import deque
        self._oi_history  = deque(maxlen=window)
        self.spike_threshold = spike_threshold

    def update(self, oi: float) -> bool:
        self._oi_history.append(oi)
        if len(self._oi_history) < 5:
            return False
        arr     = list(self._oi_history)
        mean_oi = float(np.mean(arr[:-1]))
        std_oi  = float(np.std(arr[:-1]))
        if std_oi <= 0:
            return False
        return abs(arr[-1] - mean_oi) / std_oi > self.spike_threshold


# ── Universe Scanner ─────────────────────────────────────────────────────────

_UNIVERSE_EXCLUDE_PREFIXES = ("XYZ-",)   # 合成資產（tokenized stocks/commodities）
_UNIVERSE_EXCLUDE_EXACT   = frozenset({  # 個別排除：穩定/掛鉤幣，波幅不足
    "PAXG/USDC:USDC", "USDT/USDC:USDC", "USDC/USDC:USDC",
    "DAI/USDC:USDC", "TUSD/USDC:USDC", "FDUSD/USDC:USDC",
})


def scan_universe(
    ex: ccxt.Exchange,
    min_vol_24h: float,
    scan_top_n: int,
    top_k: int,
    timeout_bars: int,
    hurst_threshold: float = 0.45,
    hl_slack: float = 1.5,
    ohlcv_limit: int = 500,
    relax_hurst_max: float = 0.55,
    min_atr_bps: float = 8.0,   # ATR(14)/close ×1e4；1m「每 bar 平均 range」尺度錯會殺晒池
) -> List[str]:
    """
    掃描全 Hyperliquid perps，返回最 mean-reverting 嘅 top_k 隻幣。

    步驟：
      1. fetch_tickers() → 所有 perp 24h stats（1 次 API call）
      2. 過濾：USDC perp + quoteVolume > min_vol_24h
      3. 按 volume 降序，取 top scan_top_n 隻做 Hurst 計算
      4. 每隻 fetch 1m OHLCV → ATR gate（排除死市）→ hurst_rs() + half_life_ou()
      5. **Strict**：H < hurst_threshold + HL OK → 理想 mean-reversion pool
      6. 若 strict 一隻都冇（crypto 常見 — 短視窗 AR(1) 全被判 trending）：
         **Relaxed**：喺高成交量候選裡面按 Hurst 升序揀 top_k（仍優先 H < relax_hurst_max）。
         入場仍由 bot 嘅 screen_coin + eligible 把關，唔會盲開倉。
    """
    logger.info("━━━ UNIVERSE SCAN START ━━━")

    # Step 1: 拉全部 ticker
    try:
        all_tickers = ex.fetch_tickers()
    except Exception as e:
        logger.warning("Universe scan: fetch_tickers failed: %s", e)
        return []

    # Step 2: filter USDC perps by volume + exclude synthetic / stable assets
    candidates: List[Tuple[str, float]] = []
    for sym, t in all_tickers.items():
        if not sym.endswith(":USDC"):
            continue
        # Exclude synthetic tokenized assets (XYZ-*) and known stables/pegged
        base = sym.split("/")[0]
        if any(base.startswith(pfx) for pfx in _UNIVERSE_EXCLUDE_PREFIXES):
            continue
        if sym in _UNIVERSE_EXCLUDE_EXACT:
            continue
        vol = float(t.get("quoteVolume") or 0)
        if vol < min_vol_24h:
            continue
        candidates.append((sym, vol))

    candidates.sort(key=lambda x: x[1], reverse=True)
    fetch_pool = [sym for sym, _ in candidates[:scan_top_n]]

    logger.info(
        "Universe: %d perps pass vol≥%.0f filter → fetching OHLCV (%d bars) for top %d by volume",
        len(candidates), min_vol_24h, ohlcv_limit, len(fetch_pool),
    )

    # Step 3-4: fetch OHLCV + compute Hurst / half-life
    scored: List[Tuple[str, float, Optional[float]]] = []
    for sym in fetch_pool:
        try:
            ohlcv = [list(x) for x in ex.fetch_ohlcv(sym, "1m", limit=ohlcv_limit)]
            time.sleep(0.4)
        except Exception as e:
            logger.debug("Universe OHLCV skip %s: %s", sym, e)
            continue

        if len(ohlcv) < 50:
            continue

        closes = [float(b[4]) for b in ohlcv]

        # Volatility gate：ATR(14)/price（bps）。之前用「每根 1m bar 平均 (H-L)/C」
        # 要求 30 bps —— 1m bar 通常只得幾 bps，會排晒所有幣。
        atr_bps = compute_atr(ohlcv, 14) / max(closes[-1], 1e-9) * 1e4
        if atr_bps < min_atr_bps:
            logger.debug(
                "Universe  %-22s  skip: ATR14/px=%.1f bps < %.0f bps min",
                sym, atr_bps, min_atr_bps,
            )
            continue

        vr = variance_ratio(closes, q=5)
        hl = half_life_ou(closes)
        if vr is None:
            continue

        h_equiv = vr * 0.5   # 映射到 Hurst-like 值供顯示
        scored.append((sym, vr, h_equiv, hl))
        logger.debug(
            "Universe  %-22s  ATR14=%.0fbps  VR=%.3f (H≈%.3f)  HL=%s",
            sym, atr_bps, vr, h_equiv, f"{hl:.1f}" if hl else "None",
        )

    if not scored:
        logger.warning(
            "Universe scan: empty pool (all skipped OHLCV/ATR gate/hurst). "
            "唔係「市場冇好選擇」—多數係 filter 太緊；可試 MR_UNIVERSE_MIN_ATR_BPS=6 或大啲 MR_UNIVERSE_SCAN_TOP_N",
        )
        return []

    # Step 5: VR filter（VR < vr_threshold 即 mean reverting）
    vr_threshold = hurst_threshold * 2.0  # 0.45 → 0.90
    hl_cap = timeout_bars * hl_slack
    eligible = [
        (sym, vr, h_equiv, hl) for sym, vr, h_equiv, hl in scored
        if vr < vr_threshold
        and hl is not None
        and hl <= hl_cap
    ]
    eligible.sort(key=lambda x: x[1])   # ascending VR（最 mean-reverting 優先）

    selected = [sym for sym, *_ in eligible[:top_k]]
    relaxed  = False

    if not selected:
        # ── Relaxed fallback（HL 估計有誤差，strict 易全滅）──────────────────
        relaxed = True
        scored.sort(key=lambda x: x[1])   # ascending VR
        preferred = [(s, vr, h, hl) for s, vr, h, hl in scored if vr < relax_hurst_max * 2.0]
        pool = preferred if preferred else scored
        selected = [sym for sym, *_ in pool[:top_k]]
        logger.warning(
            "Universe: strict VR filter matched 0 coins (typical in trending regimes). "
            "RELAXED fallback → top-%d by lowest VR among liquid perps. "
            "Entry still gated by bot screen_coin / eligible.",
            len(selected),
        )
        for sym, vr, h, hl in pool[:top_k]:
            logger.info(
                "Universe RELAXED pick %-22s  VR=%.3f (H≈%.3f)  HL=%s",
                sym, vr, h, f"{hl:.1f}" if hl else "None",
            )

    logger.info(
        "━━━ UNIVERSE SCAN DONE  strict=%s  selected=%d: %s ━━━",
        not relaxed, len(selected), selected,
    )
    for sym, vr, h, hl in eligible[:top_k]:
        logger.info("  ✓ %-22s  VR=%.3f (H≈%.3f)  HL=%.1f bars",
                    sym, vr, h, hl or 0)
    return selected


# ── Warm-up Single Symbol ────────────────────────────────────────────────────

def warmup_symbol(
    bot: MeanReversionBot,
    ex: ccxt.Exchange,
    sym: str,
    liq_trackers: Dict[str, LiquidationTracker],
    prices_cache: Dict[str, List[float]],
    ohlcv_cache: Dict[str, List],
    config: BotConfig,
) -> bool:
    """
    拉 600 根歷史 bar replay 進 bot，warm up Kalman / VPIN / VolBurst / Hurst。
    prices_for_screen=None → 唔觸發 eligibility → 唔會產生假交易。
    Replay 後手動 screen_coin 設定初始 eligibility。
    返回 True = 成功，False = 失敗（幣唔存在 / 資料不足）。
    """
    logger.info("Warm-up  %-22s  fetching 600 bars …", sym)
    try:
        wu_ohlcv = [list(x) for x in ex.fetch_ohlcv(sym, "1m", limit=600)]
        time.sleep(0.8)
    except Exception as _e:
        logger.warning("Warm-up fetch error %s: %s — skipped", sym, _e)
        return False

    if len(wu_ohlcv) < 50:
        logger.warning("Warm-up skip %s — only %d bars", sym, len(wu_ohlcv))
        return False

    ohlcv_cache[sym] = wu_ohlcv
    if sym not in prices_cache:
        prices_cache[sym] = []
    if sym not in liq_trackers:
        liq_trackers[sym] = LiquidationTracker()

    for _i in range(1, len(wu_ohlcv)):
        _bar      = wu_ohlcv[_i]
        _close    = float(_bar[4])
        _high     = float(_bar[2])
        _low      = float(_bar[3])
        _volume   = float(_bar[5])
        _spread   = max(_high - _low, _close * 1e-5) * 0.1
        _bid      = _close - _spread
        _ask      = _close + _spread
        _sz       = max(_volume * 0.5, 1e-9)
        _pchg     = _close - float(wu_ohlcv[_i - 1][4])
        _buy_flow = _volume * 0.5 * (1 + (1 if _pchg > 0 else -1) * 0.3)
        _atr      = compute_atr(wu_ohlcv[: _i + 1], 14)

        prices_cache[sym].append(_close)
        if len(prices_cache[sym]) > 600:
            prices_cache[sym] = prices_cache[sym][-600:]

        liq_trackers[sym].update(0.0)

        bot.on_bar(
            symbol=sym,
            close=_close, high=_high, low=_low, volume=_volume,
            bid=_bid, ask=_ask, bid_size=_sz, ask_size=_sz,
            bid_levels=[(_bid, _sz)], ask_levels=[(_ask, _sz)],
            atr=_atr,
            buy_flow=_buy_flow, sell_flow=_volume - _buy_flow,
            liq_spike=False,
            prices_for_screen=None,   # 唔觸發 eligibility
        )

    # 手動 screen，設定初始 eligibility
    # warm-up 用 1m bar replay，half_life_ou 返回「1m bars」數（即分鐘）。
    # 但 config.timeout_bars 係以 bar_duration_sec 為單位（e.g. 3m bar）。
    # 必須換算：timeout_1m = timeout_bars × (bar_duration_sec / 60)
    # 例：20 bars × (180s / 60) = 60 1m bars；cap = 60 × 1.5 = 90 分鐘
    _timeout_1m = max(1, int(config.timeout_bars * (config.bar_duration_sec / 60.0)))
    _st = bot.get_state(sym)
    if len(prices_cache[sym]) >= 50:
        _res = screen_coin(
            prices=prices_cache[sym],
            timeout_bars=_timeout_1m,
            hurst_threshold=config.hurst_threshold,
            hl_slack=config.hl_slack,
        )
        _st.eligible               = _res.eligible
        _st.hurst_val              = _res.hurst or _st.hurst_val
        _st.half_life_bars         = _res.half_life_bars
        _st.eligibility_checked_at = _st.bar_count

    logger.info(
        "Warm-up done %-22s  bars=%d  eligible=%-5s  H=%s  HL=%s",
        sym, len(wu_ohlcv), _st.eligible,
        f"{_st.hurst_val:.3f}" if _st.hurst_val is not None else "None",
        f"{_st.half_life_bars:.1f}" if _st.half_life_bars is not None else "None",
    )
    return True


# ── Live Order Helpers ───────────────────────────────────────────────────────
#
# 設計原則：
#   入場  → postOnly limit（maker fee 0.0384%）；若 market crossed 被 reject，
#            fallback 到 market（taker 0.0400%，只差 0.16 bps，可接受）。
#            入場成功後立即落 bracket（TP limit + SL stop-market，均 reduceOnly）。
#   出場  → 先 cancel bracket orders，再：
#            緊急出場（SL / REGIME_RED / ADVERSE_FLOW）用 market + reduceOnly；
#            正常出場（TP / DECEL / TIMEOUT）用 limit + reduceOnly + postOnly。
#   Bracket → 若 SL 或 TP bracket 落單失敗，只 WARNING log；bot 內部邏輯仍係後備。
#   Retry  → 最多 3 次，間隔 1s；三次都失敗 → ERROR log，但唔 crash bot。
#
# _live_orders: {symbol: {"tp": order_id | None, "sl": order_id | None}}
#   追蹤每個 symbol 現有嘅 bracket orders，出場時用嚟 cancel。
#
_TAKER_REASONS = frozenset({"SL", "REGIME_RED", "ADVERSE_FLOW"})
_live_orders: Dict[str, Dict[str, Optional[str]]] = {}


def _live_wait_fill(
    ex: ccxt.Exchange,
    symbol: str,
    order_id: str,
    timeout_secs: float = 10.0,
    poll_interval: float = 0.5,
) -> bool:
    """
    Poll fetch_order 直到 order 成交（closed/filled）或超時。
    用於 postOnly resting orders：必須等到 position 真實存在，
    才能落 reduceOnly bracket orders。
    返回 True = 已成交；False = 超時或取消。
    """
    elapsed = 0.0
    while elapsed < timeout_secs:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            o = ex.fetch_order(order_id, symbol)
            status = o.get("status", "")
            if status in ("closed", "filled"):
                return True
            if status in ("canceled", "cancelled", "rejected", "expired"):
                logger.info(
                    "LIVE_WAIT_FILL %s order_id=%s status=%s → skip bracket",
                    symbol, order_id, status,
                )
                return False
        except Exception as e:
            logger.debug("LIVE_WAIT_FILL fetch error %s: %s", symbol, e)
    logger.warning(
        "LIVE_WAIT_FILL %s order_id=%s timeout after %.0fs → skip bracket (internal SL/TP active)",
        symbol, order_id, timeout_secs,
    )
    return False


def _live_place_bracket(
    ex: ccxt.Exchange,
    symbol: str,
    state,  # type: ignore[no-untyped-def]
) -> None:
    """
    Entry order 確認成交、position 已存在後，落 bracket orders（均 reduceOnly）。

    TP  → limit + postOnly @ state.tp_price  (爭取 maker fill)
    SL  → stop_market，triggerPrice @ state.sl_price  (止蝕保護，taker 可接受)

    呼叫前必須確保 position 已在交易所存在，否則 reduceOnly 會被 reject。
    若 SL 或 TP 任何一個失敗，只 WARNING — bot 內部 3min bar SL/TP 仍係後備。
    Order IDs 寫入 _live_orders[symbol] 供 _live_cancel_bracket 用。
    """
    exit_side = "sell" if state.position_side == "long" else "buy"
    orders: Dict[str, Optional[str]] = {"tp": None, "sl": None}

    try:
        amount_str = ex.amount_to_precision(symbol, state.size)
    except Exception as e:
        logger.warning("LIVE_BRACKET precision error %s: %s", symbol, e)
        return

    # ── TP：limit + postOnly + reduceOnly @ tp_price ──────────────────────
    try:
        tp_px = ex.price_to_precision(symbol, state.tp_price)
        tp_order = ex.create_order(
            symbol, "limit", exit_side, float(amount_str), float(tp_px),
            {"reduceOnly": True, "postOnly": True},
        )
        orders["tp"] = tp_order.get("id")
        logger.info(
            "LIVE_TP  %s %s  qty=%s  px=%s  order_id=%s",
            exit_side.upper(), symbol, amount_str, tp_px, orders["tp"],
        )
    except Exception as e:
        logger.warning("LIVE_TP failed %s: %s (internal TP still active)", symbol, e)

    # ── SL：stop_market + reduceOnly，trigger @ sl_price ──────────────────
    try:
        sl_px = ex.price_to_precision(symbol, state.sl_price)
        sl_order = ex.create_order(
            symbol, "stop_market", exit_side, float(amount_str), float(sl_px),
            {"reduceOnly": True, "triggerPrice": float(sl_px), "slippage": 0.05},
        )
        orders["sl"] = sl_order.get("id")
        logger.info(
            "LIVE_SL  %s %s  qty=%s  trigger=%s  order_id=%s",
            exit_side.upper(), symbol, amount_str, sl_px, orders["sl"],
        )
    except Exception as e:
        logger.warning("LIVE_SL failed %s: %s (internal SL still active)", symbol, e)

    _live_orders[symbol] = orders


def _live_cancel_bracket(ex: ccxt.Exchange, symbol: str) -> None:
    """
    出場前 cancel 該 symbol 嘅 bracket orders（TP + SL）。
    若 order 已經成交或唔存在，catch exception 並 debug log（唔 crash）。
    """
    orders = _live_orders.pop(symbol, {})
    for order_type, oid in orders.items():
        if oid:
            try:
                ex.cancel_order(oid, symbol)
                logger.info(
                    "LIVE_CANCEL_%s  %s  order_id=%s",
                    order_type.upper(), symbol, oid,
                )
            except Exception as e:
                logger.debug(
                    "LIVE_CANCEL_%s %s order_id=%s: %s (may already be filled/cancelled)",
                    order_type.upper(), symbol, oid, e,
                )


def _live_min_check(ex: ccxt.Exchange, symbol: str, size: float, notional: float) -> bool:
    """
    Check Hyperliquid minimum order requirements。
    返回 True = 可以下單；False = 低於最低限制，應 skip。
    用 ex.market() 取交易所實際 limits，唔 hardcode。
    """
    try:
        mkt    = ex.market(symbol)
        limits = mkt.get("limits", {})
        min_amt  = (limits.get("amount") or {}).get("min") or 0.0
        min_cost = (limits.get("cost")   or {}).get("min") or 0.0
        if min_amt  and size     < min_amt:
            logger.warning(
                "LIVE_SKIP_MIN_SIZE  %s  size=%.6f < exchange_min=%.6f",
                symbol, size, min_amt,
            )
            return False
        if min_cost and notional < min_cost:
            logger.warning(
                "LIVE_SKIP_MIN_NOTIONAL  %s  notional=%.4f < exchange_min=%.4f",
                symbol, notional, min_cost,
            )
            return False
    except Exception as e:
        logger.debug("LIVE_MIN_CHECK skip (market info unavailable) %s: %s", symbol, e)
    return True


def _live_place_entry(
    ex: ccxt.Exchange,
    symbol: str,
    state,  # type: ignore[no-untyped-def]
    ob_bids: Optional[List] = None,
    ob_asks: Optional[List] = None,
) -> None:
    """
    Bot 剛剛虛擬開倉，mirror 到真實交易所。

    入場策略（用最新 5 層 OB 決定）：

      LONG（買貨）→ 直接 IOC market @ best_ask
        理由：price 已超賣，需要快手搶入，唔等 maker。

      SHORT（賣貨）→ 先睇 5 層 bid 深度：
        bid depth（5層）≥ 我們嘅 notional  → 嘗試 postOnly limit @ best_ask
          best_ask = 當前最優 ask，作為我嘅賣出底線（maker）
          若 postOnly 成功 → maker fill，省 taker fee
          若 postOnly 被 reject（bid 已高過 ask）→ IOC market @ best_bid
        bid depth 唔夠                         → 直接 IOC market @ best_bid
    """
    # ── 最低下單量 guard ──────────────────────────────────────────────────
    if not _live_min_check(ex, symbol, state.size, state.notional):
        return

    try:
        amount_str = ex.amount_to_precision(symbol, state.size)
    except Exception as e:
        logger.error("LIVE_ENTRY precision error %s: %s", symbol, e)
        return

    # ── OB 參考價（用最新 OB，唔用 stale microprice）──────────────────────
    best_bid = ob_bids[0][0] if ob_bids else state.entry_price
    best_ask = ob_asks[0][0] if ob_asks else state.entry_price

    def _market_ioc(order_side: str, ref_price: float, tag: str) -> bool:
        """
        落 market/IOC order（HL 需要 price 作 slippage 參考）。
        返回 True = 成功。市價單在 create_order 返回時已成交，
        sleep 1s 讓 position 在交易所完全同步後，再落 bracket。
        """
        try:
            px = ex.price_to_precision(symbol, ref_price)
            order = ex.create_order(
                symbol, "market", order_side, float(amount_str),
                float(px), {"slippage": 0.05},
            )
            logger.info(
                "LIVE_ENTRY  %s %s  qty=%s  px=%s  type=market(%s)  order_id=%s",
                order_side.upper(), symbol, amount_str, px, tag,
                order.get("id", "?"),
            )
            # Position propagation delay：market order 雖然即時成交，
            # 但交易所 position state 需要約 1s 同步，才能接受 reduceOnly orders。
            time.sleep(1)
            return True
        except Exception as e:
            logger.warning("LIVE_ENTRY market(%s) failed %s: %s", tag, symbol, e)
            return False

    # ── LONG：直接 IOC @ best_ask ─────────────────────────────────────────
    if state.position_side == "long":
        for attempt in range(3):
            if _market_ioc("buy", best_ask, "IOC"):
                _live_place_bracket(ex, symbol, state)
                return
            if attempt < 2:
                time.sleep(1)
        logger.error("LIVE_ENTRY LONG all attempts failed %s", symbol)
        return

    # ── SHORT：睇 5 層 bid depth → postOnly @ best_ask；唔夠 → IOC ─────────
    bid_depth_notional = sum(p * s for p, s in (ob_bids or [])[:5])
    has_depth = bid_depth_notional >= state.notional   # 買方深度 ≥ 我們訂單

    if has_depth and best_ask > 0:
        try:
            px = ex.price_to_precision(symbol, best_ask)
            order = ex.create_order(
                symbol, "limit", "sell", float(amount_str), float(px),
                {"postOnly": True},
            )
            oid = order.get("id")
            logger.info(
                "LIVE_ENTRY  SELL %s  qty=%s  px=%s  type=postOnly(maker)  "
                "bid_depth=%.2f  order_id=%s",
                symbol, amount_str, px, bid_depth_notional, oid,
            )
            # postOnly resting：必須等訂單真實成交，position 存在後才落 bracket。
            # poll fetch_order（最多 10s），成交後立即落 bracket。
            if oid and _live_wait_fill(ex, symbol, oid):
                _live_place_bracket(ex, symbol, state)
            else:
                logger.warning(
                    "LIVE_ENTRY SHORT %s postOnly not filled → bracket skipped "
                    "(internal SL/TP active)", symbol,
                )
            return
        except ccxt.InvalidOrder as e:
            logger.info(
                "LIVE_ENTRY postOnly rejected %s (market moved) → IOC  [%s]",
                symbol, e,
            )
        except Exception as e:
            logger.warning("LIVE_ENTRY postOnly error %s: %s", symbol, e)
    else:
        logger.info(
            "LIVE_ENTRY SHORT %s bid_depth=%.2f < notional=%.2f → skip postOnly, IOC",
            symbol, bid_depth_notional, state.notional,
        )

    # postOnly 唔成功 / depth 唔夠 → IOC @ best_bid
    for attempt in range(3):
        if _market_ioc("sell", best_bid, "IOC"):
            _live_place_bracket(ex, symbol, state)
            return
        if attempt < 2:
            time.sleep(1)
    logger.error("LIVE_ENTRY SHORT all attempts failed %s", symbol)


def _live_place_exit(ex: ccxt.Exchange, symbol: str, rec) -> None:  # type: ignore[no-untyped-def]
    """
    Bot 剛剛虛擬平倉，mirror 到真實交易所。
    rec.side        = 持倉方向（"long"|"short"），exit 係反方向
    rec.reason      = 出場原因（決定 taker vs maker）
    rec.size        = 合約數量
    rec.exit_price  = bot 嘅出場估算價（limit 出場用）
    """
    # 先 cancel bracket（TP + SL）；若已成交或不存在，debug log 不 crash。
    _live_cancel_bracket(ex, symbol)

    exit_side  = "sell" if rec.side == "long" else "buy"
    use_market = rec.reason in _TAKER_REASONS

    try:
        amount_str = ex.amount_to_precision(symbol, rec.size)
        price_str  = ex.price_to_precision(symbol, rec.exit_price)
    except Exception as e:
        logger.error("LIVE_EXIT precision error %s: %s", symbol, e)
        return

    for attempt in range(3):
        try:
            if use_market:
                # Hyperliquid market order 必須傳 price 作 slippage 參考
                order = ex.create_order(
                    symbol, "market", exit_side, float(amount_str),
                    float(price_str),
                    {"reduceOnly": True, "slippage": 0.05},
                )
            else:
                order = ex.create_order(
                    symbol, "limit", exit_side,
                    float(amount_str), float(price_str),
                    {"reduceOnly": True, "postOnly": True},
                )
            logger.info(
                "LIVE_EXIT  %s %s  reason=%s  qty=%s  px=%s  type=%s  order_id=%s",
                exit_side.upper(), symbol, rec.reason,
                amount_str, price_str,
                "market" if use_market else "limit",
                order.get("id", "?"),
            )
            return
        except ccxt.InvalidOrder as e:
            err_str = str(e)
            # "Reduce only order would increase position" = 倉位已不存在
            # （bracket TP/SL 比 bot 快一步平了倉，或之前入場失敗的 ghost position）
            # 此為預期情況，無需重試，靜默記錄即可。
            if "would increase position" in err_str or "increase position" in err_str:
                logger.info(
                    "LIVE_EXIT  %s  reason=%s → position already closed "
                    "(bracket TP/SL filled first or ghost position); no action needed.",
                    symbol, rec.reason,
                )
                return
            if not use_market:
                logger.warning(
                    "LIVE_EXIT postOnly rejected %s reason=%s → fallback market  [%s]",
                    symbol, rec.reason, e,
                )
                use_market = True
            else:
                logger.error("LIVE_EXIT market fallback rejected %s: %s", symbol, e)
                return
        except Exception as e:
            logger.warning("LIVE_EXIT attempt %d failed %s: %s", attempt + 1, symbol, e)
            if attempt < 2:
                time.sleep(1)

    logger.error("LIVE_EXIT all attempts failed %s reason=%s", symbol, rec.reason)


# ── fetch_one helper ──────────────────────────────────────────────────────────
# Main loop 只需最新幾根 bar 計 ATR + 最新 close/volume。
# 歷史 600 bars 已喺 warm-up 階段拉咗，主循環唔需要重複拉。
# limit=20 足以計 ATR(14)（需要 15+ bars），且速度提升 ~30x vs limit=600。

def _fetch_one(ex: ccxt.Exchange, sym: str) -> Optional[Dict]:
    for attempt in range(4):
        try:
            ohlcv  = [list(x) for x in ex.fetch_ohlcv(sym, "1m", limit=20)]
            time.sleep(0.2)
            ticker = ex.fetch_ticker(sym)
            time.sleep(0.2)
            ob     = ex.fetch_order_book(sym, 5)
            return {"sym": sym, "ohlcv": ohlcv, "ticker": ticker, "ob": ob}
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                wait = 2 ** attempt + 1
                logger.warning("429 %s attempt %d — sleep %.0fs", sym, attempt + 1, wait)
                time.sleep(wait)
            else:
                logger.warning("Fetch error %s: %s", sym, e)
                return None
    return None


# ── Main Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    paper        = os.environ.get("MR_PAPER", "1").strip() == "1"
    poll_sec     = float(os.environ.get("MR_POLL_SEC", "60"))
    csv_path     = os.environ.get("MR_LOG_FILE", "mr_trades.csv")
    z_entry      = float(os.environ.get("MR_Z_ENTRY", "2.0"))
    timeout_bars = int(os.environ.get("MR_TIMEOUT_BARS", "30"))
    atr_mult     = float(os.environ.get("MR_ATR_MULT", "1.0"))
    equity       = float(os.environ.get("MR_EQUITY_USDT", "1000"))

    # Position sizing fractions（% of equity per trade）
    # Hyperliquid 最低 notional = $10。
    # 公式：min_fraction × equity ≥ 10  →  min_fraction ≥ 10 / equity
    # 例：equity=$150  → min_fraction ≥ 6.67%，建議設 0.07（7%）
    # 例：equity=$500  → min_fraction ≥ 2.0%，可設 0.025
    # 例：equity=$2000 → min_fraction ≥ 0.5%，用預設 0.005
    # max_fraction 建議 = min_fraction × 3～5（允許 Kelly 放大）
    min_fraction = float(os.environ.get("MR_MIN_FRACTION", "0.005"))
    max_fraction = float(os.environ.get("MR_MAX_FRACTION", "0.05"))

    # Universe scan params
    universe_scan    = os.environ.get("MR_UNIVERSE_SCAN", "0").strip() == "1"
    universe_top_k   = int(os.environ.get("MR_UNIVERSE_TOP_K", "8"))
    universe_min_vol = float(os.environ.get("MR_UNIVERSE_MIN_VOL", "3000000"))
    universe_scan_n  = int(os.environ.get("MR_UNIVERSE_SCAN_TOP_N", "30"))
    universe_rescan_sec = float(os.environ.get("MR_UNIVERSE_RESCAN_H", "4")) * 3600
    universe_ohlcv_limit = int(os.environ.get("MR_UNIVERSE_OHLCV_BARS", "500"))
    universe_relax_hurst = float(os.environ.get("MR_UNIVERSE_RELAX_HURST", "0.55"))
    universe_min_atr_bps = float(os.environ.get("MR_UNIVERSE_MIN_ATR_BPS", "8"))

    # Align live screen_coin VR threshold with universe scan's relax threshold.
    # 之前 universe scan relaxed 用 VR<relax_hurst×2.0 揀池，但 live 每 30 bars
    # re-screen 用 BotConfig.hurst_threshold=0.45 → VR<0.90 strict，導致
    # universe relaxed 揀返嚟嘅幣即時被打成 eligible=False，永遠唔開倉。
    # 直接將 relax 值傳落 BotConfig，兩邊用同一道閘。
    from mr_bot.core.regime import SizingConfig as _SizingConfig
    config = BotConfig(
        initial_equity=equity,
        z_entry_long=-z_entry,
        z_entry_short=z_entry,
        timeout_bars=timeout_bars,
        atr_mult=atr_mult,
        bar_duration_sec=poll_sec,
        csv_path=csv_path,
        hurst_threshold=universe_relax_hurst,
        sizing_config=_SizingConfig(
            min_fraction=min_fraction,
            max_fraction=max_fraction,
        ),
    )
    logger.info(
        "VR threshold aligned: screen_coin uses VR<%.2f (from MR_UNIVERSE_RELAX_HURST=%.2f)",
        universe_relax_hurst * 2.0, universe_relax_hurst,
    )
    logger.info(
        "Sizing: equity=%.2f  min_frac=%.1f%%  max_frac=%.1f%%  "
        "→ min_notional=%.2f  max_notional=%.2f  (HL exchange_min=10 USDT)",
        equity, min_fraction * 100, max_fraction * 100,
        equity * min_fraction, equity * max_fraction,
    )
    bot = MeanReversionBot(config)
    ex  = _make_exchange()
    ex.load_markets()

    liq_trackers: Dict[str, LiquidationTracker] = {}
    ohlcv_cache:  Dict[str, List]               = {}
    prices_cache: Dict[str, List[float]]        = {}

    logger.info(
        "MR Bot 啟動  mode=%s  universe_scan=%s  poll=%.0fs  equity=%.2f",
        "PAPER" if paper else "LIVE", universe_scan, poll_sec, equity,
    )
    logger.info(
        "手續費：maker=0.0384%%  taker=0.0400%%  breakeven=%.2f bps",
        bot.fee_model.breakeven_bps(exit_is_maker=False),
    )
    logger.info("CSV → %s", os.path.abspath(csv_path))

    # ── 初始幣池 ────────────────────────────────────────────────────────────
    if universe_scan:
        symbols = scan_universe(
            ex, universe_min_vol, universe_scan_n, universe_top_k,
            timeout_bars, config.hurst_threshold, config.hl_slack,
            ohlcv_limit=universe_ohlcv_limit,
            relax_hurst_max=universe_relax_hurst,
            min_atr_bps=universe_min_atr_bps,
        )
        if not symbols:
            logger.warning("Universe scan returned empty — falling back to MR_SYMBOLS")
            universe_scan = False

    if not universe_scan:
        symbols_raw = os.environ.get("MR_SYMBOLS", DEFAULT_SYMBOLS)
        symbols     = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    logger.info("Active symbols (%d): %s", len(symbols), symbols)

    # ── Warm-up ─────────────────────────────────────────────────────────────
    logger.info("━━━ WARM-UP START — 歷史 bar replay，請稍候 ━━━")
    bot.start_warmup()
    failed_syms: List[str] = []
    for sym in symbols:
        ok = warmup_symbol(bot, ex, sym, liq_trackers, prices_cache, ohlcv_cache, config)
        if not ok:
            failed_syms.append(sym)
    if failed_syms:
        logger.warning(
            "Warm-up: %d symbols unavailable on Hyperliquid → removed: %s",
            len(failed_syms), failed_syms,
        )
        symbols = [s for s in symbols if s not in failed_syms]
    bot.end_warmup()
    bot.reset_gate_diag()   # initial warm-up：clean slate 起跳
    logger.info("━━━ WARM-UP COMPLETE — %d symbols active, bot ready to trade ━━━", len(symbols))

    last_universe_scan_ts = time.time()

    # ── Main Loop ────────────────────────────────────────────────────────────
    while True:
        loop_start = time.time()

        # ── Periodic Universe Rescan ────────────────────────────────────────
        if universe_scan and (loop_start - last_universe_scan_ts) >= universe_rescan_sec:
            logger.info("Scheduled universe rescan …")
            new_symbols = scan_universe(
                ex, universe_min_vol, universe_scan_n, universe_top_k,
                timeout_bars, config.hurst_threshold, config.hl_slack,
                ohlcv_limit=universe_ohlcv_limit,
                relax_hurst_max=universe_relax_hurst,
                min_atr_bps=universe_min_atr_bps,
            )
            if new_symbols:
                added   = [s for s in new_symbols if s not in symbols]
                removed = [s for s in symbols if s not in new_symbols]

                if added:
                    logger.info("Universe: adding %d new symbols: %s", len(added), added)
                    bot.start_warmup()
                    added = [s for s in added
                             if warmup_symbol(bot, ex, s, liq_trackers, prices_cache, ohlcv_cache, config)]
                    bot.end_warmup()
                    if added:
                        logger.info("Universe: %d symbols warmed-up successfully", len(added))

                if removed:
                    logger.info(
                        "Universe: retiring %d symbols: %s "
                        "(existing positions will timeout naturally)",
                        len(removed), removed,
                    )
                    for sym in removed:
                        st = bot.get_state(sym)
                        st.eligible = False   # 停止新開倉，持倉照常管理

                symbols = new_symbols
                logger.info("Active symbols updated (%d): %s", len(symbols), symbols)

            last_universe_scan_ts = loop_start

        # ── Fetch Market Data ────────────────────────────────────────────────
        results: Dict[str, Dict] = {}
        for i, sym in enumerate(symbols, 1):
            logger.info("Fetching %d/%d  %s …", i, len(symbols), sym)
            data = _fetch_one(ex, sym)
            if data:
                results[data["sym"]] = data
            time.sleep(0.5)

        # ── Process Each Symbol ──────────────────────────────────────────────
        for sym in symbols:
            data = results.get(sym)
            if not data:
                continue

            fresh  = data["ohlcv"]   # 最新 20 bars（from _fetch_one limit=20）
            ticker = data["ticker"]
            ob     = data["ob"]

            if not fresh:
                continue

            # 把新 bars merge 進 ohlcv_cache（warm-up 已有 600 bars，直接 append）
            cached = ohlcv_cache.get(sym, [])
            if cached:
                last_ts = cached[-1][0]
                new_bars = [b for b in fresh if b[0] > last_ts]
                cached.extend(new_bars)
                if len(cached) > 700:
                    cached = cached[-700:]
            else:
                cached = fresh
            ohlcv_cache[sym] = cached

            ohlcv = cached   # 用完整歷史計 ATR（warm-up 後有 600+ bars）
            if len(ohlcv) < 5:
                continue

            bar    = ohlcv[-1]
            close  = float(bar[4])
            volume = float(bar[5])
            high   = float(bar[2])
            low    = float(bar[3])

            bid, bid_size, ask, ask_size, bid_lvls, ask_lvls = parse_ob(ob)
            atr_val = compute_atr(ohlcv, 14)

            prices_cache[sym].append(close)
            if len(prices_cache[sym]) > 600:
                prices_cache[sym] = prices_cache[sym][-600:]

            oi        = float(ticker.get("info", {}).get("openInterest") or 0.0)
            liq_spike = liq_trackers[sym].update(oi)

            price_chg = close - float(ohlcv[-2][4]) if len(ohlcv) >= 2 else 0.0
            buy_flow  = volume * 0.5 * (1 + (1 if price_chg > 0 else -1) * 0.3)
            sell_flow = volume - buy_flow

            # Snapshot 入場前持倉狀態，用嚟偵測新開倉
            _st_before = bot.get_state(sym)
            _was_in_pos = _st_before.in_position

            rec = bot.on_bar(
                symbol=sym,
                close=close, high=high, low=low, volume=volume,
                bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
                bid_levels=bid_lvls, ask_levels=ask_lvls,
                atr=atr_val,
                buy_flow=buy_flow, sell_flow=sell_flow,
                liq_spike=liq_spike,
                prices_for_screen=prices_cache[sym] if len(prices_cache[sym]) >= 200 else None,
            )

            if not paper:
                _st_after = bot.get_state(sym)

                # ── 新開倉：bot 剛剛由無倉 → 有倉 ──────────────────────────
                if not _was_in_pos and _st_after.in_position:
                    _live_place_entry(ex, sym, _st_after, ob_bids=bid_lvls, ob_asks=ask_lvls)

                # ── 平倉：bot 返回 TradeRecord ──────────────────────────────
                if rec is not None:
                    _live_place_exit(ex, sym, rec)

        # ── Summary（每 10 分鐘）────────────────────────────────────────────
        if int(loop_start) % 600 < int(poll_sec):
            s = bot.summary()
            logger.info(
                "SUMMARY  equity=%.2f  pnl=%+.4f  trades=%d  "
                "win_rate=%.1f%%  PF=%.3f  fee_paid=%.4f",
                s["equity"], s["pnl"], s["trades"],
                s.get("win_rate", 0), s.get("profit_factor", 0),
                s.get("total_fee_paid", 0),
            )

        # ── Sleep ────────────────────────────────────────────────────────────
        elapsed    = time.time() - loop_start
        sleep_time = max(0.0, poll_sec - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBYE !!!")
