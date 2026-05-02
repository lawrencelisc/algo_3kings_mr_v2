"""
core/regime.py — Regime Detection + 動態幣池篩選 + 智能止損

Opus 4.6 建議：
  - 幣池篩選只用兩個核心 filter（Hurst + half-life），避免過度篩選
  - Regime Score 用 rule-based voting（唔係 weighted linear sum）
  - SL 用分段式 time-decay（heuristic，唔係 fitted curve）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from .indicators import (
    RollingHurst,
    RollingVR,
    half_life_ou,
    hurst_rs,
    variance_ratio,
    VPIN,
    VolBurstDetector,
    OBIAcceleration,
)


# ────────────────────────────────────────────────────────────────────────────
# Regime Zone
# ────────────────────────────────────────────────────────────────────────────

class RegimeZone(str, Enum):
    GREEN  = "green"   # 正常交易
    YELLOW = "yellow"  # 縮倉 50% + 收緊 SL
    RED    = "red"     # 停止開倉 + 已有倉位 aggressive exit


# ────────────────────────────────────────────────────────────────────────────
# Coin Eligibility（Hurst + Half-life 兩關，不多不少）
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CoinEligibility:
    """
    幣種適性評估結果。

    Opus 4.6 建議：
      只用 Hurst + half-life 做 hard filter，
      其他（ADF、spread CV）做 monitoring 但唔做 hard gate。
    """
    symbol: str
    hurst: Optional[float]
    half_life_bars: Optional[float]
    eligible: bool
    reason: str
    reason_code: str = ""   # "data" | "vr" | "hl" | "ok"（供 gate funnel breakdown 用）


def screen_coin(
    prices: Sequence[float],
    timeout_bars: int,
    hurst_threshold: float = 0.45,   # 向後兼容，內部轉換為 VR 門檻
    hl_slack: float = 1.5,
) -> CoinEligibility:
    """
    兩關篩選（ROUND2：改用 Variance Ratio 取代 R/S Hurst）：

      1. VR(5) < vr_threshold
         vr_threshold = hurst_threshold × 2（0.45 → 0.90）
         VR < 0.90 代表 mean reversion 夠強

      2. half_life < timeout_bars × hl_slack

    改用 VR 的原因：
      R/S Hurst 假設 price series stationary，
      crypto price 係 non-stationary（unit root），
      結果永遠 H≈1.0，所有幣都 eligible=False。
      VR 用 log-return（stationary），對 crypto 天然 robust。
    """
    arr = list(prices)

    vr_threshold = hurst_threshold * 2.0  # 0.45 → 0.90
    vr = variance_ratio(arr, q=5)
    hl = half_life_ou(arr)

    # 映射 VR 到 hurst-like 值（供 DIAG log 顯示）
    h_equiv = (vr * 0.5) if vr is not None else None

    if vr is None:
        return CoinEligibility(
            symbol="", hurst=None, half_life_bars=None,
            eligible=False, reason="insufficient_data_for_VR (need 22+ bars)",
            reason_code="data",
        )

    if vr >= vr_threshold:
        return CoinEligibility(
            symbol="", hurst=h_equiv, half_life_bars=hl,
            eligible=False,
            reason=f"VR={vr:.3f}>={vr_threshold:.2f} (trending/random walk)",
            reason_code="vr",
        )

    if hl is None:
        return CoinEligibility(
            symbol="", hurst=h_equiv, half_life_bars=None,
            eligible=False, reason="half_life: beta>=0 (trending) or insufficient data",
            reason_code="hl",
        )

    if hl > timeout_bars * hl_slack:
        return CoinEligibility(
            symbol="", hurst=h_equiv, half_life_bars=hl,
            eligible=False,
            reason=f"HL={hl:.1f}bars > timeout*slack={timeout_bars * hl_slack:.1f}",
            reason_code="hl",
        )

    return CoinEligibility(
        symbol="", hurst=h_equiv, half_life_bars=hl,
        eligible=True,
        reason=f"ok: VR={vr:.3f} (H~{h_equiv:.3f}) HL={hl:.1f}bars",
        reason_code="ok",
    )


# ────────────────────────────────────────────────────────────────────────────
# Composite Regime Score（rule-based voting，唔係 weighted sum）
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeSignals:
    vpin_pct: float          # 0-100，VPIN rolling percentile
    liq_spike: bool          # 清算 spike 偵測
    vol_burst_ratio: Optional[float]  # short/long vol ratio
    obi_accel: Optional[float]       # OBI 加速度


def compute_regime_zone(sig: RegimeSignals) -> tuple[RegimeZone, str]:
    """
    Rule-based voting（Opus 4.6 建議：比 weighted linear sum robust）。

    Red 觸發條件（任一）：
      - VPIN > 80th pct（高毒性流量）
      - liquidation spike（cascade 開始）

    Yellow 觸發條件（Red 以外，任兩個）：
      - VPIN > 65th pct
      - vol_burst_ratio > 2.0（短期 vol 突破長期 2 倍）
      - OBI 加速度絕對值 > 0.15（深度快速消失）

    Green：其餘情況

    唔需要 optimize weight：每個 signal 係 binary vote，直觀且 robust。
    """
    red_votes = 0
    yellow_votes = 0
    reasons = []

    # Red signals
    if sig.vpin_pct > 80:
        red_votes += 1
        reasons.append(f"VPIN>{sig.vpin_pct:.0f}pct")
    if sig.liq_spike:
        red_votes += 1
        reasons.append("LIQ_SPIKE")

    # Yellow signals
    if 65 < sig.vpin_pct <= 80:
        yellow_votes += 1
        reasons.append(f"VPIN_ELEVATED({sig.vpin_pct:.0f}pct)")
    if sig.vol_burst_ratio is not None and sig.vol_burst_ratio > 2.0:
        yellow_votes += 1
        reasons.append(f"VOL_BURST({sig.vol_burst_ratio:.1f}x)")
    if sig.obi_accel is not None and abs(sig.obi_accel) > 0.15:
        yellow_votes += 1
        reasons.append(f"OBI_ACCEL({sig.obi_accel:+.3f})")

    if red_votes >= 1:
        return RegimeZone.RED, " | ".join(reasons)
    if yellow_votes >= 2:
        return RegimeZone.YELLOW, " | ".join(reasons)
    return RegimeZone.GREEN, "normal"


# ────────────────────────────────────────────────────────────────────────────
# Dynamic SL（分段 time-decay，Opus 4.6 建議用 step function 取代 exponential）
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DynamicSLConfig:
    """
    Trailing SL 配置（取代舊嘅 phase shrinking）。

    舊版（phase shrinking）：持倉越久 SL 越緊，trending 時主動收近 entry，
    加快 stopout——已證明係 trending market 嘅毒藥。

    新版（trailing-only）：
      - SL 距離永遠保持 base_sl（唔隨時間收緊）
      - 但若已賺超過 trail_trigger_atr × ATR：
          → SL trail 到 entry ± trail_lock_atr × ATR（鎖部分利潤）
      - 適用於 trending 市場（reward/risk 不會隨時間惡化）
        亦適用於正常 mean reversion（賺到位再縮 SL）

    Regime override 保留（YELLOW/RED 仍縮 SL，因為毒性高應快速 flush）。

    atr_mult：ATR-normalized SL 距離（per coin / per vol regime）
    """
    atr_mult: float = 1.0           # SL 距離 = atr_mult × ATR
    trail_trigger_atr: float = 1.0  # 賺超過 1 × ATR 才開始 trail
    trail_lock_atr:    float = 0.3  # trail 到 entry ± 0.3 × ATR


def compute_sl_distance(
    base_sl: float,
    entry_time: float,
    half_life_bars: float,
    bar_duration_sec: float,
    config: DynamicSLConfig,
    regime_zone: RegimeZone,
) -> float:
    """
    計算當前 SL 距離（以 price 為單位）。

    Trailing-only 設計：SL 距離永遠 = base_sl（唔隨時間收緊），
    但 _manage_position 會用 trail_trigger_atr / trail_lock_atr 決定
    SL price 是否要 move 向更有利方向（只 move，唔 shrink）。

    Regime override：YELLOW 縮 50%，RED 縮 30%（毒性高應快速 flush）。

    base_sl：入場時嘅初始 SL 距離（atr_mult × ATR）
    half_life_bars / bar_duration_sec：保留作向後兼容（已不再使用）
    """
    sl_dist = base_sl

    if regime_zone == RegimeZone.YELLOW:
        sl_dist *= 0.5
    elif regime_zone == RegimeZone.RED:
        sl_dist *= 0.3

    return max(sl_dist, 1e-8)


# ────────────────────────────────────────────────────────────────────────────
# Position Sizing（Quarter-Kelly，Opus 4.6 建議）
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SizingConfig:
    """
    Opus 4.6 建議：
      - 唔用 rolling Kelly（estimation error 大），用 quarter-Kelly + shrinkage
      - 用 rolling edge estimate，但 shrink μ 向 0（打八折）
      - Hurst 越小（更強 mean reversion），size 越大
      - max_fraction：單幣最大 equity 佔比
    """
    max_fraction: float = 0.05    # 最多用 5% equity 做單幣
    min_fraction: float = 0.005   # 最少 0.5%
    kelly_shrinkage: float = 0.8  # μ 打八折，應對 estimation error
    quarter_kelly: bool = True    # 用 1/4 Kelly


def compute_position_size(
    equity: float,
    entry_price: float,
    sl_distance: float,
    hurst: float,
    roll_mu: float,        # rolling 平均每筆 P&L（absolute，USDT）
    roll_sigma2: float,    # rolling P&L variance
    config: SizingConfig,
    regime_zone: RegimeZone,
) -> float:
    """
    返回建議的 notional size（USDT）。

    步驟：
      1. 計 quarter-Kelly（基於 shrunk edge estimate）
      2. 按 Hurst 調整（Hurst 越小 → size 越大，最多 1.5x，最少 0.5x）
      3. 按 Regime 縮（Yellow = 0.5x，Red = 0x）
      4. Clip 到 [min_fraction, max_fraction] × equity

    SL-based position sizing：
      每筆最大虧損 = risk_per_trade × equity
      size_contracts = risk / sl_distance
    """
    if regime_zone == RegimeZone.RED:
        return 0.0

    # Quarter-Kelly fraction（基於 edge estimate）
    if roll_sigma2 > 0 and roll_mu > 0:
        mu_shrunk = roll_mu * config.kelly_shrinkage
        kelly_f = mu_shrunk / roll_sigma2
        if config.quarter_kelly:
            kelly_f *= 0.25
        fraction = max(config.min_fraction, min(kelly_f, config.max_fraction))
    else:
        # 唔夠 history：用最保守 fraction
        fraction = config.min_fraction

    # Hurst adjustment（per Opus 4.6 建議：H 小 → 更強 MR → 加注）
    hurst_adj = 1.0
    if hurst < 0.3:
        hurst_adj = 1.5
    elif hurst < 0.4:
        hurst_adj = 1.2
    elif hurst > 0.45:
        hurst_adj = 0.7
    fraction *= hurst_adj

    # Regime adjustment
    if regime_zone == RegimeZone.YELLOW:
        fraction *= 0.5

    fraction = max(config.min_fraction, min(fraction, config.max_fraction))
    notional = equity * fraction
    return max(notional, 0.0)
