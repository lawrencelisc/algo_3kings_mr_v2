"""
core/execution.py — 執行層：手續費模型、Lee-Ready、Adverse Flow Exit

手續費係所有盈虧計算嘅基礎，必須完整正確。
Hyperliquid 費率（截圖確認）：
  Maker: 0.0384%（limit order，ALO 掛單成交）
  Taker: 0.0400%（market order，或 limit 即時成交）

設計原則：
  - 每筆交易入場 + 出場費用都要入帳
  - Limit 出場（TP）用 maker rate，市價出場（SL/urgent）用 taker rate
  - 手續費直接從 equity 扣，唔可以「忘記」
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Fee Model
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FeeModel:
    """
    Hyperliquid 實際費率（截圖確認）。
    Maker = 0.0384%，Taker = 0.0400%。

    來回費用（全 maker）= 0.0768%
    來回費用（maker + taker SL）= 0.0784%

    Breakeven move（bps）= 0.0784 × 100 / price × price ≈ 7.84 bps
    即每單至少要賺 7.84 bps 才能保本。
    """
    maker_rate: float = 0.000384   # 0.0384%
    taker_rate: float = 0.000400   # 0.0400%

    def entry_fee(self, notional: float, is_maker: bool = True) -> float:
        """入場手續費（正數代表扣除）。"""
        rate = self.maker_rate if is_maker else self.taker_rate
        return notional * rate

    def exit_fee(self, notional: float, is_maker: bool) -> float:
        """出場手續費（正數代表扣除）。"""
        rate = self.maker_rate if is_maker else self.taker_rate
        return notional * rate

    def round_trip_fee(self, notional: float, exit_is_maker: bool = False) -> float:
        """來回總手續費（entry always maker，exit 按 reason）。"""
        return self.entry_fee(notional, is_maker=True) + self.exit_fee(notional, is_maker=exit_is_maker)

    def breakeven_bps(self, exit_is_maker: bool = False) -> float:
        """保本最低需要多少 bps 移動。"""
        rt = self.maker_rate + (self.maker_rate if exit_is_maker else self.taker_rate)
        return rt * 1e4


# ────────────────────────────────────────────────────────────────────────────
# Trade Record（完整手續費記錄）
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    side: str              # "long" | "short"
    entry_price: float
    exit_price: float
    notional: float        # entry notional（USDT）
    size: float            # contract size
    gross_pnl: float
    fee_entry: float       # 開倉費
    fee_exit: float        # 平倉費
    fee_total: float       # 合計
    net_pnl: float         # gross - fee_total
    hold_bars: int
    hold_sec: float
    reason: str            # TP | SL | DECEL | TIMEOUT | REGIME_RED | ADVERSE_FLOW
    regime_at_exit: str
    hurst_at_entry: Optional[float]
    half_life_at_entry: Optional[float]
    z_at_entry: float
    vpin_pct_at_entry: float
    size_mult: float       # regime / hurst 調整後嘅倉位乘數
    entry_is_maker: bool
    exit_is_maker: bool
    closed_at: float       # unix timestamp

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 6),
            "exit_price": round(self.exit_price, 6),
            "notional": round(self.notional, 4),
            "size": round(self.size, 6),
            "gross_pnl": round(self.gross_pnl, 6),
            "fee_entry": round(self.fee_entry, 6),
            "fee_exit": round(self.fee_exit, 6),
            "fee_total": round(self.fee_total, 6),
            "net_pnl": round(self.net_pnl, 6),
            "hold_bars": self.hold_bars,
            "hold_sec": round(self.hold_sec, 1),
            "reason": self.reason,
            "regime_at_exit": self.regime_at_exit,
            "hurst_at_entry": round(self.hurst_at_entry, 4) if self.hurst_at_entry else None,
            "half_life_at_entry": round(self.half_life_at_entry, 2) if self.half_life_at_entry else None,
            "z_at_entry": round(self.z_at_entry, 4),
            "vpin_pct_at_entry": round(self.vpin_pct_at_entry, 1),
            "size_mult": round(self.size_mult, 4),
            "entry_is_maker": self.entry_is_maker,
            "exit_is_maker": self.exit_is_maker,
            "closed_at_ts": self.closed_at,
        }


# ────────────────────────────────────────────────────────────────────────────
# Microprice-based Limit Order Simulation
# ────────────────────────────────────────────────────────────────────────────

def simulate_limit_fill(
    side: str,
    limit_price: float,
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    max_slippage_bps: float = 5.0,
) -> Tuple[Optional[float], bool]:
    """
    模擬 limit order 成交。
    返回 (fill_price, is_maker)。
    如果唔 fill 返回 (None, False)。

    is_maker = True 代表成功排隊，False 代表 aggressive fill（taker）。
    """
    if side == "long":
        best_ask = asks[0][0] if asks else None
        if best_ask is None:
            return None, False
        # 如果 limit_price >= best_ask：taker fill
        if limit_price >= best_ask:
            slippage = (limit_price - best_ask) / best_ask * 1e4
            if slippage <= max_slippage_bps:
                return best_ask, False  # taker
        # Maker：price 排喺 bid side，等成交
        # 簡化：假設 limit = microprice 左右，大機率 maker
        return limit_price, True
    else:  # short
        best_bid = bids[0][0] if bids else None
        if best_bid is None:
            return None, False
        if limit_price <= best_bid:
            slippage = (best_bid - limit_price) / best_bid * 1e4
            if slippage <= max_slippage_bps:
                return best_bid, False  # taker
        return limit_price, True


# ────────────────────────────────────────────────────────────────────────────
# Lee-Ready Trade Classification + Adverse Flow Monitor
# ────────────────────────────────────────────────────────────────────────────

class LeeReadyClassifier:
    """
    Lee-Ready (1991) tick test + quote test。
    用嚟分類每筆 trade 係 buy-initiated 定 sell-initiated。

    你已有呢個模組，呢度係精簡版整合入 adverse flow monitor。
    """

    def __init__(self) -> None:
        self._prev_trade_price: Optional[float] = None
        self._mid: Optional[float] = None

    def update_mid(self, bid: float, ask: float) -> None:
        self._mid = (bid + ask) / 2.0

    def classify(self, trade_price: float, trade_size: float) -> str:
        """
        返回 "buy" | "sell" | "unknown"。
        Quote rule 優先，tick rule 後備。
        """
        # Quote rule
        if self._mid is not None:
            if trade_price > self._mid:
                self._prev_trade_price = trade_price
                return "buy"
            elif trade_price < self._mid:
                self._prev_trade_price = trade_price
                return "sell"

        # Tick rule（後備）
        if self._prev_trade_price is not None:
            if trade_price > self._prev_trade_price:
                self._prev_trade_price = trade_price
                return "buy"
            elif trade_price < self._prev_trade_price:
                self._prev_trade_price = trade_price
                return "sell"

        self._prev_trade_price = trade_price
        return "unknown"


class AdverseFlowMonitor:
    """
    持倉中持續監控逆向 flow，提早 exit。

    Opus 4.6 建議：
      如果持倉方向相反嘅 net flow 持續 N 根 bar > threshold，
      唔等 SL，提早 exit。

    避免 overfit：N 用 3-5 就夠，唔需要 optimize。
    threshold 用 flow imbalance 嘅 rolling 標準差嘅倍數。
    """

    def __init__(
        self,
        adverse_bars_threshold: int = 3,
        imbalance_sigma_mult: float = 1.5,
        window: int = 50,
    ) -> None:
        self.adverse_bars_threshold = adverse_bars_threshold
        self.imbalance_sigma_mult = imbalance_sigma_mult
        self._imbalances: Deque[float] = deque(maxlen=window)
        self._adverse_count = 0

    def update(self, buy_flow: float, sell_flow: float, position_side: str) -> bool:
        """
        返回 True = 觸發 adverse flow exit。
        position_side: "long" | "short"
        """
        total = buy_flow + sell_flow
        if total <= 0:
            return False

        imbalance = (buy_flow - sell_flow) / total  # +1 = all buy，-1 = all sell
        self._imbalances.append(imbalance)

        if len(self._imbalances) < 10:
            return False

        arr = np.array(list(self._imbalances))
        sigma = float(np.std(arr))
        threshold = self.imbalance_sigma_mult * sigma

        # 逆向：long 持倉但 sell flow 主導（imbalance 負），short 持倉但 buy flow 主導
        if position_side == "long":
            is_adverse = imbalance < -threshold
        else:
            is_adverse = imbalance > threshold

        if is_adverse:
            self._adverse_count += 1
        else:
            self._adverse_count = 0

        return self._adverse_count >= self.adverse_bars_threshold

    def reset(self) -> None:
        self._adverse_count = 0


# ────────────────────────────────────────────────────────────────────────────
# Account（equity tracking + P&L）
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Account:
    """
    完整帳戶管理。
    所有費用都即時從 equity 扣除，唔會有「忘記扣費」嘅情況。
    """
    initial_equity: float
    fee_model: FeeModel = field(default_factory=FeeModel)

    equity: float = field(init=False)
    total_fee_paid: float = field(init=False, default=0.0)
    trades: List[TradeRecord] = field(default_factory=list, init=False)
    _trade_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.equity = self.initial_equity

    def deduct_entry_fee(self, notional: float, is_maker: bool = True) -> float:
        """開倉即時扣費，返回費用金額。"""
        fee = self.fee_model.entry_fee(notional, is_maker)
        self.equity -= fee
        self.total_fee_paid += fee
        return fee

    def close_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        notional: float,
        size: float,
        fee_entry: float,      # 已喺 deduct_entry_fee 扣過
        reason: str,
        hold_bars: int,
        hold_sec: float,
        regime_at_exit: str,
        hurst_at_entry: Optional[float],
        half_life_at_entry: Optional[float],
        z_at_entry: float,
        vpin_pct_at_entry: float,
        size_mult: float,
        entry_is_maker: bool = True,
    ) -> TradeRecord:
        """
        平倉：計算 gross PnL、扣出場費、更新 equity。
        SL / REGIME_RED / ADVERSE_FLOW → taker exit
        TP / TIMEOUT / DECEL → maker exit
        """
        taker_reasons = {"SL", "REGIME_RED", "ADVERSE_FLOW"}
        exit_is_maker = reason not in taker_reasons

        exit_notional = exit_price * size
        fee_exit = self.fee_model.exit_fee(exit_notional, is_maker=exit_is_maker)

        if side == "long":
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size

        fee_total = fee_entry + fee_exit
        net_pnl = gross_pnl - fee_exit  # entry_fee 已扣，呢度只扣 exit_fee
        # 但 TradeRecord.net_pnl 顯示完整來回費用
        net_pnl_full = gross_pnl - fee_total

        # 更新 equity（只加 gross_pnl - fee_exit，因為 fee_entry 已扣）
        self.equity += gross_pnl - fee_exit
        self.total_fee_paid += fee_exit

        self._trade_counter += 1
        rec = TradeRecord(
            trade_id=self._trade_counter,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            notional=notional,
            size=size,
            gross_pnl=gross_pnl,
            fee_entry=fee_entry,
            fee_exit=fee_exit,
            fee_total=fee_total,
            net_pnl=net_pnl_full,  # 完整費用後嘅 net
            hold_bars=hold_bars,
            hold_sec=hold_sec,
            reason=reason,
            regime_at_exit=regime_at_exit,
            hurst_at_entry=hurst_at_entry,
            half_life_at_entry=half_life_at_entry,
            z_at_entry=z_at_entry,
            vpin_pct_at_entry=vpin_pct_at_entry,
            size_mult=size_mult,
            entry_is_maker=entry_is_maker,
            exit_is_maker=exit_is_maker,
            closed_at=time.time(),
        )
        self.trades.append(rec)
        return rec

    def summary(self) -> dict:
        if not self.trades:
            return {
                "equity": round(self.equity, 4),
                "pnl": round(self.equity - self.initial_equity, 4),
                "trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "total_fee_paid": round(self.total_fee_paid, 4),
                "fee_pct_of_gross": 0.0,
            }
        nets = [t.net_pnl for t in self.trades]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]
        gross_win = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        gross_loss = abs(sum(t.gross_pnl for t in self.trades if t.gross_pnl <= 0))
        return {
            "equity": round(self.equity, 4),
            "pnl": round(self.equity - self.initial_equity, 4),
            "trades": len(self.trades),
            "win_rate": round(len(wins) / len(nets) * 100, 1) if nets else 0,
            "avg_win": round(sum(wins) / len(wins), 4) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else 0,
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
            "total_fee_paid": round(self.total_fee_paid, 4),
            "fee_pct_of_gross": round(
                self.total_fee_paid / max(gross_win + gross_loss, 1e-9) * 100, 1
            ),
        }
