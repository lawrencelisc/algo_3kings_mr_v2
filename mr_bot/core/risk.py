"""
core/risk.py — Portfolio-level risk controls

兩個獨立 guard，每 bar 都要 update：

  CorrelationGuard（side-aware）
    追蹤各幣種 rolling log-return，入場前 check「方向性集中風險」：
      effective directional corr：
        same side（兩邊都 long 或都 short）  → eff_c = +ρ
        opposite side（一 long 一 short）    → eff_c = −ρ
      解釋：對住已 long 嘅倉，新 long 同 corr=0.7 = 真集中（eff=+0.7 → block）；
            新 short 同 corr=0.7 = 對沖（eff=−0.7 → 唔 block，反而減 portfolio risk）。

      Rules：
        1. 任一已持倉幣，eff_c ≥ threshold → block（單筆方向性集中）
        2. 對所有「正向 contribution」（eff_c > 0）求平均，≥ avg_threshold → block

      呢個改法解決：46 隻高度相關 crypto perp 入面，舊版只用 |ρ| 會將
      hedge trade 都當成集中風險，令第二單之後幾乎冇可能入場。

  DrawdownThrottle
    監控最近 window_sec 秒的 equity drawdown。
    超過 dd_limit → 把 size_multiplier 降至 throttle_mult（default 0.5）。
    Hysteresis：要回升至 recovery_frac × dd_limit 以下才復原，
    避免 boundary 附近頻繁切換。
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# CorrelationGuard
# ────────────────────────────────────────────────────────────────────────────

class CorrelationGuard:
    """
    防止「方向性集中風險」（同方向高度相關幣種同時持倉）。

    每 bar 呼叫 update(symbol, close)。
    入場前呼叫 check_entry(symbol, new_side, held_positions)。

    Side-aware 邏輯：
      effective_corr = +ρ  if 同方向（long+long 或 short+short）
                     = −ρ  if 反方向（long+short）
      block 條件 = eff_c ≥ threshold（單筆）或 平均正向 eff_c ≥ avg_threshold。
      反方向高 ρ → eff_c 變負 → 永遠唔 block（因為實際係對沖）。
    """

    def __init__(
        self,
        window: int = 200,
        threshold: float = 0.6,
        avg_threshold: float = 0.5,
    ) -> None:
        self._window = window
        self._threshold = threshold
        self._avg_threshold = avg_threshold
        self._log_rets: Dict[str, Deque[float]] = {}
        self._prev_close: Dict[str, float] = {}

    def update(self, symbol: str, close: float) -> None:
        if symbol not in self._log_rets:
            self._log_rets[symbol] = deque(maxlen=self._window)
            self._prev_close[symbol] = close
            return
        prev = self._prev_close[symbol]
        if prev > 0 and close > 0:
            self._log_rets[symbol].append(math.log(close / prev))
        self._prev_close[symbol] = close

    def _corr(self, sym_a: str, sym_b: str) -> Optional[float]:
        if sym_a not in self._log_rets or sym_b not in self._log_rets:
            return None
        a = list(self._log_rets[sym_a])
        b = list(self._log_rets[sym_b])
        n = min(len(a), len(b))
        if n < 20:
            return None
        a, b = a[-n:], b[-n:]
        mu_a = sum(a) / n
        mu_b = sum(b) / n
        cov = sum((x - mu_a) * (y - mu_b) for x, y in zip(a, b)) / (n - 1)
        var_a = sum((x - mu_a) ** 2 for x in a) / (n - 1)
        var_b = sum((y - mu_b) ** 2 for y in b) / (n - 1)
        denom = math.sqrt(max(var_a, 1e-12) * max(var_b, 1e-12))
        return cov / denom

    def check_entry(
        self,
        symbol: str,
        new_side: str,
        held_positions: Dict[str, str],
    ) -> Tuple[bool, str]:
        """
        Side-aware entry check。

        Parameters
        ----------
        symbol : str
            想入場嘅幣。
        new_side : "long" | "short"
            想入場嘅方向。
        held_positions : Dict[symbol, side]
            已持倉嘅 {幣 → 方向}。空 dict = 無倉。

        Returns
        -------
        (blocked, reason)
            blocked=True 代表「方向性集中風險過高」，唔應該開倉。
            反方向 hedge（new_side 同 held_side 唔同）會令 eff_c = −ρ，
            屬於 risk-reducing trade，永遠唔 block。
        """
        if not held_positions:
            return False, ""

        dir_corrs: List[float] = []
        for held, held_side in held_positions.items():
            if held == symbol:
                continue
            c = self._corr(symbol, held)
            if c is None:
                continue
            same_dir = (new_side == held_side)
            eff_c = c if same_dir else -c

            if eff_c >= self._threshold:
                return True, (
                    f"directional corr({symbol}/{new_side},{held}/{held_side})"
                    f"={eff_c:+.2f} >= {self._threshold}"
                )
            if eff_c > 0:
                dir_corrs.append(eff_c)

        if dir_corrs:
            avg_c = sum(dir_corrs) / len(dir_corrs)
            if avg_c >= self._avg_threshold:
                return True, (
                    f"avg directional corr={avg_c:+.2f} >= {self._avg_threshold}"
                    f" (n_same_dir={len(dir_corrs)})"
                )

        return False, ""


# ────────────────────────────────────────────────────────────────────────────
# DrawdownThrottle
# ────────────────────────────────────────────────────────────────────────────

class DrawdownThrottle:
    """
    監控 rolling window 內的 equity 高水位回撤。

    超過 dd_limit → is_throttled=True，size_multiplier 降至 throttle_mult。
    回升到 recovery_frac × dd_limit 以下才解除（hysteresis）。
    """

    def __init__(
        self,
        window_sec: float = 3600.0,
        dd_limit: float = 0.005,
        throttle_mult: float = 0.5,
        recovery_frac: float = 0.5,
    ) -> None:
        self._window_sec = window_sec
        self._dd_limit = dd_limit
        self._throttle_mult = throttle_mult
        self._recovery_frac = recovery_frac

        # (timestamp, equity) pairs
        self._history: Deque[Tuple[float, float]] = deque()
        self._is_throttled: bool = False

    def update(self, equity: float) -> None:
        now = time.time()
        self._history.append((now, equity))

        # 清除過期紀錄
        cutoff = now - self._window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if not self._history:
            return

        # 高水位（rolling window 內最高 equity）
        peak = max(eq for _, eq in self._history)
        dd = (peak - equity) / max(peak, 1e-9)
        self._current_drawdown = dd

        if not self._is_throttled:
            if dd >= self._dd_limit:
                self._is_throttled = True
        else:
            # Hysteresis：要低於 recovery_frac × dd_limit 才解除
            if dd <= self._dd_limit * self._recovery_frac:
                self._is_throttled = False

    @property
    def is_throttled(self) -> bool:
        return self._is_throttled

    @property
    def size_multiplier(self) -> float:
        return self._throttle_mult if self._is_throttled else 1.0

    @property
    def current_drawdown(self) -> float:
        return getattr(self, "_current_drawdown", 0.0)

    def status_str(self) -> str:
        dd_pct = self.current_drawdown * 100
        return f"dd={dd_pct:.2f}% throttle={self._is_throttled} mult={self.size_multiplier:.1f}x"
