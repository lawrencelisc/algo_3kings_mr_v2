"""
core/risk.py — Portfolio-level risk controls

兩個獨立 guard，每 bar 都要 update：

  CorrelationGuard
    追蹤各幣種 rolling log-return，入場前 check：
      1. 新幣 vs 任何已持倉幣的 pairwise corr 超過 threshold → block
      2. 全組合平均 corr 超過 avg_threshold → block（隱性集中風險）

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
    防止隱性集中風險（同方向高度相關幣種同時持倉）。

    每 bar 呼叫 update(symbol, close)。
    入場前呼叫 check_entry(symbol, held_symbols)。
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
        held_symbols: List[str],
    ) -> Tuple[bool, str]:
        """
        返回 (blocked, reason)。
        blocked=True 代表唔應該開倉。
        """
        if not held_symbols:
            return False, ""

        corrs = []
        for held in held_symbols:
            if held == symbol:
                continue
            c = self._corr(symbol, held)
            if c is None:
                continue
            if abs(c) >= self._threshold:
                return True, (
                    f"pairwise corr({symbol},{held})={c:.2f} >= {self._threshold}"
                )
            corrs.append(abs(c))

        if corrs:
            avg_c = sum(corrs) / len(corrs)
            if avg_c >= self._avg_threshold:
                return True, (
                    f"avg portfolio corr={avg_c:.2f} >= {self._avg_threshold}"
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
