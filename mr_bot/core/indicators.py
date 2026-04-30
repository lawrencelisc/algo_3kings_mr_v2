"""
core/indicators.py — 均值回歸 Bot 指標庫
Opus 4.6 方案實現：model-based 優先，避免 overfit

包含：
  - Microprice（microstructure theory，無 free parameter）
  - Kalman Filter Z-Score（state-space，MLE 估 Q/R）
  - Hurst Exponent（rolling，R/S 法）
  - Half-life of mean reversion（OU process AR(1) 估）
  - VPIN（Easley et al. 2012，bucket-based）
  - Realized volatility burst detection
  - Order book depth imbalance acceleration

設計原則：
  每個指標盡量少 tunable parameter，且 parameter 有 economic rationale
  唔用 historical fitting 嚟決定 threshold，用 rolling percentile 做自適應
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Microprice（取代 mid-price，microstructure 標準做法）
# ────────────────────────────────────────────────────────────────────────────

def microprice(bid: float, bid_size: float, ask: float, ask_size: float) -> float:
    """
    Microprice = (bid × ask_size + ask × bid_size) / (bid_size + ask_size)

    偏向流動性較薄嗰邊，比 mid-price 少 spread bias。
    無 free parameter，pure theory。
    """
    total = bid_size + ask_size
    if total <= 0:
        return (bid + ask) / 2.0
    return (bid * ask_size + ask * bid_size) / total


def microprice_from_ob(levels: List[Tuple[float, float]], side_sign: int = 1) -> float:
    """
    從 order book 3-5 層計加權 microprice。
    levels: [(price, size), ...]，bid 從高到低，ask 從低到高。
    side_sign: bid = -1（從最優 bid 往下），ask = +1。
    取最優兩層加權即可，更多層嘅邊際效益低。
    """
    if not levels:
        return 0.0
    total_size = sum(s for _, s in levels[:3])
    if total_size <= 0:
        return levels[0][0]
    return sum(p * s for p, s in levels[:3]) / total_size


# ────────────────────────────────────────────────────────────────────────────
# Kalman Filter（local level model，MLE 估 Q/R）
# ────────────────────────────────────────────────────────────────────────────

class KalmanZScore:
    """
    將 price series 建模為 local level model：
        x_t = x_{t-1} + w_t,  w ~ N(0, Q)   ← latent fair value
        y_t = x_t + v_t,      v ~ N(0, R)   ← 觀測 microprice

    Z-score = (y_t - x̂_{t|t-1}) / sqrt(P_{t|t-1} + R)

    Q/R ratio 用 rolling MLE 估（warm-up 完成後自動更新），
    唔係手動設定，避免 overfit。

    Attributes
    ----------
    warm_up : int
        啟動 MLE 估計前需要嘅最少觀測數
    mle_window : int
        rolling MLE 嘅 window size（建議 200-500）
    """

    def __init__(self, warm_up: int = 100, mle_window: int = 300) -> None:
        self.warm_up = warm_up
        self.mle_window = mle_window

        # Kalman state
        self._x: Optional[float] = None   # x̂_{t|t}
        self._P: float = 1.0              # P_{t|t}
        self._Q: float = 1e-4             # initial guess
        self._R: float = 1e-2             # initial guess

        # buffer for MLE
        self._obs: Deque[float] = deque(maxlen=mle_window)
        self._innovations: Deque[float] = deque(maxlen=mle_window)
        self._innov_vars: Deque[float] = deque(maxlen=mle_window)
        self._n = 0

    def _mle_update_qr(self) -> None:
        """
        簡化 MLE：用 innovation sequence 估 Q + R。
        innovation variance = P_{t|t-1} + R ≈ Q + R（當 P 穩定後）
        用連續兩步 innovation 嘅 autocovariance 分離 Q 同 R。
        呢個係 Harvey (1990) 嘅近似做法。
        """
        if len(self._innovations) < self.warm_up:
            return
        innov = np.array(self._innovations)
        # Innovation variance estimate
        S_hat = float(np.var(innov))
        # Autocovariance lag-1
        if len(innov) > 1:
            ac1 = float(np.mean(innov[1:] * innov[:-1]))
        else:
            ac1 = 0.0
        # Q ≈ -ac1（如果 ac1 < 0），R ≈ S_hat + ac1
        Q_new = max(-ac1, 1e-8)
        R_new = max(S_hat + ac1, 1e-8)
        # Smooth update：唔 jump，slow EMA blend
        alpha = 0.05
        self._Q = (1 - alpha) * self._Q + alpha * Q_new
        self._R = (1 - alpha) * self._R + alpha * R_new

    def update(self, y: float) -> Optional[float]:
        """
        接受新 microprice 觀測值，返回 z-score。
        前 warm_up 個 bar 返回 None（尚未 warm up）。
        """
        self._obs.append(y)
        self._n += 1

        # 初始化
        if self._x is None:
            self._x = y
            self._P = self._R
            return None

        # Predict
        x_pred = self._x
        P_pred = self._P + self._Q

        # Innovation
        innov = y - x_pred
        S = P_pred + self._R          # innovation variance
        self._innovations.append(innov)
        self._innov_vars.append(S)

        # Update
        K = P_pred / S                # Kalman gain
        self._x = x_pred + K * innov
        self._P = (1 - K) * P_pred

        # Rolling MLE（每 50 步更新一次，唔係每步，省 CPU）
        if self._n % 50 == 0:
            self._mle_update_qr()

        if self._n < self.warm_up:
            return None

        z = innov / math.sqrt(max(S, 1e-12))
        return z

    @property
    def fair_value(self) -> Optional[float]:
        return self._x

    @property
    def qr_ratio(self) -> float:
        return self._Q / max(self._R, 1e-12)


# ────────────────────────────────────────────────────────────────────────────
# Hurst Exponent（R/S 法，rolling）
# ────────────────────────────────────────────────────────────────────────────

def variance_ratio(prices: Sequence[float], q: int = 5) -> Optional[float]:
    """
    Lo-MacKinlay (1988) Variance Ratio Test。

    VR(q) = Var(q-period log-return) / (q × Var(1-period log-return))

    VR < 1.0 → mean reverting（越細越強）
    VR ≈ 1.0 → random walk
    VR > 1.0 → trending / momentum

    取代 R/S Hurst 嘅原因：
      R/S Hurst 假設 price series 係 stationary，
      但 crypto price 係 non-stationary（有 unit root），
      結果永遠計出 H≈1.0，完全失效。
      VR 用 log-return（天然 stationary），對 crypto robust，
      無需 calibrate min_subseries。

    篩選門檻：VR(5) < 0.90 → mean reverting，入池
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < q * 4 + 2:
        return None
    rets = np.diff(np.log(np.maximum(arr, 1e-9)))
    n = len(rets)
    mu = float(np.mean(rets))
    var1 = float(np.sum((rets - mu) ** 2) / (n - 1))
    if var1 <= 0:
        return None
    rets_q = np.array([np.sum(rets[i:i + q]) for i in range(0, n - q + 1)])
    mu_q = float(np.mean(rets_q))
    m = len(rets_q)
    if m < 2:
        return None
    var_q = float(np.sum((rets_q - mu_q) ** 2) / (m - 1)) / q
    return var_q / var1


def hurst_rs(prices: Sequence[float], min_subseries: int = 10) -> Optional[float]:
    """
    已棄用：R/S Hurst 對 crypto price series 不可靠（永遠返回 H≈1.0）。

    此函數保留作向後兼容，內部改為呼叫 variance_ratio()：
      VR < 1 → mean reverting，映射：H_equiv = VR × 0.5
      例：VR=0.74 → H_equiv=0.37（< 0.45 門檻，通過篩選）
          VR=1.10 → H_equiv=0.55（> 0.45 門檻，踢出）
    """
    vr = variance_ratio(prices)
    if vr is None:
        return None
    return max(0.0, min(1.0, vr * 0.5))


class RollingVR:
    """每次 push 新 price，按需計算 rolling VR(q)。"""

    def __init__(self, window: int = 300, q: int = 5, compute_every: int = 20) -> None:
        self.window = window
        self.q = q
        self.compute_every = compute_every
        self._buf: Deque[float] = deque(maxlen=window)
        self._last_vr: Optional[float] = None
        self._n = 0

    def update(self, price: float) -> Optional[float]:
        self._buf.append(price)
        self._n += 1
        if self._n % self.compute_every == 0 and len(self._buf) >= self.q * 4 + 2:
            self._last_vr = variance_ratio(list(self._buf), self.q)
        return self._last_vr

    @property
    def is_mean_reverting(self) -> bool:
        return self._last_vr is not None and self._last_vr < 0.90


class RollingHurst:
    """向後兼容 wrapper，內部用 RollingVR（VR×0.5 映射回 Hurst-like 值）。"""

    def __init__(self, window: int = 300, compute_every: int = 20) -> None:
        self._vr = RollingVR(window=window, q=5, compute_every=compute_every)

    def update(self, price: float) -> Optional[float]:
        vr = self._vr.update(price)
        if vr is None:
            return None
        return max(0.0, min(1.0, vr * 0.5))


# ────────────────────────────────────────────────────────────────────────────
# Half-life of Mean Reversion（OU process，AR(1) 估）
# ────────────────────────────────────────────────────────────────────────────

def half_life_ou(prices: Sequence[float]) -> Optional[float]:
    """
    對 price series 跑 OLS AR(1) regression：
        Δp_t = β × p_{t-1} + α + ε

    half-life = -ln(2) / ln(1 + β)   （等價 = -ln(2) / β 當 β 接近 0）

    返回 minutes（假設 prices 係每分鐘 bar）。
    β 應為負數（mean reverting），正數代表 trending，返回 None。
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 20:
        return None
    delta = np.diff(arr)
    lag = arr[:-1]
    # OLS: delta = beta * lag + alpha
    lag_demeaned = lag - np.mean(lag)
    beta = float(np.sum(lag_demeaned * delta) / (np.sum(lag_demeaned ** 2) + 1e-12))
    if beta >= 0:
        return None  # trending，唔係 mean reverting
    # half-life in bars
    hl = -math.log(2) / math.log(1 + beta) if abs(1 + beta) < 1 else -math.log(2) / beta
    return max(0.5, min(hl, 9999.0))


# ────────────────────────────────────────────────────────────────────────────
# VPIN（Volume-Synchronized Probability of Informed Trading）
# Easley, Lopez de Prado, O'Hara (2012)
# ────────────────────────────────────────────────────────────────────────────

class VPIN:
    """
    Bucket-based VPIN。
    每個 bucket 累積固定 volume（bucket_size），
    然後用 price return 做 bulk volume classification（BVC）估 buy/sell split。

    VPIN = rolling N 個 bucket 嘅 |buy_vol - sell_vol| / total_vol

    唔做 parameter fitting，只用 rolling percentile 做 threshold。
    """

    def __init__(self, bucket_size: float = 1000.0, n_buckets: int = 50) -> None:
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets

        self._current_vol = 0.0
        self._current_buy_vol = 0.0
        self._prev_close: Optional[float] = None

        # rolling imbalances per bucket
        self._imbalances: Deque[float] = deque(maxlen=n_buckets)
        self._history: Deque[float] = deque(maxlen=500)  # for percentile

    def update(self, close: float, volume: float) -> Optional[float]:
        """
        每根 bar 更新。
        返回當前 VPIN（0~1），若 bucket 未填滿則返回上次值。
        """
        if self._prev_close is None:
            self._prev_close = close
            return None

        # Bulk Volume Classification：用 return sign 估 buy fraction
        ret = close - self._prev_close
        sigma_ret = 1e-6  # 防止 div/0，後面會動態更新
        if self._history:
            arr = np.array(list(self._history)[-50:])
            sigma_ret = max(float(np.std(np.diff(arr))), 1e-6)

        # buy fraction = Φ(return / sigma)（常態分佈 CDF）
        z = ret / sigma_ret
        buy_fraction = float(0.5 * (1 + math.erf(z / math.sqrt(2))))

        buy_vol = volume * buy_fraction
        sell_vol = volume * (1 - buy_fraction)

        self._current_vol += volume
        self._current_buy_vol += buy_vol
        self._prev_close = close
        self._history.append(close)

        # 填滿一個 bucket
        if self._current_vol >= self.bucket_size:
            imb = abs(self._current_buy_vol - (self._current_vol - self._current_buy_vol))
            self._imbalances.append(imb / self.bucket_size)
            self._current_vol = 0.0
            self._current_buy_vol = 0.0

        if len(self._imbalances) < 5:
            return None

        vpin_val = float(np.mean(self._imbalances))
        return vpin_val

    def percentile_rank(self, current_vpin: float) -> float:
        """返回當前 VPIN 喺歷史分佈嘅 percentile（0-100）。"""
        if len(self._imbalances) < 10:
            return 50.0
        arr = np.array(list(self._imbalances))
        return float(np.sum(arr <= current_vpin) / len(arr) * 100)


# ────────────────────────────────────────────────────────────────────────────
# Realized Volatility Burst Detection
# ────────────────────────────────────────────────────────────────────────────

class VolBurstDetector:
    """
    偵測 realized volatility 是否突破正常水平。
    唔需要 GARCH，用簡單 rolling window 比較足夠。
    """

    def __init__(self, short_window: int = 10, long_window: int = 100) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self._returns: Deque[float] = deque(maxlen=long_window)

    def update(self, price: float) -> Optional[float]:
        self._returns.append(price)

        if len(self._returns) < self.long_window:
            return None

        prices = list(self._returns)
        rets = [
            (prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-9)
            for i in range(1, len(prices))
        ]
        short_rets = rets[-self.short_window:]
        if len(short_rets) < 2 or len(rets) < 2:
            return None
        short_vol = float(np.std(short_rets, ddof=1))
        long_vol = float(np.std(rets, ddof=1))
        if long_vol <= 0:
            return None
        return short_vol / long_vol


# ────────────────────────────────────────────────────────────────────────────
# Order Book Depth Imbalance Acceleration
# ────────────────────────────────────────────────────────────────────────────

class OBIAcceleration:
    """
    追蹤 order book imbalance（OBI）嘅一階差分（加速度）。
    OBI = (bid_depth - ask_depth) / (bid_depth + ask_depth)

    imbalance 加速消失 → 做市商撤單 → 價格可能大幅移動（預警信號）
    """

    def __init__(self, window: int = 10) -> None:
        self._obis: Deque[float] = deque(maxlen=window)

    def update(self, bid_depth: float, ask_depth: float) -> Optional[float]:
        total = bid_depth + ask_depth
        if total <= 0:
            return None
        obi = (bid_depth - ask_depth) / total
        self._obis.append(obi)
        if len(self._obis) < 3:
            return None
        # 一階差分均值（acceleration）
        obis = list(self._obis)
        d1 = [obis[i] - obis[i - 1] for i in range(1, len(obis))]
        return float(np.mean(d1[-3:]))
