"""
core/cooldown.py — 幣種 Cooldown 機制

規則：
  - 滾動 24 小時內同一幣種輸兩次 SL → 坐監（cooldown）
  - Cooldown 期間該幣種停止開新倉，已有倉位繼續正常管理
  - Cooldown 時長：預設 4 小時（可 .env 設定 MR_COOLDOWN_HOURS）
  - 狀態持久化：每次更新都寫入 JSON 檔案
  - 程式重啟時自動讀取 JSON，恢復未到期的 cooldown

持久化格式（cooldown_state.json）：
  {
    "BTC/USDC:USDC": {
      "sl_timestamps": [1777526860.0, 1777527415.0],   // 24h 內的 SL 時間戳
      "cooldown_until": 1777541415.0                    // None 代表未在 cooldown
    },
    ...
  }

重啟邏輯：
  - 讀取 JSON，剔除 24h 外的 SL 記錄
  - cooldown_until > now → 幣種仍在 cooldown，顯示剩餘時間
  - cooldown_until <= now → 自動解除
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("cooldown")

_24H = 86400.0   # 秒


@dataclass
class CooldownState:
    """單一幣種的 cooldown 狀態。"""
    sl_timestamps: List[float] = field(default_factory=list)   # 滾動 24h 內的 SL 時間戳
    cooldown_until: Optional[float] = None                     # None = 未在 cooldown


class CooldownManager:
    """
    幣種 Cooldown 管理器。

    使用方式（在 main loop 裡）：

        # 初始化（重啟時自動恢復狀態）
        cooldown = CooldownManager(
            cooldown_hours=4.0,
            max_sl_in_24h=2,
            state_path="cooldown_state.json",
        )

        # 每次 SL 出場後呼叫
        cooldown.record_sl(symbol)

        # 入場前 check
        if cooldown.is_cooling(symbol):
            logger.info("COOLDOWN %s — skip entry", symbol)
            continue

        # 重啟時顯示狀態摘要
        cooldown.log_status()
    """

    def __init__(
        self,
        cooldown_hours: float = 4.0,
        max_sl_in_24h: int = 2,
        state_path: str = "cooldown_state.json",
    ) -> None:
        self._cooldown_sec = cooldown_hours * 3600
        self._max_sl = max_sl_in_24h
        self._state_path = state_path
        self._states: Dict[str, CooldownState] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """從 JSON 恢復狀態，自動清理過期記錄。"""
        if not os.path.isfile(self._state_path):
            logger.info("Cooldown: 未發現狀態檔案 %s，全新啟動", self._state_path)
            return
        try:
            with open(self._state_path) as f:
                raw = json.load(f)
            now = time.time()
            loaded = 0
            for sym, d in raw.items():
                # 清除 24h 外的 SL 記錄
                sl_ts = [t for t in d.get("sl_timestamps", []) if now - t < _24H]
                cd_until = d.get("cooldown_until")
                # 如果 cooldown 已過期，解除但保留 24h 內的 SL 記錄（累計模式）
                if cd_until is not None and cd_until <= now:
                    logger.info("✅ COOLDOWN EXPIRED（重啟）%s → 解除，保留 24h SL 記錄", sym)
                    cd_until = None
                    # sl_ts 已在上方清除 24h 外記錄，保留 24h 內的繼續累計
                self._states[sym] = CooldownState(
                    sl_timestamps=sl_ts,
                    cooldown_until=cd_until,
                )
                loaded += 1
            logger.info("Cooldown: 從 %s 恢復 %d 個幣種狀態", self._state_path, loaded)
        except Exception as e:
            logger.warning("Cooldown: 讀取狀態檔案失敗 (%s)，重新開始", e)

    def _save(self) -> None:
        """持久化當前狀態到 JSON。"""
        try:
            data = {
                sym: {
                    "sl_timestamps": s.sl_timestamps,
                    "cooldown_until": s.cooldown_until,
                }
                for sym, s in self._states.items()
            }
            # 原子寫入（先寫 tmp 再 rename，避免寫到一半掉電）
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning("Cooldown: 寫入狀態失敗 (%s)", e)

    # ── Core Logic ───────────────────────────────────────────────────────────

    def _state(self, sym: str) -> CooldownState:
        if sym not in self._states:
            self._states[sym] = CooldownState()
        return self._states[sym]

    def record_sl(self, sym: str) -> bool:
        """
        記錄一次 SL 出場。
        返回 True = 剛觸發 cooldown（本次 SL 是第 N 次）。
        """
        now = time.time()
        st = self._state(sym)

        # 清除 24h 外的舊 SL 記錄
        st.sl_timestamps = [t for t in st.sl_timestamps if now - t < _24H]
        st.sl_timestamps.append(now)

        triggered = False
        if len(st.sl_timestamps) >= self._max_sl and st.cooldown_until is None:
            st.cooldown_until = now + self._cooldown_sec
            triggered = True
            remaining_h = self._cooldown_sec / 3600
            logger.warning(
                "🚫 COOLDOWN TRIGGERED  %s  "
                "（24h 內第 %d 次 SL）→ 坐監 %.1f 小時 until %s",
                sym,
                len(st.sl_timestamps),
                remaining_h,
                _fmt_time(st.cooldown_until),
            )
        else:
            sl_count = len(st.sl_timestamps)
            logger.info(
                "SL recorded  %s  24h_count=%d/%d  cooldown=%s",
                sym, sl_count, self._max_sl,
                "active" if st.cooldown_until else "none",
            )

        self._save()
        return triggered

    def is_cooling(self, sym: str) -> bool:
        """
        返回 True = 幣種仍在 cooldown，應拒絕開新倉。

        Cooldown 解除後行為（累計模式）：
          sl_timestamps 保留 24h 視窗內的記錄。
          復出後只要再多一次 SL，仍在 24h 視窗內的舊記錄加上新記錄
          若總數 >= max_sl，即刻再次觸發坐監。
          即：24h 內每多輸一次就再坐一次，直到舊記錄自然滾出 24h 視窗。
        """
        st = self._state(sym)
        if st.cooldown_until is None:
            return False
        now = time.time()
        if now >= st.cooldown_until:
            logger.info("✅ COOLDOWN LIFTED  %s  (已到期，24h 內 SL 記錄保留累計)", sym)
            st.cooldown_until = None
            # 保留 24h 內的 SL 記錄（累計模式）
            st.sl_timestamps = [t for t in st.sl_timestamps if now - t < _24H]
            self._save()
            return False
        return True

    def remaining_sec(self, sym: str) -> float:
        """返回剩餘 cooldown 秒數（0 = 未在 cooldown）。"""
        st = self._state(sym)
        if st.cooldown_until is None:
            return 0.0
        return max(0.0, st.cooldown_until - time.time())

    def sl_count_24h(self, sym: str) -> int:
        """返回該幣種滾動 24h 內的 SL 次數。"""
        now = time.time()
        st = self._state(sym)
        return sum(1 for t in st.sl_timestamps if now - t < _24H)

    # ── Status / Reporting ───────────────────────────────────────────────────

    def log_status(self) -> None:
        """
        啟動時輸出完整 cooldown 狀態摘要。
        讓交易者清楚知道哪些幣種仍在坐監。
        """
        now = time.time()
        cooling = []
        warned = []

        for sym, st in self._states.items():
            sl_count = sum(1 for t in st.sl_timestamps if now - t < _24H)
            if st.cooldown_until and now < st.cooldown_until:
                rem_h = (st.cooldown_until - now) / 3600
                cooling.append((sym, sl_count, rem_h, st.cooldown_until))
            elif sl_count >= self._max_sl - 1:
                # 快到 cooldown 邊緣（已有 N-1 次 SL）
                warned.append((sym, sl_count))

        logger.info("=" * 60)
        logger.info("COOLDOWN STATUS（程式重啟）")
        logger.info("  Cooldown 設定：24h 內 %d 次 SL → 坐監 %.1fh", self._max_sl, self._cooldown_sec / 3600)

        if cooling:
            logger.warning("  🚫 坐監中 (%d 隻)：", len(cooling))
            for sym, sl_n, rem_h, until in cooling:
                logger.warning(
                    "    %-25s  24h_SL=%d  剩餘 %.1fh  until=%s",
                    sym, sl_n, rem_h, _fmt_time(until),
                )
        else:
            logger.info("  ✅ 無幣種在坐監")

        if warned:
            logger.warning("  ⚠️  接近邊緣（再一次 SL 即觸發）：")
            for sym, sl_n in warned:
                logger.warning("    %-25s  24h_SL=%d/%d", sym, sl_n, self._max_sl)

        if not cooling and not warned:
            logger.info("  所有幣種正常")
        logger.info("=" * 60)

    def status_dict(self) -> Dict:
        """返回所有幣種的 cooldown 狀態 dict（供 SUMMARY log 用）。"""
        now = time.time()
        result = {}
        for sym, st in self._states.items():
            sl_n = sum(1 for t in st.sl_timestamps if now - t < _24H)
            cooling = st.cooldown_until is not None and now < st.cooldown_until
            result[sym] = {
                "sl_24h": sl_n,
                "cooling": cooling,
                "remaining_h": round((st.cooldown_until - now) / 3600, 2) if cooling else 0.0,
            }
        return result

    def all_cooling(self) -> List[str]:
        """返回所有仍在 cooldown 的幣種列表。"""
        now = time.time()
        return [
            sym for sym, st in self._states.items()
            if st.cooldown_until is not None and now < st.cooldown_until
        ]


# ── Helper ───────────────────────────────────────────────────────────────────

def _fmt_time(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
