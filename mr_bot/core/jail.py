"""
core/jail.py — 自動長期坐監（Symbol Jail）

設計目標：
  完全唔經人手管理 trending 主導的幣。
  以前要 ops 手動填 MR_EXCLUDE_SYMBOLS，依家由 bot 自動偵測 + 自動釋放。

層級對比：
  ┌──────────────────────┬──────────────────────────┬──────────────────┐
  │ 機制                 │ 觸發                     │ 期限             │
  ├──────────────────────┼──────────────────────────┼──────────────────┤
  │ Per-symbol cooldown  │ 24h 內 2 次 SL          │ 4h（短期）       │
  │ RegimePauseGuard     │ 全 universe 24h 8 次 SL │ 4h（全局）       │
  │ SymbolJail（本）     │ 累犯 cooldown / 連敗     │ 7d～30d（長期）  │
  │ MR_EXCLUDE_SYMBOLS   │ 人手 .env              │ 永久（人手）     │
  └──────────────────────┴──────────────────────────┴──────────────────┘

入監條件（任一觸發）：
  1. 7 日內進入 4h cooldown ≥ jail_cooldown_count（default 3）次
     → 短期坐監冇用，越坐越輸 → 升級長期
  2. 連續 SL ≥ jail_sl_streak（default 5）次（中間冇 TP / DECEL profit）
  3. 最近 jail_pnl_lookback（default 10）筆交易 net PnL 總和 < jail_pnl_threshold
     （default -0.5 USDC，~$10 notional 對應 5% 累計虧損）

釋放條件（必須全部達到）：
  1. 已坐滿 jail_min_days（default 7）日
  2. AND VR < vr_release_threshold（default 0.9，與 RegimePauseGuard 一致）
  3. AND 0 < half_life ≤ hl_release_max_mult × timeout_bars（default 2）

每日 check 一次；唔達標延長 1 日。  
硬上限 jail_max_days（default 30）：強制釋放並 WARNING（建議人手評估）。

持久化（jail_state.json）：
{
  "BTC/USDC:USDC": {
    "jailed_at":         1777526860.0,
    "release_after":     1778131660.0,
    "trigger_reason":    "cooldown_repeated",
    "trigger_detail":    "3 cooldowns in 7d",
    "extends":           0,
    "consecutive_sl":    5,
    "cooldown_ts":       [...],     # 7 日內 cooldown 觸發時間戳
    "recent_pnl":        [...],     # 最近 N 筆 net_pnl
    "history": [
      {"jailed_at": ..., "released_at": ..., "reason": "...", "release_kind": "auto"|"forced"}
    ]
  }
}
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger("jail")

_7D = 7 * 86400.0
_1D = 86400.0


@dataclass
class JailState:
    """單一幣種嘅 jail 狀態。"""
    # 入監資訊
    jailed_at:      Optional[float] = None
    release_after:  Optional[float] = None     # 最早可釋放時間（now >= 此值 + check_release pass）
    trigger_reason: str = ""                   # cooldown_repeated / sl_streak / pnl_drawdown
    trigger_detail: str = ""
    extends:        int = 0                    # 已延長次數

    # 入監前嘅累積證據（永久 keep，方便偵測累犯）
    consecutive_sl:  int = 0                                        # SL streak（贏一次即 reset）
    cooldown_ts:     List[float] = field(default_factory=list)      # 7 日內 cooldown 觸發時間戳
    recent_pnl:      Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    # 出獄歷史（純 audit log）
    history:         List[Dict] = field(default_factory=list)

    @property
    def is_jailed(self) -> bool:
        return self.jailed_at is not None


class SymbolJail:
    """自動長期坐監系統。"""

    def __init__(
        self,
        # 入監條件
        jail_cooldown_count: int = 3,           # 7d 內 cooldown 次數閾值
        jail_sl_streak:      int = 5,           # 連續 SL 閾值
        jail_pnl_lookback:   int = 10,          # 計 PnL 用幾多筆
        jail_pnl_threshold:  float = -0.5,      # 累計 PnL < 此值即入監（USDC）
        # 釋放條件
        jail_min_days:       float = 7.0,
        jail_max_days:       float = 30.0,
        vr_release_threshold:    float = 0.9,
        hl_release_max_mult:     float = 2.0,
        timeout_bars:        int = 30,
        # 持久化
        state_path:          str = "jail_state.json",
    ) -> None:
        self.jail_cooldown_count = jail_cooldown_count
        self.jail_sl_streak      = jail_sl_streak
        self.jail_pnl_lookback   = jail_pnl_lookback
        self.jail_pnl_threshold  = jail_pnl_threshold
        self.jail_min_sec        = jail_min_days * 86400
        self.jail_max_sec        = jail_max_days * 86400
        self.vr_release_threshold = vr_release_threshold
        self.hl_release_max_bars  = hl_release_max_mult * timeout_bars
        self.state_path          = state_path

        self._states: Dict[str, JailState] = {}
        # 防 spam：每個 symbol 兩次 release-check 之間至少間隔此秒數
        self._last_release_check: Dict[str, float] = {}
        self._release_check_interval = 3600.0    # 1 小時 check 一次釋放

        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.isfile(self.state_path):
            logger.info("Jail: 未發現狀態檔案 %s，全新啟動", self.state_path)
            return
        try:
            with open(self.state_path) as f:
                raw = json.load(f)
            now = time.time()
            loaded = 0
            for sym, d in raw.items():
                # 7 日外的 cooldown ts 自動清除
                cd_ts = [t for t in d.get("cooldown_ts", []) if now - t < _7D]
                pnl_q: Deque[float] = deque(
                    d.get("recent_pnl", []), maxlen=self.jail_pnl_lookback
                )
                self._states[sym] = JailState(
                    jailed_at      = d.get("jailed_at"),
                    release_after  = d.get("release_after"),
                    trigger_reason = d.get("trigger_reason", ""),
                    trigger_detail = d.get("trigger_detail", ""),
                    extends        = d.get("extends", 0),
                    consecutive_sl = d.get("consecutive_sl", 0),
                    cooldown_ts    = cd_ts,
                    recent_pnl     = pnl_q,
                    history        = d.get("history", []),
                )
                loaded += 1
            logger.info("Jail: 從 %s 恢復 %d 個幣種狀態", self.state_path, loaded)
        except Exception as e:
            logger.warning("Jail: 讀取狀態檔案失敗 (%s)，重新開始", e)

    def _save(self) -> None:
        try:
            data = {}
            for sym, s in self._states.items():
                data[sym] = {
                    "jailed_at":      s.jailed_at,
                    "release_after":  s.release_after,
                    "trigger_reason": s.trigger_reason,
                    "trigger_detail": s.trigger_detail,
                    "extends":        s.extends,
                    "consecutive_sl": s.consecutive_sl,
                    "cooldown_ts":    s.cooldown_ts,
                    "recent_pnl":     list(s.recent_pnl),
                    "history":        s.history[-20:],   # 只保留最近 20 條
                }
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.state_path)
        except Exception as e:
            logger.warning("Jail: 寫入狀態失敗 (%s)", e)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _state(self, sym: str) -> JailState:
        if sym not in self._states:
            self._states[sym] = JailState()
        return self._states[sym]

    def _trim_cooldown_ts(self, st: JailState, now: float) -> None:
        st.cooldown_ts = [t for t in st.cooldown_ts if now - t < _7D]

    def _put_in_jail(
        self,
        sym: str,
        st: JailState,
        reason: str,
        detail: str,
    ) -> None:
        now = time.time()
        st.jailed_at      = now
        st.release_after  = now + self.jail_min_sec
        st.trigger_reason = reason
        st.trigger_detail = detail
        st.extends        = 0
        logger.warning(
            "🔒 JAIL  %s  → 坐監 %.0f 日（最早 %s UTC 可申請釋放）  reason=%s [%s]",
            sym,
            self.jail_min_sec / 86400,
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(st.release_after)),
            reason, detail,
        )
        self._save()

    # ── Observation API（main loop 每次出場後呼叫）───────────────────────────

    def observe_sl(self, sym: str, net_pnl: float) -> bool:
        """
        記錄一次 SL 出場。
        返回 True = 此次 SL 觸發入監。
        """
        st = self._state(sym)
        st.consecutive_sl += 1
        st.recent_pnl.append(net_pnl)

        if st.is_jailed:
            self._save()
            return False

        # ── 條件 2：SL streak ──
        if st.consecutive_sl >= self.jail_sl_streak:
            self._put_in_jail(
                sym, st,
                reason="sl_streak",
                detail=f"{st.consecutive_sl} consecutive SL",
            )
            return True

        # ── 條件 3：rolling PnL ──
        if len(st.recent_pnl) >= self.jail_pnl_lookback:
            cum = sum(st.recent_pnl)
            if cum < self.jail_pnl_threshold:
                self._put_in_jail(
                    sym, st,
                    reason="pnl_drawdown",
                    detail=f"last {self.jail_pnl_lookback} trades cum={cum:+.4f}",
                )
                return True

        self._save()
        return False

    def observe_win(self, sym: str, net_pnl: float) -> None:
        """記錄一次 TP / DECEL / 任何盈利出場 → reset SL streak。"""
        st = self._state(sym)
        st.consecutive_sl = 0
        st.recent_pnl.append(net_pnl)
        self._save()

    def observe_neutral_exit(self, sym: str, net_pnl: float) -> None:
        """非 SL 出場（TIMEOUT / REGIME_RED 等）：記 PnL，唔 reset streak（保守）。"""
        st = self._state(sym)
        st.recent_pnl.append(net_pnl)
        self._save()

    def observe_cooldown_trigger(self, sym: str) -> bool:
        """
        每次 per-symbol cooldown 觸發後呼叫。
        若 7d 內 cooldown 達到閾值 → 升級為長期 jail。
        返回 True = 此次升級為 jail。
        """
        now = time.time()
        st = self._state(sym)
        self._trim_cooldown_ts(st, now)
        st.cooldown_ts.append(now)

        if st.is_jailed:
            self._save()
            return False

        if len(st.cooldown_ts) >= self.jail_cooldown_count:
            self._put_in_jail(
                sym, st,
                reason="cooldown_repeated",
                detail=f"{len(st.cooldown_ts)} cooldowns in 7d",
            )
            return True

        self._save()
        return False

    # ── Release Check（main loop 每輪呼叫）──────────────────────────────────

    def check_release(
        self,
        sym: str,
        vr: Optional[float],
        half_life: Optional[float],
    ) -> bool:
        """
        嘗試釋放單一幣種。
        - 未坐滿 min sentence → False（唔 log）
        - 達 max sentence → 強制釋放（forced）
        - VR & HL 達標 → 自動釋放（auto）
        - 否則延長 1 日

        返回 True = 此次成功釋放。
        """
        st = self._state(sym)
        if not st.is_jailed:
            return False

        now = time.time()
        elapsed = now - (st.jailed_at or now)

        # ── 硬上限：強制釋放 ──
        if elapsed >= self.jail_max_sec:
            self._release(
                sym, st, kind="forced",
                detail=f"max sentence {self.jail_max_sec/86400:.0f}d reached",
            )
            return True

        # ── 未到最早釋放時間 ──
        if st.release_after is not None and now < st.release_after:
            return False

        # ── 防 spam：每 1h check 一次 ──
        last = self._last_release_check.get(sym, 0.0)
        if now - last < self._release_check_interval:
            return False
        self._last_release_check[sym] = now

        # ── 達標 check ──
        vr_ok = vr is not None and vr < self.vr_release_threshold
        hl_ok = (
            half_life is not None
            and half_life > 0
            and half_life <= self.hl_release_max_bars
        )

        if vr_ok and hl_ok:
            self._release(
                sym, st, kind="auto",
                detail=f"VR={vr:.3f}<{self.vr_release_threshold} HL={half_life:.1f}≤{self.hl_release_max_bars:.0f}",
            )
            return True

        # ── 唔達標 → 延長 1 日 ──
        st.extends += 1
        st.release_after = now + _1D
        vr_str = f"{vr:.3f}" if vr is not None else "n/a"
        hl_str = f"{half_life:.1f}" if half_life is not None else "n/a"
        logger.warning(
            "🔒 JAIL EXTEND  %s  → +1d (extends=%d)  VR=%s HL=%s  "
            "(need VR<%.2f & 0<HL≤%.0f)",
            sym, st.extends, vr_str, hl_str,
            self.vr_release_threshold, self.hl_release_max_bars,
        )
        self._save()
        return False

    def _release(self, sym: str, st: JailState, kind: str, detail: str) -> None:
        """執行釋放動作，更新 history。"""
        now = time.time()
        served_days = (now - (st.jailed_at or now)) / 86400
        log_fn = logger.warning if kind == "forced" else logger.info
        prefix = "⚠️ JAIL FORCED" if kind == "forced" else "🔓 JAIL RELEASED"
        log_fn(
            "%s  %s  served=%.1fd  reason=%s [%s]  trigger_was=%s",
            prefix, sym, served_days, kind, detail, st.trigger_reason,
        )
        st.history.append({
            "jailed_at":     st.jailed_at,
            "released_at":   now,
            "served_days":   round(served_days, 2),
            "trigger":       st.trigger_reason,
            "release_kind":  kind,
            "release_detail": detail,
        })
        st.jailed_at      = None
        st.release_after  = None
        st.trigger_reason = ""
        st.trigger_detail = ""
        st.extends        = 0
        # 釋放後保守 reset：俾 fresh start，但保留 history 供統計
        st.consecutive_sl = 0
        st.cooldown_ts    = []
        st.recent_pnl.clear()
        self._save()

    # ── Query API ────────────────────────────────────────────────────────────

    def is_jailed(self, sym: str) -> bool:
        """主 entry gate：True = 禁止開新倉。"""
        return self._state(sym).is_jailed

    def all_jailed(self) -> List[str]:
        return [sym for sym, st in self._states.items() if st.is_jailed]

    def remaining_days(self, sym: str) -> float:
        """返回距離下次可 check release 的日數（0 = 隨時可 check）。"""
        st = self._state(sym)
        if not st.is_jailed or st.release_after is None:
            return 0.0
        return max(0.0, (st.release_after - time.time()) / 86400)

    def status_dict(self, sym: str) -> Dict:
        """單一 symbol 嘅狀態 dict。"""
        st = self._state(sym)
        out: Dict = {
            "jailed":         st.is_jailed,
            "consecutive_sl": st.consecutive_sl,
            "cooldown_count_7d": len(st.cooldown_ts),
            "recent_pnl_n":   len(st.recent_pnl),
            "recent_pnl_sum": round(sum(st.recent_pnl), 4) if st.recent_pnl else 0.0,
        }
        if st.is_jailed:
            out["trigger"]    = st.trigger_reason
            out["detail"]     = st.trigger_detail
            out["extends"]    = st.extends
            out["wait_days"]  = round(self.remaining_days(sym), 2)
        return out

    def log_status(self) -> None:
        """啟動時 / 定期匯總所有坐監狀態。"""
        now = time.time()
        jailed = []
        warned = []   # 接近邊緣（streak / cooldown 接近觸發）

        for sym, st in self._states.items():
            if st.is_jailed:
                wait = max(0.0, ((st.release_after or now) - now) / 86400)
                served = (now - (st.jailed_at or now)) / 86400
                jailed.append((sym, served, wait, st.trigger_reason, st.extends))
            else:
                self._trim_cooldown_ts(st, now)
                # 接近邊緣
                if (st.consecutive_sl >= self.jail_sl_streak - 1
                    or len(st.cooldown_ts) >= self.jail_cooldown_count - 1):
                    warned.append((sym, st.consecutive_sl, len(st.cooldown_ts)))

        logger.info("=" * 60)
        logger.info("JAIL STATUS")
        logger.info(
            "  入監閾值：cooldown≥%d/7d | SL_streak≥%d | last_%d_PnL<%.2f",
            self.jail_cooldown_count, self.jail_sl_streak,
            self.jail_pnl_lookback, self.jail_pnl_threshold,
        )
        logger.info(
            "  釋放條件：≥%.0fd + VR<%.2f + 0<HL≤%.0f  (max=%.0fd)",
            self.jail_min_sec / 86400, self.vr_release_threshold,
            self.hl_release_max_bars, self.jail_max_sec / 86400,
        )

        if jailed:
            logger.warning("  🔒 坐監中 (%d 隻)：", len(jailed))
            for sym, served, wait, reason, ext in jailed:
                logger.warning(
                    "    %-25s  served=%.1fd  next_check_in=%.1fd  "
                    "trigger=%s  extends=%d",
                    sym, served, wait, reason, ext,
                )
        else:
            logger.info("  ✅ 無幣種坐監")

        if warned:
            logger.warning("  ⚠️  接近入監邊緣：")
            for sym, sl_n, cd_n in warned:
                logger.warning(
                    "    %-25s  SL_streak=%d/%d  cooldown_7d=%d/%d",
                    sym, sl_n, self.jail_sl_streak,
                    cd_n, self.jail_cooldown_count,
                )

        if not jailed and not warned:
            logger.info("  所有幣種正常")
        logger.info("=" * 60)
