from typing import Dict, List, Optional, Any

from scipy.signal import find_peaks
import numpy as np
import pandas as pd


# --------------------------- 通用指标 --------------------------- #

def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)

    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100

    K = np.zeros_like(rsv, dtype=float)
    D = np.zeros_like(rsv, dtype=float)
    for i in range(len(df)):
        if i == 0:
            K[i] = D[i] = 50.0
        else:
            K[i] = 2 / 3 * K[i - 1] + 1 / 3 * rsv.iloc[i]
            D[i] = 2 / 3 * D[i - 1] + 1 / 3 * K[i]
    J = 3 * K - 2 * D
    return df.assign(K=K, D=D, J=J)


def compute_bbi(df: pd.DataFrame) -> pd.Series:
    ma3 = df["close"].rolling(3).mean()
    ma6 = df["close"].rolling(6).mean()
    ma12 = df["close"].rolling(12).mean()
    ma24 = df["close"].rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4


def compute_rsv(
    df: pd.DataFrame,
    n: int,
) -> pd.Series:
    """
    按公式：RSV(N) = 100 × (C - LLV(L,N)) ÷ (HHV(C,N) - LLV(L,N))
    - C 用收盘价最高值 (HHV of close)
    - L 用最低价最低值 (LLV of low)
    """
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_close_n = df["close"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_close_n - low_n + 1e-9) * 100.0
    return rsv


def compute_dif(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """计算 MACD 指标中的 DIF (EMA fast - EMA slow)。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def bbi_deriv_uptrend(
    bbi: pd.Series,
    *,
    min_window: int,
    max_window: int | None = None,
    q_threshold: float = 0.0,
) -> bool:
    """
    判断 BBI 是否“整体上升”。

    令最新交易日为 T，在区间 [T-w+1, T]（w 自适应，w ≥ min_window 且 ≤ max_window）
    内，先将 BBI 归一化：BBI_norm(t) = BBI(t) / BBI(T-w+1)。

    再计算一阶差分 Δ(t) = BBI_norm(t) - BBI_norm(t-1)。  
    若 Δ(t) 的前 q_threshold 分位数 ≥ 0，则认为该窗口通过；只要存在
    **最长** 满足条件的窗口即可返回 True。q_threshold=0 时退化为
    “全程单调不降”（旧版行为）。

    Parameters
    ----------
    bbi : pd.Series
        BBI 序列（最新值在最后一位）。
    min_window : int
        检测窗口的最小长度。
    max_window : int | None
        检测窗口的最大长度；None 表示不设上限。
    q_threshold : float, default 0.0
        允许一阶差分为负的比例（0 ≤ q_threshold ≤ 1）。
    """
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold 必须位于 [0, 1] 区间内")

    bbi = bbi.dropna()
    if len(bbi) < min_window:
        return False

    longest = min(len(bbi), max_window or len(bbi))

    # 自最长窗口向下搜索，找到任一满足条件的区间即通过
    for w in range(longest, min_window - 1, -1):
        seg = bbi.iloc[-w:]                # 区间 [T-w+1, T]
        norm = seg / seg.iloc[0]           # 归一化
        diffs = np.diff(norm.values)       # 一阶差分
        if np.quantile(diffs, q_threshold) >= 0:
            return True
    return False


def _find_peaks(
    df: pd.DataFrame,
    *,
    column: str = "high",
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
    rel_height: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    
    if column not in df.columns:
        raise KeyError(f"'{column}' not found in DataFrame columns: {list(df.columns)}")

    y = df[column].to_numpy()

    indices, props = find_peaks(
        y,
        distance=distance,
        prominence=prominence,
        height=height,
        width=width,
        rel_height=rel_height,
        **kwargs,
    )

    peaks_df = df.iloc[indices].copy()
    peaks_df["is_peak"] = True

    # Flatten SciPy arrays into columns (only those with same length as indices)
    for key, arr in props.items():
        if isinstance(arr, (list, np.ndarray)) and len(arr) == len(indices):
            peaks_df[f"peak_{key}"] = arr

    return peaks_df


# --------------------------- Selector 类 --------------------------- #
class BBIKDJSelector:
    """
    自适应 *BBI(导数)* + *KDJ* 选股器
        • BBI: 允许 50% 比例的回撤
        • KDJ: J < 13 ；或位于历史 J 的 10% 分位及以下
        • MACD: DIF > 0
        • 收盘价波动幅度 ≤ price_range_pct
        • 当日K线涨幅在 -2% ~ +2%
        • 当日K线振幅在 +-7%之间
        • 白线 在黄线上面
        • 当日收盘价在黄线上面
    """

    def __init__(
        self,
        j_threshold: float = -5,
        bbi_min_window: int = 90,
        max_window: int = 90,
        price_range_pct: float = 100.0,
        bbi_q_threshold: float = 0.05,
        j_q_threshold: float = 0.10,
        zdf_low: float = -2.0,
        zdf_high: float = 2.0,
        zd_main: float = 4.0,
        zd_gem: float = 7.0,
        m1: int = 14,
        m2: int = 28,
        m3: int = 57,
        m4: int = 114,
    ) -> None:
        self.j_threshold = j_threshold  # J 值进入超卖区的绝对阈值 (10)
        self.bbi_min_window = bbi_min_window  # BBI 趋势判断的最小窗口 (20天)
        self.max_window = max_window  # 最大回溯窗口 (60天，中期的时间尺度)
        self.price_range_pct = price_range_pct  # 回溯窗口内，收盘价波动幅度限制 (100%)
        self.bbi_q_threshold = bbi_q_threshold  # BBI 趋势判断内，BBI 趋势回撤容忍度 (50%)
        self.j_q_threshold = j_q_threshold  # J 值回溯窗口内 历史分位数 (10%)
        self.zdf_low = zdf_low  # 当日跌幅阈值 (-2%)
        self.zdf_high = zdf_high  # 当日涨幅阈值 (+2%)
        self.zd_main = zd_main  # 主板 当日振幅阈值 (4%)
        self.zd_gem = zd_gem  #  创业板/科创板/北交所 当日振幅阈值 (7%)
        self.m1 = m1  # 14
        self.m2 = m2  # 28
        self.m3 = m3  # 57
        self.m4 = m4  # 114

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame, code: str) -> bool:
        if len(hist) < 2:
            return False
            
        hist = hist.copy()  # 历史窗口内的行情数据（开盘、收盘、最高、最低价等）
        hist["BBI"] = compute_bbi(hist)  # 该窗口内每天的 BBI 值

        # 0. 回溯窗口期内 收盘价波动幅度约束（回溯窗口内，收盘价波动幅度 <= 100%，排除“疯牛”和“死鱼”）
        win = hist.tail(self.max_window)  # 最近 max_window 天内的行情数据
        high, low = win["close"].max(), win["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):
            return False

        # 2. KDJ 过滤 —— (当日 J 值 <= 13 或 <= 10% 分位数)
        kdj = compute_kdj(hist)  # 历史窗口内每日的 KDJ 值
        j_today = float(kdj.iloc[-1]["J"])  # 当日的 J 值
        # 最近 60 天 K 线的 J 分位
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))  # 最近 60 天中，J 值的 j_q_threshold(10%) 分位数（即有10%的 J 值 <= 该数值）
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        # 3. MACD：DIF > 0
        hist["DIF"] = compute_dif(hist)
        if hist["DIF"].iloc[-1] <= 0:
            return False

        # 4. 当日涨跌幅
        close_today = hist['close'].iloc[-1]
        close_yesterday = hist['close'].iloc[-2]
        zdf = (close_today - close_yesterday) / close_yesterday * 100
        if not (self.zdf_low <= zdf <= self.zdf_high):
            return False

        # 5. 当日振幅
        zd = (hist['high'].iloc[-1] - hist['low'].iloc[-1]) / close_yesterday * 100
        is_main_board = code.startswith('60') or code.startswith('00')
        is_gem_board = code.startswith('300') or code.startswith('688') or code.startswith('8')
        
        zd_cond = False
        if is_main_board and zd <= self.zd_main:
            zd_cond = True
        elif is_gem_board and zd <= self.zd_gem:
            zd_cond = True
        
        if not zd_cond:
            return False

        # 6. 知行趋势 & 收盘价 > 黄线
        st = hist["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
        ma1 = hist["close"].rolling(self.m1).mean()
        ma2 = hist["close"].rolling(self.m2).mean()
        ma3 = hist["close"].rolling(self.m3).mean()
        ma4 = hist["close"].rolling(self.m4).mean()
        lt = (ma1 + ma2 + ma3 + ma4) / 4
        
        if st.iloc[-1] <= lt.iloc[-1] or close_today <= lt.iloc[-1]:
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 额外预留 20 根 K 线缓冲, 增加 M4 的长度
            required_len = int(max(self.max_window + 20, self.m4 + 1))
            hist = hist.tail(required_len)
            if self._passes_filters(hist, code):
                picks.append(code)
        return picks
    
    
class SuperB1Selector:
    """SuperB1 选股器

    过滤逻辑概览
    ----------------
    1. **历史匹配 (t_m)** — 在 *lookback_n* 个交易日窗口内，至少存在一日
       满足 :class:`BBIKDJSelector`。

    2. **盘整区间** — 区间 ``[t_m, date-1]`` 收盘价波动率不超过 ``close_vol_pct``。

    3. **当日下跌** — ``(close_{date-1} - close_date) / close_{date-1}``
       ≥ ``price_drop_pct``。

    4. **J 值极低** — ``J < j_threshold`` *或* 位于历史 ``j_q_threshold`` 分位。
    """

    # ---------------------------------------------------------------------
    # 构造函数
    # ---------------------------------------------------------------------
    def __init__(
        self,
        *,
        lookback_n: int = 60,
        close_vol_pct: float = 0.05,
        price_drop_pct: float = 0.03,
        j_threshold: float = -5,
        j_q_threshold: float = 0.10,
        # ↓↓↓ 新增：嵌套 BBIKDJSelector 配置
        B1_params: Optional[Dict[str, Any]] = None
    ) -> None:
        # ---------- 参数合法性检查 ----------
        if lookback_n < 2:
            raise ValueError("lookback_n 应 ≥ 2")
        if not (0 < close_vol_pct < 1):
            raise ValueError("close_vol_pct 应位于 (0, 1) 区间")
        if not (0 < price_drop_pct < 1):
            raise ValueError("price_drop_pct 应位于 (0, 1) 区间")
        if not (0 <= j_q_threshold <= 1):
            raise ValueError("j_q_threshold 应位于 [0, 1] 区间")
        if B1_params is None:
            raise ValueError("bbi_params没有给出")

        # ---------- 基本参数 ----------
        self.lookback_n = lookback_n          # 回溯窗口 (10天)
        self.close_vol_pct = close_vol_pct    # 盘整期内的 收盘价波动幅度限制 (2%)
        self.price_drop_pct = price_drop_pct  # 当日收盘价相对前一日下跌的幅度阈值 (2%)
        self.j_threshold = j_threshold        # 当日 J 值进入超卖区的绝对阈值 (10)
        self.j_q_threshold = j_q_threshold    # J 值回溯窗口内 历史分位数 (10%)

        # ---------- 内部 BBIKDJSelector ----------
        self.bbi_selector = BBIKDJSelector(**(B1_params or {}))

        # 为保证给 BBIKDJSelector 提供足够历史，预留额外缓冲
        self._extra_for_bbi = self.bbi_selector.max_window + 20

    # 单支股票过滤核心
    def _passes_filters(self, hist: pd.DataFrame, code: str) -> bool:
        """*hist* 必须按日期升序，且最后一行为目标 *date*。"""
        if len(hist) < 2:
            return False

        # ---------- Step-0: 数据量判断 ----------
        if len(hist) < self.lookback_n + self._extra_for_bbi:
            return False

        # ---------- Step-1: 搜索满足 BBIKDJ 的 t_m (前 11 天内找到一个B1日 && 该日到前一日的盘整期>=3天 && 盘整期内的收盘价波动<=2%) ----------
        lb_hist = hist.tail(self.lookback_n + 1)  # 前 11 天的行情数据
        tm_idx: int | None = None
        # 遍历前 11 天的回溯窗口
        for idx in lb_hist.index[:-1]:
            if self.bbi_selector._passes_filters(hist.loc[:idx], code):  # 如果该日到 B1
                tm_idx = idx
                stable_seg = hist.loc[tm_idx : hist.index[-2], "close"]  # 获取该日 到 前一日（即“盘整期”）的 收盘价
                if len(stable_seg) < 3:
                    tm_idx = None
                    break
                # 盘整期需 >= 3天
                high, low = stable_seg.max(), stable_seg.min()
                if low <= 0 or (high / low - 1) > self.close_vol_pct:  # 如果盘整期内 收盘价波动 > 2%，则淘汰
                    tm_idx = None
                    continue
                else:
                    break
        if tm_idx is None:
            return False

        # ---------- Step-3: 当日收盘价相对于前一日的跌幅 >= 2% (今天在“挖坑”) ----------
        close_today, close_prev = hist["close"].iloc[-1], hist["close"].iloc[-2]
        if close_prev <= 0 or (close_prev - close_today) / close_prev < self.price_drop_pct:
            return False

        # ---------- Step-4: 当日 J 值极低 (<= 10 或 <= 10% 分位数) ----------
        kdj = compute_kdj(hist)
        j_today = float(kdj["J"].iloc[-1])
        j_window = kdj["J"].iloc[-self.lookback_n:].dropna()
        j_q_val = float(j_window.quantile(self.j_q_threshold)) if not j_window.empty else np.nan
        if not (j_today < self.j_threshold or j_today <= j_q_val):
            return False

        return True

    # 批量选股接口
    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks: List[str] = []
        min_len = self.lookback_n + self._extra_for_bbi

        for code, df in data.items():
            hist = df[df["date"] <= date].tail(min_len)
            if len(hist) < min_len:
                continue
            if self._passes_filters(hist, code):
                picks.append(code)

        return picks


class PeakKDJSelector:
    """
    Peaks + KDJ 选股器    
    """

    def __init__(
        self,
        j_threshold: float = -5,
        max_window: int = 90,
        fluc_threshold: float = 0.03,
        gap_threshold: float = 0.02,
        j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.max_window = max_window
        self.fluc_threshold = fluc_threshold  # 当日↔peak_(t-n) 波动率上限
        self.gap_threshold = gap_threshold    # oc_prev 必须高于区间最低收盘价的比例
        self.j_q_threshold = j_q_threshold

    # ---------- 单支股票过滤 ---------- #
        # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if hist.empty:
            return False

        hist = hist.copy().sort_values("date")
        hist["oc_max"] = hist[["open", "close"]].max(axis=1)

        # 1. 提取 peaks
        peaks_df = _find_peaks(
            hist,
            column="oc_max",
            distance=6,
            prominence=0.5,
        )
        
        # 至少两个峰      
        date_today = hist.iloc[-1]["date"]
        peaks_df = peaks_df[peaks_df["date"] < date_today]
        if len(peaks_df) < 2:               
            return False

        peak_t = peaks_df.iloc[-1]          # 最新一个峰
        peaks_list = peaks_df.reset_index(drop=True)
        oc_t = peak_t.oc_max
        total_peaks = len(peaks_list)

        # 2. 回溯寻找 peak_(t-n)
        target_peak = None        
        for idx in range(total_peaks - 2, -1, -1):
            peak_prev = peaks_list.loc[idx]
            oc_prev = peak_prev.oc_max
            if oc_t <= oc_prev:             # 要求 peak_t > peak_(t-n)
                continue

            # 只有当“总峰数 ≥ 3”时才检查区间内其他峰 oc_max
            if total_peaks >= 3 and idx < total_peaks - 2:
                inter_oc = peaks_list.loc[idx + 1 : total_peaks - 2, "oc_max"]
                if not (inter_oc < oc_prev).all():
                    continue

            # 新增： oc_prev 高于区间最低收盘价 gap_threshold
            date_prev = peak_prev.date
            mask = (hist["date"] > date_prev) & (hist["date"] < peak_t.date)
            min_close = hist.loc[mask, "close"].min()
            if pd.isna(min_close):
                continue                    # 区间无数据
            if oc_prev <= min_close * (1 + self.gap_threshold):
                continue

            target_peak = peak_prev
            
            break

        if target_peak is None:
            return False

        # 3. 当日收盘价波动率
        close_today = hist.iloc[-1]["close"]
        fluc_pct = abs(close_today - target_peak.close) / target_peak.close
        if fluc_pct > self.fluc_threshold:
            return False

        # 4. KDJ 过滤
        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            hist = hist.tail(self.max_window + 20)  # 额外缓冲
            if self._passes_filters(hist):
                picks.append(code)
        return picks
    

class BBIShortLongSelector:
    """
    BBI 上升 + 短/长期 RSV 条件 + DIF > 0 选股器
    """
    def __init__(
        self,
        n_short: int = 3,
        n_long: int = 21,
        m: int = 3,
        bbi_min_window: int = 90,
        max_window: int = 150,
        bbi_q_threshold: float = 0.05,
    ) -> None:
        if m < 2:
            raise ValueError("m 必须 ≥ 2")
        self.n_short = n_short
        self.n_long = n_long
        self.m = m
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.bbi_q_threshold = bbi_q_threshold   # 新增参数

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)

        # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):
            return False

        # 2. 计算短/长期 RSV -----------------
        hist["RSV_short"] = compute_rsv(hist, self.n_short)
        hist["RSV_long"] = compute_rsv(hist, self.n_long)

        if len(hist) < self.m:
            return False                        # 数据不足

        win = hist.iloc[-self.m :]              # 最近 m 天
        long_ok = (win["RSV_long"] >= 80).all() # 长期 RSV 全 ≥ 80

        short_series = win["RSV_short"]
        short_start_end_ok = (
            short_series.iloc[0] >= 80 and short_series.iloc[-1] >= 80
        )
        short_has_below_20 = (short_series < 20).any()

        if not (long_ok and short_start_end_ok and short_has_below_20):
            return False

        # 3. MACD：DIF > 0 -------------------
        hist["DIF"] = compute_dif(hist)
        return hist["DIF"].iloc[-1] > 0

    # ---------- 多股票批量 ---------- #
    def select(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 预留足够长度：RSV 计算窗口 + BBI 检测窗口 + m
            need_len = (
                max(self.n_short, self.n_long)
                + self.bbi_min_window
                + self.m
            )
            hist = hist.tail(max(need_len, self.max_window))
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BreakoutVolumeKDJSelector:
    """
    放量突破 + KDJ + DIF>0 + 收盘价波动幅度 选股器   
    """

    def __init__(
        self,
        j_threshold: float = 0.0,
        up_threshold: float = 3.0,
        volume_threshold: float = 2.0 / 3,
        offset: int = 15,
        max_window: int = 120,
        price_range_pct: float = 10.0,
        j_q_threshold: float = 0.10,        # ← 新增
    ) -> None:
        self.j_threshold = j_threshold
        self.up_threshold = up_threshold
        self.volume_threshold = volume_threshold
        self.offset = offset
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.j_q_threshold = j_q_threshold  # ← 新增

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if len(hist) < self.offset + 2:
            return False

        hist = hist.tail(self.max_window).copy()

        # ---- 收盘价波动幅度约束 ----
        high, low = hist["close"].max(), hist["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        # ---- 技术指标 ----
        hist = compute_kdj(hist)
        hist["pct_chg"] = hist["close"].pct_change() * 100
        hist["DIF"] = compute_dif(hist)

        # 0) 指定日约束：J < j_threshold 或位于历史分位；且 DIF > 0
        j_today = float(hist["J"].iloc[-1])

        j_window = hist["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        # 若不满足任一 J 条件，则淘汰
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False
        if hist["DIF"].iloc[-1] <= 0:
            return False

        # ---- 放量突破条件 ----
        n = len(hist)
        wnd_start = max(0, n - self.offset - 1)
        last_idx = n - 1

        for t_idx in range(wnd_start, last_idx):  # 探索突破日 T
            row = hist.iloc[t_idx]

            # 1) 单日涨幅
            if row["pct_chg"] < self.up_threshold:
                continue

            # 2) 相对放量
            vol_T = row["volume"]
            if vol_T <= 0:
                continue
            vols_except_T = hist["volume"].drop(index=hist.index[t_idx])
            if not (vols_except_T <= self.volume_threshold * vol_T).all():
                continue

            # 3) 创新高
            if row["close"] <= hist["close"].iloc[:t_idx].max():
                continue

            # 4) T 之后 J 值维持高位
            if not (hist["J"].iloc[t_idx:last_idx] > hist["J"].iloc[-1] - 10).all():
                continue

            return True  # 满足所有条件

        return False

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            if self._passes_filters(hist):
                picks.append(code)
        return picks
