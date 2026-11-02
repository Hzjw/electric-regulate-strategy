"""读入附件1~4，装配单日/多日调度 LP 并求解。"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

DT       = 1.0 / 6.0     # 单段时长(小时)，10 分钟
N_T      = 144           # 一天的段数
N_DAY    = 365           # 附件2/3/4 覆盖天数
# 往返效率 0.9，充放各 √0.9
ETA_C    = 0.9 ** 0.5    # 充电效率 = sqrt(0.9)
ETA_D    = 0.9 ** 0.5    # 放电效率 = sqrt(0.9)
P_MAX    = 5000.0        # 最大充放电功率 (kW)
E_MAX    = P_MAX * DT    # 单段最大充/放电电量 (kWh)
SOC_MIN  = 1200.0        # 储电量运行下界 (kWh)
SOC_MAX  = 10800.0       # 储电量运行上界 (kWh)
SOC_INIT = 6000.0        # 2025-01-01 00:00 初始储电量 (kWh)

# 末端储电量下界
SOC_TERM = 6000.0

# 吞吐量正则化：取充放电最少的解
THRU_PEN = 1e-6

BETA_EMG = 5.0           # 紧急购电电价倍数
BETA_DEF = 0.5           # 违约（少买）电价倍数
BETA_EXC = 1.5           # 超额（多买）电价倍数

# 报告期 2025-02-01 ~ 12-31
REPORT_START = pd.Timestamp("2025-02-01")
REPORT_END   = pd.Timestamp("2025-12-31")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT  = os.path.join(ROOT, "data")
RES  = os.path.join(ROOT, "results")
IMG  = os.path.join(ROOT, "figures")
for folder in (RES, IMG):
    os.makedirs(folder, exist_ok=True)

# 六大时段口径：每段 4 小时 = 24 段
BLOCKS = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
          ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]


def load_all(verbose=True):
    """读入全部附件，返回各数据 dict。"""
    df1 = pd.read_excel(os.path.join(ATT, "附件1.xlsx"))
    # 显式转 float
    typical = {
        "price":   pd.to_numeric(df1["电价"], errors="raise").to_numpy(float),
        "load_kw": pd.to_numeric(df1["小区负载"], errors="raise").to_numpy(float),
        "pv_kw":   pd.to_numeric(df1["光伏发电预测功率"], errors="raise").to_numpy(float),
    }

    xl = pd.ExcelFile(os.path.join(ATT, "附件2.xlsx"))
    ld = xl.parse("小区负载", header=0)
    pv = xl.parse("光伏发电实际功率", header=0)
    dates = pd.to_datetime(ld.iloc[:, 0]).reset_index(drop=True)
    actual = {
        "dates":   dates,
        "load_kw": ld.iloc[:, 1:1 + N_T].apply(pd.to_numeric, errors="raise").to_numpy(float),
        # 光伏负值截零（夜间浮点噪声）
        "pv_kw":   np.clip(pv.iloc[:, 1:1 + N_T].apply(pd.to_numeric, errors="raise")
                           .to_numpy(float), 0.0, None),
    }

    df4 = pd.read_excel(os.path.join(ATT, "附件4.xlsx"))
    price_rt = {
        "dates": pd.to_datetime(df4.iloc[:, 0]).reset_index(drop=True),
        "price": df4.iloc[:, 1:1 + N_T].apply(pd.to_numeric, errors="raise").to_numpy(float),
    }

    # 附件3：全年光伏预报 365x4x24，第2维为发布时刻
    df3 = pd.read_excel(os.path.join(ATT, "附件3.xlsx"))
    date_col = df3.iloc[:, 0].ffill()      # 日期列向下填充
    time_col = df3.iloc[:, 1].astype(str).str.strip()
    stamps = pd.to_datetime(date_col.astype(str).str.strip(), format="mixed", errors="coerce")
    days = pd.DatetimeIndex(sorted(stamps.dropna().unique()))
    day_index = {day: i for i, day in enumerate(days)}
    fc_3d = np.full((N_DAY, 4, 24), np.nan)
    for row in range(len(df3)):
        day = stamps.iloc[row]
        if pd.isna(day):
            continue
        label = time_col.iloc[row]
        if label in ("0:00", "6:00", "12:00", "18:00"):
            fc_3d[day_index[day], ("0:00", "6:00", "12:00", "18:00").index(label), :] = \
                pd.to_numeric(df3.iloc[row, 2:26], errors="coerce").to_numpy(float)
    forecast = {"dates": days, "fc": fc_3d}

    if verbose:
        print("典型日", typical["price"].shape, "附件2", actual["load_kw"].shape,
              "附件3", fc_3d.shape, "附件4", price_rt["price"].shape)
    return {"typical": typical, "actual": actual, "forecast": forecast, "price_rt": price_rt}


def forecast_to_10min(fc_day, k, method="step"):
    """第 k 个发布时刻 24 点预报还原为 144 段 10 分钟。"""
    base = (0, 6, 12, 18)[k]
    vals = np.asarray(fc_day[k], float)
    hourly = np.zeros(24)
    # j = h + 1 - base（时间戳为区间终点）
    for h in range(24):
        j = (h + 1 - base) % 24
        if j == 0:
            j = 24           # 发布时刻小时取"预报24小时"
        hourly[h] = vals[j - 1]
    if method == "step":
        # 段 t 覆盖 ((t-1)/6, t/6]，整点取左端点
        idx = np.floor(np.arange(N_T) * DT).astype(int)
        return hourly[np.clip(idx, 0, 23)]
    elif method == "linear":
        grid = np.arange(1, N_T + 1) * DT
        xp = np.concatenate([[0.0], np.arange(1, 24).astype(float), [24.0]])
        fp = np.concatenate([[hourly[0]], hourly[1:24], [hourly[0]]])
        return np.interp(grid, xp, fp)
    raise ValueError("method 只能是 'step' 或 'linear'")


def solve_day_lp(price, load_kw, pv_kw, soc_start=SOC_INIT, soc_end=None,
                 use_grid=True, verbose=False, c_cyc=0.0):
    """单日调度 LP：未知量分 5 块 g,c,d,s,E。"""
    n = N_T
    # 目标系数：购电项 + 循环成本
    c_obj = np.concatenate([price, np.zeros(4 * n)])
    if c_cyc:
        c_obj[2 * n:3 * n] = c_cyc        # 第 3 个变量块 = 放电量 d_t

    rows, cols, vals, b_eq = [], [], [], []

    # 母线平衡 g_t + pv*dt + d_t = L*dt + c_t + s_t
    for t in range(n):
        r = t
        rows += [r, r, r, r]; cols += [t, n + t, 2 * n + t, 3 * n + t]
        vals += [1.0, -1.0, 1.0, -1.0]
        b_eq.append(load_kw[t] * DT - pv_kw[t] * DT)

    # SOC 递推 E_t = E_{t-1} + eta_c*c_t - d_t/eta_d
    for t in range(n):
        r = n + t
        rows.append(r); cols.append(4 * n + t); vals.append(1.0)
        if t == 0:
            b_eq.append(soc_start)        # E_0 - eta_c*c_0 + d_0/eta_d = E_init
        else:
            rows.append(r); cols.append(4 * n + t - 1); vals.append(-1.0)
            b_eq.append(0.0)
        rows.append(r); cols.append(n + t); vals.append(-ETA_C)
        rows.append(r); cols.append(2 * n + t); vals.append(1.0 / ETA_D)

    A_eq = coo_matrix((vals, (rows, cols)), shape=(2 * n, 5 * n)).tocsr()

    lb = np.zeros(5 * n)
    ub = np.empty(5 * n)
    ub[0:n]         = np.inf                  # 购电量只买不卖
    ub[n:2 * n]     = E_MAX                   # 充电功率折算上限
    ub[2 * n:3 * n] = E_MAX                   # 放电功率折算上限
    ub[3 * n:4 * n] = pv_kw * DT              # 弃光不得超过光伏发电量
    ub[4 * n:5 * n] = SOC_MAX
    lb[4 * n:5 * n] = SOC_MIN
    if soc_end is not None:
        # 末端储电量收紧上下界
        ub[4 * n + n - 1] = soc_end
        lb[4 * n + n - 1] = soc_end

    bounds = list(zip(lb, ub))

    res = linprog(c_obj, A_eq=A_eq, b_eq=np.array(b_eq), bounds=bounds,
                  method="highs")         # HiGHS 求解器

    if not res.success:
        return {"status": res.status, "message": res.message}

    x = res.x
    out = {
        "status": res.status, "message": res.message, "cost": float(res.fun),
        "g": x[0:n].copy(), "c": x[n:2 * n].copy(), "d": x[2 * n:3 * n].copy(),
        "s": x[3 * n:4 * n].copy(), "E": x[4 * n:5 * n].copy(),
        # 平衡约束对偶：边际供电成本
        "dual_balance": (res.eqlin.marginals[:n].copy()
                         if getattr(res, "eqlin", None) is not None else None),
    }
    if verbose:
        print("单日LP", "status", res.status, "费用", "%.2f" % out["cost"],
              "购电量", "%.1f" % out["g"].sum(), "弃光", "%.1f" % out["s"].sum())
    return out


def block_sum(arr, lo, hi):
    """按 4 小时区块求和（result 模板口径）。"""
    return sum(arr[lo:hi] for (_, lo, hi) in BLOCKS)


def block_sums(arr):
    """6 个 4 小时区块的合计（元组）。"""
    return [float(arr[lo:hi].sum()) for (_, lo, hi) in BLOCKS]


def solve_multiday_lp(price_2d, load_2d, pv_2d, soc_init=SOC_INIT,
                      use_emg=True, n_days=None, verbose=True, c_cyc=0.0):
    """多日耦合调度 LP：展平 i=d*144+t，返回 (D,144)。"""
    n_day = price_2d.shape[0] if n_days is None else n_days
    price = price_2d[:n_day].ravel()
    load = load_2d[:n_day].ravel()
    pv = pv_2d[:n_day].ravel()
    n_seg = n_day * N_T
    n_blk = 6 if use_emg else 5       # 变量块数：g,c,d,s,E(,e)

    off_g, off_c, off_d, off_s, off_e = 0, n_seg, 2 * n_seg, 3 * n_seg, 4 * n_seg
    off_e2 = 5 * n_seg if use_emg else None

    c_obj = np.zeros(n_blk * n_seg)
    c_obj[off_g:off_g + n_seg] = price
    if use_emg:
        c_obj[off_e2:off_e2 + n_seg] = BETA_EMG * price
    if c_cyc:
        # 循环成本按放电量计提
        c_obj[off_d:off_d + n_seg] = c_cyc

    idx = np.arange(n_seg)
    rows, cols, vals, b_eq = [], [], [], []

    # 母线平衡，n_seg 条等式
    rows.append(idx);            cols.append(off_g + idx); vals.append(np.ones(n_seg))
    rows.append(idx);            cols.append(off_c + idx); vals.append(-np.ones(n_seg))
    rows.append(idx);            cols.append(off_d + idx); vals.append(np.ones(n_seg))
    rows.append(idx);            cols.append(off_s + idx); vals.append(-np.ones(n_seg))
    if use_emg:
        rows.append(idx);        cols.append(off_e2 + idx); vals.append(np.ones(n_seg))
    b_eq.append(load * DT - pv * DT)

    # SOC 递推：n_seg 条等式，初始值在右端项
    soc_r = n_seg + idx
    rows.append(soc_r);      cols.append(off_e + idx); vals.append(np.ones(n_seg))
    rows.append(soc_r);      cols.append(off_c + idx); vals.append(-ETA_C * np.ones(n_seg))
    rows.append(soc_r);      cols.append(off_d + idx); vals.append((1.0 / ETA_D) * np.ones(n_seg))
    rows.append(soc_r[1:]);  cols.append(off_e + idx[1:] - 1); vals.append(-np.ones(n_seg - 1))
    b_eq.append(np.zeros(n_seg)); b_eq[-1][0] = soc_init

    A_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * n_seg, n_blk * n_seg)).tocsr()
    b_vec = np.concatenate(b_eq)

    lb = np.zeros(n_blk * n_seg); ub = np.empty(n_blk * n_seg)
    ub[off_g:off_g + n_seg]    = np.inf
    ub[off_c:off_c + n_seg]    = E_MAX
    ub[off_d:off_d + n_seg]    = E_MAX
    ub[off_s:off_s + n_seg]    = pv * DT
    lb[off_e:off_e + n_seg]    = SOC_MIN
    ub[off_e:off_e + n_seg]    = SOC_MAX
    if use_emg:
        ub[off_e2:off_e2 + n_seg] = np.inf

    if verbose:
        print("多日LP", "天数", n_day, "变量", n_blk * n_seg, "等式约束", 2 * n_seg,
              "非零元", A_eq.nnz)
    res = linprog(c_obj, A_eq=A_eq, b_eq=b_vec, bounds=list(zip(lb, ub)), method="highs")

    if not res.success:
        return {"status": res.status, "message": res.message}

    x = res.x
    out = {
        "status": res.status, "message": res.message,
        "g": x[off_g:off_g + n_seg].reshape(n_day, N_T).copy(),
        "e": (x[off_e2:off_e2 + n_seg].reshape(n_day, N_T).copy()
              if use_emg else np.zeros((n_day, N_T))),
        "c": x[off_c:off_c + n_seg].reshape(n_day, N_T).copy(),
        "d": x[off_d:off_d + n_seg].reshape(n_day, N_T).copy(),
        "s": x[off_s:off_s + n_seg].reshape(n_day, N_T).copy(),
        "E": x[off_e:off_e + n_seg].reshape(n_day, N_T).copy(),
    }
    out["cost_plan"] = float((price.reshape(n_day, N_T) * out["g"]).sum())
    out["cost_emg"]  = float(BETA_EMG * (price.reshape(n_day, N_T) * out["e"]).sum())
    out["cost"]      = out["cost_plan"] + out["cost_emg"]
    if verbose:
        print("多日LP", "status", res.status, "总费用", "%.2f" % out["cost"],
              "计划", "%.2f" % out["cost_plan"], "紧急", "%.2f" % out["cost_emg"],
              "紧急购电", "%.6f" % out["e"].sum())
    return out


def soc_day_boundary(E_2d):
    """由 (D,144) 逐段储电量提取每日首末值。"""
    n_day = E_2d.shape[0]
    soc0 = np.empty(n_day); soc24 = E_2d[:, -1].copy()
    soc0[0] = SOC_INIT
    soc0[1:] = E_2d[:-1, -1]
    return soc0, soc24


N_T_ = N_T
_DECIDE_T = {0: 0, 1: 36, 2: 72, 3: 108}   # 6:00->36段, 12:00->72段, 18:00->108段
_STAGE_CACHE = {}


def _stage_structure(n=N_T):
    """阶段 LP 稀疏结构，缓存复用。"""
    if "A" in _STAGE_CACHE:
        return _STAGE_CACHE["I"], _STAGE_CACHE["A_eq"], _STAGE_CACHE["A_ub"]
    blk = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2", "GA", "U", "V"])}
    idx = np.arange(n)
    eq_r, eq_c, eq_v = [], [], []
    for name, sign in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        eq_r.append(idx); eq_c.append(blk[name] + idx); eq_v.append(sign * np.ones(n))
    soc_r = n + idx
    eq_r.append(soc_r); eq_c.append(blk["E"] + idx); eq_v.append(np.ones(n))
    eq_r.append(soc_r); eq_c.append(blk["C"] + idx); eq_v.append(-ETA_C * np.ones(n))
    eq_r.append(soc_r); eq_c.append(blk["D"] + idx); eq_v.append((1.0 / ETA_D) * np.ones(n))
    eq_r.append(soc_r[1:]); eq_c.append(blk["E"] + idx[1:] - 1); eq_v.append(-np.ones(n - 1))
    A_eq = coo_matrix((np.concatenate(eq_v), (np.concatenate(eq_r), np.concatenate(eq_c))),
                      shape=(2 * n, 9 * n)).tocsr()
    u_r, u_c, u_v = [], [], []
    u_r.append(idx); u_c.append(blk["GD"] + idx); u_v.append(np.ones(n))
    u_r.append(idx); u_c.append(blk["GA"] + idx); u_v.append(-np.ones(n))
    r_u = n + idx
    u_r.append(r_u); u_c.append(blk["GA"] + idx); u_v.append(-np.ones(n))
    u_r.append(r_u); u_c.append(blk["U"] + idx);  u_v.append(-np.ones(n))
    r_v = 2 * n + idx
    u_r.append(r_v); u_c.append(blk["GA"] + idx); u_v.append(np.ones(n))
    u_r.append(r_v); u_c.append(blk["V"] + idx);  u_v.append(-np.ones(n))
    A_ub = coo_matrix((np.concatenate(u_v), (np.concatenate(u_r), np.concatenate(u_c))),
                      shape=(3 * n, 9 * n)).tocsr()
    _STAGE_CACHE.update({"I": blk, "A_eq": A_eq, "A_ub": A_ub})
    return blk, A_eq, A_ub


_STAGE2_CACHE = {}


def _stage2_structure(t_decide):
    """按 t_decide 缓存的阶段 LP 稀疏结构。"""
    if t_decide in _STAGE2_CACHE:
        return _STAGE2_CACHE[t_decide]
    m = N_T - t_decide
    blk = {k: i * m for i, k in enumerate(["GA", "C", "D", "S", "E", "E2", "U", "V"])}
    idx = np.arange(m)
    rows, cols, vals = [], [], []
    for name, sign in [("GA", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        rows.append(idx); cols.append(blk[name] + idx); vals.append(sign * np.ones(m))
    soc_r = m + idx
    rows.append(soc_r); cols.append(blk["E"] + idx); vals.append(np.ones(m))
    rows.append(soc_r); cols.append(blk["C"] + idx); vals.append(-ETA_C * np.ones(m))
    rows.append(soc_r); cols.append(blk["D"] + idx); vals.append((1.0 / ETA_D) * np.ones(m))
    rows.append(soc_r[1:]); cols.append(blk["E"] + idx[1:] - 1); vals.append(-np.ones(m - 1))
    A_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * m, 8 * m)).tocsr()
    # U/V 两条约束分占不同行号
    ru, cu, vu = [], [], []
    ru.append(idx); cu.append(blk["GA"] + idx); vu.append(-np.ones(m))
    ru.append(idx); cu.append(blk["U"] + idx); vu.append(-np.ones(m))
    rv = m + idx
    ru.append(rv); cu.append(blk["GA"] + idx); vu.append(np.ones(m))
    ru.append(rv); cu.append(blk["V"] + idx); vu.append(-np.ones(m))
    A_ub = coo_matrix((np.concatenate(vu), (np.concatenate(ru), np.concatenate(cu))),
                      shape=(2 * m, 8 * m)).tocsr()
    _STAGE2_CACHE[t_decide] = (blk, A_eq, A_ub)
    return blk, A_eq, A_ub


def solve_stage_lp(price, load_forecast, pv_forecast, g_plan, t_decide, soc_at):
    """阶段 LP：只重优化 [t_decide,144)。"""
    blk, A_eq, A_ub = _stage2_structure(t_decide)
    m = N_T - t_decide
    t = np.arange(t_decide, N_T)
    idx = np.arange(m)
    c_obj = np.zeros(8 * m)
    c_obj[blk["GA"] + idx] = BETA_DEF * price[t]
    c_obj[blk["V"] + idx] = (BETA_EXC - BETA_DEF) * price[t]
    c_obj[blk["E2"] + idx] = BETA_EMG * price[t]
    b_eq = np.concatenate([load_forecast[t] * DT - pv_forecast[t] * DT,
                           np.r_[soc_at, np.zeros(m - 1)]])
    b_ub = np.concatenate([-g_plan[t], g_plan[t]])
    lb = np.zeros(8 * m); ub = np.empty(8 * m)
    ub[blk["GA"]:blk["GA"] + m] = np.inf
    ub[blk["C"]:blk["C"] + m] = E_MAX
    ub[blk["D"]:blk["D"] + m] = E_MAX
    ub[blk["S"]:blk["S"] + m] = pv_forecast[t] * DT
    lb[blk["E"]:blk["E"] + m] = SOC_MIN; ub[blk["E"]:blk["E"] + m] = SOC_MAX
    ub[blk["E2"]:blk["E2"] + m] = np.inf
    ub[blk["U"]:blk["U"] + m] = np.inf
    ub[blk["V"]:blk["V"] + m] = np.inf
    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                  bounds=list(zip(lb, ub)), method="highs")
    return res, blk, m


def forecast_bias_table(fc10_all, pv_2d):
    """扩张窗口偏差表 BIAS[d,k,t]；pv 校正用。"""
    n_day, n_ver, n_seg = fc10_all.shape
    bias = np.zeros((n_day, n_ver, n_seg))
    for k in range(n_ver):
        cum_bias = np.zeros(n_seg); count = 0
        for day in range(n_day):
            if count > 0:
                bias[day, k] = cum_bias / count
            fc_row = fc10_all[day, k]
            if not np.isnan(fc_row).any():
                cum_bias = cum_bias + (fc_row - pv_2d[day]); count += 1
    return bias


def _settle_structure(n=N_T):
    """实际结算 LP 稀疏结构，缓存复用。"""
    if "AS" in _STAGE_CACHE:
        return _STAGE_CACHE["IS"], _STAGE_CACHE["AS_eq"], _STAGE_CACHE["AS_ub"]
    blk = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(n)
    eq_r, eq_c, eq_v = [], [], []
    for name, sign in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        eq_r.append(idx); eq_c.append(blk[name] + idx); eq_v.append(sign * np.ones(n))
    soc_r = n + idx
    eq_r.append(soc_r); eq_c.append(blk["E"] + idx); eq_v.append(np.ones(n))
    eq_r.append(soc_r); eq_c.append(blk["C"] + idx); eq_v.append(-ETA_C * np.ones(n))
    eq_r.append(soc_r); eq_c.append(blk["D"] + idx); eq_v.append((1.0 / ETA_D) * np.ones(n))
    eq_r.append(soc_r[1:]); eq_c.append(blk["E"] + idx[1:] - 1); eq_v.append(-np.ones(n - 1))
    A_eq = coo_matrix((np.concatenate(eq_v), (np.concatenate(eq_r), np.concatenate(eq_c))),
                      shape=(2 * n, 6 * n)).tocsr()
    # 结算购电量不超过已承诺量
    A_ub = coo_matrix((np.concatenate([np.ones(n), -np.ones(n)]),
                       (np.concatenate([idx, idx]),
                        np.concatenate([blk["GD"] + idx,
                                        blk["GA"] if False else np.zeros(n, dtype=int)]))),
                      shape=(n, 6 * n)).tocsr()
    one_rows = list(range(n)); gd_cols = list(blk["GD"] + idx); one_vals = [1.0] * n
    A_ub = coo_matrix((np.array(one_vals), (np.array(one_rows), np.array(gd_cols))),
                      shape=(n, 6 * n)).tocsr()
    _STAGE_CACHE.update({"IS": blk, "AS_eq": A_eq, "AS_ub": A_ub})
    return blk, A_eq, A_ub


def settle_actual_lp(price, load_kw, pv_actual, committed, soc_start):
    """按当日实际光伏与负载做最终结算。"""
    n = N_T
    blk, A_eq, A_ub = _settle_structure(n)
    idx = np.arange(n)
    c_obj = np.zeros(6 * n)
    c_obj[blk["E2"] + idx] = BETA_EMG * price
    # 吞吐量正则化：取充放电最少的解
    c_obj[blk["C"] + idx] = THRU_PEN
    c_obj[blk["D"] + idx] = THRU_PEN
    b_eq = np.concatenate([load_kw * DT - pv_actual * DT,
                           np.r_[soc_start, np.zeros(n - 1)]])
    b_ub = committed.copy()
    lb = np.zeros(6 * n); ub = np.empty(6 * n)
    ub[blk["GD"]:blk["GD"] + n] = committed
    ub[blk["C"]:blk["C"] + n] = E_MAX
    ub[blk["D"]:blk["D"] + n] = E_MAX
    ub[blk["S"]:blk["S"] + n] = pv_actual * DT
    lb[blk["E"]:blk["E"] + n] = SOC_MIN; ub[blk["E"]:blk["E"] + n] = SOC_MAX
    ub[blk["E2"]:blk["E2"] + n] = np.inf
    return linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                   bounds=list(zip(lb, ub)), method="highs")


def run_rolling(price_2d, load_2d, pv_2d, fc10, active_stages=(1, 2, 3),
                soc_init=SOC_INIT, verbose=False):
    """全年逐日滚动求解（问3 主模型）。"""
    n_day = price_2d.shape[0]
    g_plan = np.zeros((n_day, N_T)); g_adj = np.zeros((n_day, N_T))
    charge = np.zeros((n_day, N_T)); discharge = np.zeros((n_day, N_T))
    curtail = np.zeros((n_day, N_T)); soc_all = np.zeros((n_day, N_T))
    emg = np.zeros((n_day, N_T))
    soc = soc_init
    for day in range(n_day):
        plan = solve_day_lp(price_2d[day], load_2d[day], fc10[day, 0], soc_start=soc, soc_end=None)
        if plan["status"] != 0:
            raise RuntimeError("第 %d 天计划阶段失败" % day)
        g_plan[day] = plan["g"]
        committed = g_plan[day].copy()
        for k in active_stages:
            t_dec = _DECIDE_T[k]
            # t_dec 时刻的真实储电量作为阶段初值
            settle0 = settle_actual_lp(price_2d[day], load_2d[day], pv_2d[day], committed, soc)
            blk = _STAGE_CACHE["IS"]
            soc_at = float(settle0.x[blk["E"] + t_dec - 1]) if settle0.success else soc
            stage, stage_blk, m_stage = solve_stage_lp(price_2d[day], load_2d[day], fc10[day, k],
                                                       g_plan[day], t_dec, soc_at)
            if not stage.success:
                raise RuntimeError("第 %d 天阶段%d 失败" % (day, k))
            committed[t_dec:] = stage.x[stage_blk["GA"]:stage_blk["GA"] + m_stage]
        g_adj[day] = committed
        settle = settle_actual_lp(price_2d[day], load_2d[day], pv_2d[day], committed, soc)
        if not settle.success:
            raise RuntimeError("第 %d 天实际结算失败" % day)
        y = settle.x
        blk = _STAGE_CACHE["IS"]
        charge[day] = y[blk["C"]:blk["C"] + N_T]; discharge[day] = y[blk["D"]:blk["D"] + N_T]
        curtail[day] = y[blk["S"]:blk["S"] + N_T]; soc_all[day] = y[blk["E"]:blk["E"] + N_T]
        emg[day] = y[blk["E2"]:blk["E2"] + N_T]
        soc = soc_all[day, -1]
        if verbose and (day + 1) % 90 == 0:
            print("已完成 %d/%d 天" % (day + 1, n_day))
    return {"g_plan": g_plan, "g_adj": g_adj, "c": charge, "d": discharge,
            "s": curtail, "E": soc_all, "e": emg}


def cost_breakdown(price_2d, gp, ga, ee):
    """按 Reading A 口径核算四项费用。"""
    short_kwh = np.maximum(gp - ga, 0.0)
    over_kwh = np.maximum(ga - gp, 0.0)
    return {
        "base": float((price_2d * np.minimum(gp, ga)).sum()),
        "def":  float((price_2d * BETA_DEF * short_kwh).sum()),
        "exc":  float((price_2d * BETA_EXC * over_kwh).sum()),
        "emg":  float(BETA_EMG * (price_2d * ee).sum()),
        "u_kwh": float(short_kwh.sum()), "v_kwh": float(over_kwh.sum()), "e_kwh": float(ee.sum()),
    }
