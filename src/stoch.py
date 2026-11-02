"""随机优化模块：点预测、情景生成与两阶段随机规划。"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

DT = 1.0 / 6.0
N_T = 144


def day_type(dates):
    """日型分类：周五、周六为 0，其余为 1。"""
    dow = pd.DatetimeIndex(dates).dayofweek        # 周一=0 ... 周日=6
    return np.where((dow == 4) | (dow == 5), 0, 1)


def point_forecast(hist_2d, dtypes, d, k=8, decay=0.85):
    """相似日加权点预测：最近 k 天，按距离指数衰减。"""
    cands = [j for j in range(d) if dtypes[j] == dtypes[d]]
    if not cands:
        cands = list(range(d))
    cands = cands[-k:]
    weights = np.array([decay ** (len(cands) - 1 - i) for i in range(len(cands))])
    weights = weights / weights.sum()
    return np.tensordot(weights, hist_2d[cands], axes=(0, 0))


def residual_pool(actual_2d, dtypes, d, k=40, eps=0.0):
    """相对残差池：同型日 实际/预测 的逐段比值，eps 平滑。"""
    pool = []
    if d == 0:                      # 无历史时返回全 1 曲线
        return np.ones((1, actual_2d.shape[1]))
    cands = [j for j in range(d) if dtypes[j] == dtypes[d]]
    if len(cands) < 5:
        cands = list(range(d))
    for j in cands[-k:]:
        fc = point_forecast(actual_2d, dtypes, j, k=8)
        # 分母加极小正数
        ratio = (actual_2d[j] + eps) / np.maximum(fc + eps, 1e-9)
        pool.append(np.clip(ratio, 0.2, 2.5))
    return np.array(pool)


def sample_scenarios(pool, n_scen, rng):
    """整日成块抽取情景。"""
    idx = rng.integers(0, len(pool), size=n_scen)
    return pool[idx]


def solve_two_stage(price, load_fc, pv_fc, load_scen, pv_scen,
                    soc_start, soc_term, beta_emg=5.0, n_t=N_T, c_cyc=0.0):
    """两阶段随机规划：阶段一定计划购电量，阶段二情景内调度。"""
    n_scen = load_scen.shape[0]
    n = n_t
    # 变量分块：g_plan(n) + 每情景 6 块 (g_draw,c,d,s,E,e)
    n_var = n + n_scen * 6 * n
    cost = np.zeros(n_var)
    cost[0:n] = price                                    # 第一阶段：计划购电费用
    for w in range(n_scen):
        base = n + w * 6 * n
        cost[base + 5 * n: base + 6 * n] = (beta_emg / n_scen) * price
        if c_cyc:
            # 循环成本按第二阶段放电量期望计提
            cost[base + 2 * n: base + 3 * n] = c_cyc / n_scen

    rows, cols, vals, b_eq = [], [], [], []
    rows_u, cols_u, vals_u, b_ub = [], [], [], []
    idx = np.arange(n)
    for w in range(n_scen):
        base = n + w * 6 * n
        off_gd, off_c, off_d, off_s, off_e, off_emg = (base + 0 * n, base + 1 * n,
                                                       base + 2 * n, base + 3 * n,
                                                       base + 4 * n, base + 5 * n)
        row = w * 2 * n + idx
        for off, co in [(off_gd, 1.0), (off_emg, 1.0), (off_d, 1.0), (off_c, -1.0), (off_s, -1.0)]:
            rows.append(row); cols.append(off + idx); vals.append(co * np.ones(n))
        b_eq.append(load_scen[w] * DT - pv_scen[w] * DT)
        row_soc = w * 2 * n + n + idx
        rows.append(row_soc); cols.append(off_e + idx); vals.append(np.ones(n))
        rows.append(row_soc); cols.append(off_c + idx)
        vals.append(-__import__("lp_core").ETA_C * np.ones(n))
        rows.append(row_soc); cols.append(off_d + idx)
        vals.append((1.0 / __import__("lp_core").ETA_D) * np.ones(n))
        rows.append(row_soc[1:]); cols.append(off_e + idx[1:] - 1); vals.append(-np.ones(n - 1))
        rhs = np.zeros(n); rhs[0] = soc_start
        b_eq.append(rhs)
        rows_u.append(w * n + idx); cols_u.append(off_gd + idx); vals_u.append(np.ones(n))
        rows_u.append(w * n + idx); cols_u.append(idx); vals_u.append(-np.ones(n))
        b_ub.append(np.zeros(n))

    A_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * n_scen * n, n_var)).tocsr()
    A_ub = coo_matrix((np.concatenate(vals_u), (np.concatenate(rows_u), np.concatenate(cols_u))),
                      shape=(n_scen * n, n_var)).tocsr()

    import lp_core as L
    lb = np.zeros(n_var); ub = np.empty(n_var)
    ub[0:n] = np.inf
    for w in range(n_scen):
        base = n + w * 6 * n
        ub[base + 0 * n: base + 1 * n] = np.inf
        ub[base + 1 * n: base + 2 * n] = L.E_MAX
        ub[base + 2 * n: base + 3 * n] = L.E_MAX
        ub[base + 3 * n: base + 4 * n] = pv_scen[w] * DT
        lb[base + 4 * n: base + 5 * n] = L.SOC_MIN
        ub[base + 4 * n: base + 5 * n] = L.SOC_MAX
        ub[base + 5 * n: base + 6 * n] = np.inf
    res = linprog(cost, A_eq=A_eq, b_eq=np.concatenate(b_eq),
                  A_ub=A_ub, b_ub=np.concatenate(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        return None
    # d_mean：情景平均放电量 (kWh)
    sol = res.x
    d_sum = np.zeros(n)
    for w in range(n_scen):
        base = n + w * 6 * n
        d_sum += sol[base + 2 * n: base + 3 * n]
    return {"g_plan": res.x[0:n].copy(), "obj": float(res.fun), "x": res.x, "S": n_scen, "n": n,
            "d_mean": d_sum / n_scen}


def extract_scenario(res, w):
    """取第 w 个情景的第二阶段结果。"""
    n, n_scen = res["n"], res["S"]
    base = n + w * 6 * n
    x = res["x"]
    return {"g_draw": x[base + 0 * n: base + 1 * n],
            "c": x[base + 1 * n: base + 2 * n],
            "d": x[base + 2 * n: base + 3 * n],
            "s": x[base + 3 * n: base + 4 * n],
            "E": x[base + 4 * n: base + 5 * n],
            "e": x[base + 5 * n: base + 6 * n]}


def advance_window(price, g_adj, load_act, pv_act, soc_start, i0, i1):
    """已发生窗口 [i0,i1) 的确定性调度，返回末端储能。"""
    import lp_core as L
    m = i1 - i0
    if m <= 0:
        return soc_start, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    idx = np.arange(m)
    # g_draw 松弛量：计划量是计费量而非取用量
    blocks = {k: i * m for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    cost = np.zeros(6 * m)
    cost[blocks["E2"] + idx] = L.BETA_EMG * price[i0:i1]
    cost[blocks["C"] + idx] = L.THRU_PEN
    cost[blocks["D"] + idx] = L.THRU_PEN
    eq_r, eq_c, eq_v, b_eq = [], [], [], []
    for name, co in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        eq_r.append(idx); eq_c.append(blocks[name] + idx); eq_v.append(co * np.ones(m))
    b_eq.append(load_act[i0:i1] * L.DT - pv_act[i0:i1] * L.DT)
    row_soc = m + idx
    eq_r.append(row_soc); eq_c.append(blocks["E"] + idx); eq_v.append(np.ones(m))
    eq_r.append(row_soc); eq_c.append(blocks["C"] + idx); eq_v.append(-L.ETA_C * np.ones(m))
    eq_r.append(row_soc); eq_c.append(blocks["D"] + idx); eq_v.append((1.0 / L.ETA_D) * np.ones(m))
    eq_r.append(row_soc[1:]); eq_c.append(blocks["E"] + idx[1:] - 1); eq_v.append(-np.ones(m - 1))
    rhs = np.zeros(m); rhs[0] = soc_start
    b_eq.append(rhs)
    A_eq = coo_matrix((np.concatenate(eq_v), (np.concatenate(eq_r), np.concatenate(eq_c))),
                      shape=(2 * m, 6 * m)).tocsr()
    lb = np.zeros(6 * m); ub = np.empty(6 * m)
    ub[blocks["GD"]:blocks["GD"] + m] = g_adj[i0:i1]      # 抽取量不超过计划量
    ub[blocks["C"]:blocks["C"] + m] = L.E_MAX
    ub[blocks["D"]:blocks["D"] + m] = L.E_MAX
    ub[blocks["S"]:blocks["S"] + m] = pv_act[i0:i1] * L.DT
    lb[blocks["E"]:blocks["E"] + m] = L.SOC_MIN
    ub[blocks["E"]:blocks["E"] + m] = L.SOC_MAX
    ub[blocks["E2"]:blocks["E2"] + m] = np.inf
    res = linprog(cost, A_eq=A_eq, b_eq=np.concatenate(b_eq),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        return soc_start, np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)
    x = res.x
    return (x[blocks["E"] + m - 1],
            x[blocks["C"]:blocks["C"] + m], x[blocks["D"]:blocks["D"] + m],
            x[blocks["S"]:blocks["S"] + m], x[blocks["E2"]:blocks["E2"] + m])


def solve_stage_scen(price, g_plan, g_adj_prev, tdec, soc_start,
                     load_fc, pv_fc, load_scen, pv_scen, soc_term=None,
                     beta_def=None, beta_exc=None, beta_emg=None):
    """问题三情景化调整阶段 LP：未来时段用历史误差生成情景。"""
    import lp_core as L
    if beta_def is None: beta_def = L.BETA_DEF
    if beta_exc is None: beta_exc = L.BETA_EXC
    if beta_emg is None: beta_emg = L.BETA_EMG
    m = N_T - tdec
    if m <= 0:
        return None
    n_scen = load_scen.shape[0]
    # 变量分块：g_adj | u | v | 每情景 6 块
    n_adj, n_uv = m, 2 * m
    base_scen = n_adj + n_uv
    n_var = base_scen + n_scen * 6 * m
    idx = np.arange(m)
    price_f = price[tdec:]
    g_plan_f = g_plan[tdec:]

    cost = np.zeros(n_var)
    # 目标 pi*g_adj+(beta_exc-1)*pi*v；GA/U/V 顺序固定
    cost[0:m] = price_f
    cost[n_adj: n_adj + m] = beta_def * price_f
    cost[n_adj + m: n_adj + 2 * m] = (beta_exc - 1.0) * price_f

    rows, cols, vals, b_eq = [], [], [], []
    rows_u, cols_u, vals_u, b_ub = [], [], [], []
    # u、v 两条约束分占不同行号
    rows_u.append(idx); cols_u.append(0 + idx); vals_u.append(-np.ones(m))
    rows_u.append(idx); cols_u.append(n_adj + idx); vals_u.append(-np.ones(m))
    b_ub.append(-g_plan_f)
    rows_u.append(m + idx); cols_u.append(0 + idx); vals_u.append(np.ones(m))
    rows_u.append(m + idx); cols_u.append(n_adj + 2 * m + idx); vals_u.append(-np.ones(m))
    b_ub.append(g_plan_f)

    for w in range(n_scen):
        base = base_scen + w * 6 * m
        off_gd, off_c, off_d, off_s, off_e, off_emg = (base, base + m, base + 2 * m,
                                                       base + 3 * m, base + 4 * m, base + 5 * m)
        cost[off_emg + idx] = (beta_emg / n_scen) * price_f
        row = w * 2 * m + idx
        for off, co in [(off_gd, 1.0), (off_emg, 1.0), (off_d, 1.0), (off_c, -1.0), (off_s, -1.0)]:
            rows.append(row); cols.append(off + idx); vals.append(co * np.ones(m))
        b_eq.append(load_scen[w][tdec:] * L.DT - pv_scen[w][tdec:] * L.DT)
        row_soc = w * 2 * m + m + idx
        rows.append(row_soc); cols.append(off_e + idx); vals.append(np.ones(m))
        rows.append(row_soc); cols.append(off_c + idx); vals.append(-L.ETA_C * np.ones(m))
        rows.append(row_soc); cols.append(off_d + idx); vals.append((1.0 / L.ETA_D) * np.ones(m))
        rows.append(row_soc[1:]); cols.append(off_e + idx[1:] - 1); vals.append(-np.ones(m - 1))
        rhs = np.zeros(m); rhs[0] = soc_start
        b_eq.append(rhs)
        rows_u.append(2 * m + w * m + idx); cols_u.append(off_gd + idx)
        vals_u.append(np.ones(m))
        rows_u.append(2 * m + w * m + idx); cols_u.append(0 + idx); vals_u.append(-np.ones(m))
        b_ub.append(np.zeros(m))

    A_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * n_scen * m, n_var)).tocsr()
    A_ub = coo_matrix((np.concatenate(vals_u), (np.concatenate(rows_u), np.concatenate(cols_u))),
                      shape=((2 + n_scen) * m, n_var)).tocsr()

    lb = np.zeros(n_var); ub = np.empty(n_var)
    ub[0:m] = np.inf                       # g_adj 未来可自由调整
    ub[n_adj: n_adj + 2 * m] = np.inf      # u, v
    for w in range(n_scen):
        base = base_scen + w * 6 * m
        ub[base:base + m] = np.inf
        ub[base + m:base + 2 * m] = L.E_MAX
        ub[base + 2 * m:base + 3 * m] = L.E_MAX
        ub[base + 3 * m:base + 4 * m] = pv_scen[w][tdec:] * L.DT
        lb[base + 4 * m:base + 5 * m] = L.SOC_MIN
        ub[base + 4 * m:base + 5 * m] = L.SOC_MAX
        ub[base + 5 * m:base + 6 * m] = np.inf
        if soc_term is not None:
            lb[base + 4 * m + m - 1] = soc_term
    res = linprog(cost, A_eq=A_eq, b_eq=np.concatenate(b_eq), A_ub=A_ub,
                  b_ub=np.concatenate(b_ub), bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        return None
    return {"g_adj": res.x[0:m].copy(), "tdec": tdec, "S": n_scen, "m": m,
            "base": base_scen, "obj": float(res.fun), "x": res.x}


def fc_residual_pool(fc10_all, actual_2d, d, k, eps=50.0, kk=40):
    """按发布时刻 k 分层的预报误差残差池，样本外。"""
    pool = []
    for j in range(d):
        f = fc10_all[j, k]
        if np.nanmax(f) < 1.0:            # 预报异常日跳过
            continue
        ratio = (actual_2d[j] + eps) / (f + eps)
        pool.append(np.clip(ratio, 0.2, 2.5))
    if not pool:
        pool = [np.ones(N_T)]
    return np.array(pool[-kk:])


def _same_type_days(dtypes, d, k):
    """目标日之前同型的最近 k 天。"""
    cands = [j for j in range(d) if dtypes[j] == dtypes[d]]
    if len(cands) < 5:
        cands = list(range(d))
    return cands[-k:]


def load_forecast(load_2d, dtypes, d, k=12, decay=0.88):
    """分解式负载点预测：日总量与日内形状两层外推。"""
    cands = _same_type_days(dtypes, d, k)
    weights = np.array([decay ** (len(cands) - 1 - i) for i in range(len(cands))])
    weights = weights / weights.sum()
    tot = load_2d[cands].sum(axis=1) * DT                  # 各历史日的日总量
    tot_hat = float(weights @ tot)
    shapes = load_2d[cands] / np.maximum(tot[:, None] / 24.0, 1e-9)
    shape_hat = weights @ shapes
    return (tot_hat / 24.0) * shape_hat


def load_residual_pool(load_2d, dtypes, d, k=60):
    """分解式残差池：第 0 列 rho，其余列 sigma。"""
    types = np.asarray(dtypes)
    key = np.zeros(load_2d.shape[1] + 1)
    key[0] = 1.0
    pool = [key]
    cands = [j for j in range(d) if types[j] == types[d]]
    if len(cands) < 5:
        cands = list(range(d))
    for j in cands[-k:]:
        if j < 3:
            continue
        f = load_forecast(load_2d, dtypes, j, k=12)
        tot_f = f.sum() * DT
        tot_a = load_2d[j].sum() * DT
        rho = tot_a / max(tot_f, 1e-9)
        sig = load_2d[j] / np.maximum(f, 1e-9)
        row = np.empty(load_2d.shape[1] + 1)
        row[0] = np.clip(rho, 0.5, 1.8)
        row[1:] = np.clip(sig, 0.5, 1.8)
        pool.append(row)
    return np.array(pool)


def load_scenarios(pool, load_pt, n_scen, rng):
    """负荷情景 (T_hat*rho/24)*(s_hat*sigma)"""
    idx = rng.integers(0, len(pool), size=n_scen)
    rows = pool[idx]
    tot_pt = load_pt.sum() * DT                     # 点预测的日总量
    shape_pt = load_pt / np.maximum(tot_pt / 24.0, 1e-9)
    out = (tot_pt * rows[:, 0:1] / 24.0) * (shape_pt[None, :] * rows[:, 1:])
    return out


def clearsky_ref(pv_2d, d, k=45):
    """晴空参考：同季窗口光伏出力的上包络。"""
    lo, hi = max(0, d - k), d
    if hi - lo < 10:
        lo, hi = 0, max(d, 1)
    win = pv_2d[lo:hi]
    ref = np.percentile(win, 92, axis=0)
    # 轻度平滑
    ker = np.ones(5) / 5.0
    ref = np.convolve(np.r_[ref[0], ref[0], ref, ref[-1], ref[-1]], ker, mode="same")[2:-2]
    return np.maximum(ref, 1.0)


def pv_forecast_persistence(pv_2d, dtypes, d, k=12, decay=0.88):
    """光伏点预测：晴空指数持续性法。"""
    ref_d = clearsky_ref(pv_2d, d)
    cands = _same_type_days(dtypes, d, k)
    clearness = []
    for j in cands:
        ref_j = clearsky_ref(pv_2d, j)
        clearness.append(pv_2d[j] / np.maximum(ref_j, 1.0))
    weights = np.array([decay ** (len(cands) - 1 - i) for i in range(len(cands))])
    weights = weights / weights.sum()
    clearness_hat = weights @ np.array(clearness)
    return np.clip(clearness_hat, 0.0, 1.05) * ref_d


def pv_residual_pool_persist(pv_2d, dtypes, d, k=60, eps=50.0):
    """光伏持续性预报残差池。"""
    pool = []
    cands = [j for j in range(d) if dtypes[j] == dtypes[d]]
    if len(cands) < 5:
        cands = list(range(d))
    for j in cands[-k:]:
        if j < 3:
            continue
        f = pv_forecast_persistence(pv_2d, dtypes, j, k=12)
        pool.append(np.clip((pv_2d[j] + eps) / (f + eps), 0.3, 2.2))
    if not pool:
        pool = [np.ones(pv_2d.shape[1])]
    return np.array(pool)


def soc_trajectory(price, g_fixed, load_kw, pv_kw, soc_start):
    """给定已承诺购电量，按当日实际负载/光伏做全天优化。"""
    # 物理常量统一从 lp_core 取
    import lp_core as L
    n = N_T
    blocks = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(n)
    cost = np.zeros(6 * n)
    cost[blocks["E2"] + idx] = L.BETA_EMG * price
    cost[blocks["C"] + idx] = L.THRU_PEN
    cost[blocks["D"] + idx] = L.THRU_PEN
    eq_r, eq_c, eq_v, b_eq = [], [], [], []
    for name, co in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        eq_r.append(idx); eq_c.append(blocks[name] + idx); eq_v.append(co * np.ones(n))
    b_eq.append(load_kw * DT - pv_kw * DT)
    row_soc = n + idx
    eq_r.append(row_soc); eq_c.append(blocks["E"] + idx); eq_v.append(np.ones(n))
    eq_r.append(row_soc); eq_c.append(blocks["C"] + idx); eq_v.append(-L.ETA_C * np.ones(n))
    eq_r.append(row_soc); eq_c.append(blocks["D"] + idx); eq_v.append((1.0 / L.ETA_D) * np.ones(n))
    eq_r.append(row_soc[1:]); eq_c.append(blocks["E"] + idx[1:] - 1); eq_v.append(-np.ones(n - 1))
    rhs = np.zeros(n); rhs[0] = soc_start
    b_eq.append(rhs)
    A_eq = coo_matrix((np.concatenate(eq_v), (np.concatenate(eq_r), np.concatenate(eq_c))),
                      shape=(2 * n, 6 * n)).tocsr()
    lb = np.zeros(6 * n); ub = np.empty(6 * n)
    ub[blocks["GD"]:blocks["GD"] + n] = g_fixed
    ub[blocks["C"]:blocks["C"] + n] = L.E_MAX
    ub[blocks["D"]:blocks["D"] + n] = L.E_MAX
    ub[blocks["S"]:blocks["S"] + n] = pv_kw * DT
    lb[blocks["E"]:blocks["E"] + n] = L.SOC_MIN
    ub[blocks["E"]:blocks["E"] + n] = L.SOC_MAX
    ub[blocks["E2"]:blocks["E2"] + n] = np.inf
    res = linprog(cost, A_eq=A_eq, b_eq=np.concatenate(b_eq),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        return None
    x = res.x
    return (x[blocks["E"]:blocks["E"] + n], x[blocks["C"]:blocks["C"] + n],
            x[blocks["D"]:blocks["D"] + n],
            x[blocks["S"]:blocks["S"] + n], x[blocks["E2"]:blocks["E2"] + n])
