"""问3 情景化滚动验证：阶段 LP 改用情景化随机版本"""
import os, sys, io, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import lp_core as L
import stoch as ST
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

N = L.N_T
N_SCEN_DAY = 20     # 日前两阶段 SP 情景数
S_STAGE = 8         # 阶段 LP 情景数
DECIDE_T = {1: 36, 2: 72, 3: 108}
REP0 = 31
SL = slice(REP0, L.N_DAY)

D = L.load_all(verbose=False)
price = D["typical"]["price"]
load_2d = D["actual"]["load_kw"]
pv_2d = D["actual"]["pv_kw"]
FC = D["forecast"]["fc"]
dates = D["actual"]["dates"]
DTYPE = ST.day_type(dates)
FC10 = np.zeros((L.N_DAY, 4, N))
for d in range(L.N_DAY):
    for k in range(4):
        FC10[d, k] = L.forecast_to_10min(FC[d], k)
BIAS = L.forecast_bias_table(FC10, pv_2d)
PVD = {}


def build_stage_scen(tdec, S):
    """情景化阶段 LP 的系数矩阵：GA/U/V 三个共享变量，每个情景各占一套储能变量。"""
    m = N - tdec
    nv = 3 * m + 6 * S * m
    ga_ofs, u_ofs, v_ofs = 0, m, 2 * m
    seg = np.arange(m)
    eq_row, eq_col, eq_val = [], [], []
    for s in range(S):
        blk = 3 * m + s * 6 * m
        ofs = {name: blk + i * m for i, name in enumerate(["GD", "C", "D", "S", "E", "E2"])}
        row = s * 2 * m + seg
        for name, coef in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
            eq_row.append(row)
            eq_col.append(ofs[name] + seg)
            eq_val.append(coef * np.ones(m))
        row_soc = s * 2 * m + m + seg
        eq_row.append(row_soc)
        eq_col.append(ofs["E"] + seg)
        eq_val.append(np.ones(m))
        eq_row.append(row_soc)
        eq_col.append(ofs["C"] + seg)
        eq_val.append(-L.ETA_C * np.ones(m))
        eq_row.append(row_soc)
        eq_col.append(ofs["D"] + seg)
        eq_val.append((1.0 / L.ETA_D) * np.ones(m))
        eq_row.append(row_soc[1:])
        eq_col.append(ofs["E"] + seg[1:] - 1)
        eq_val.append(-np.ones(m - 1))
    A_eq = coo_matrix((np.concatenate(eq_val), (np.concatenate(eq_row), np.concatenate(eq_col))),
                      shape=(2 * S * m, nv)).tocsr()
    ub_row, ub_col, ub_val = [], [], []
    ub_row.append(seg)
    ub_col.append(ga_ofs + seg)
    ub_val.append(-np.ones(m))
    ub_row.append(seg)
    ub_col.append(u_ofs + seg)
    ub_val.append(-np.ones(m))
    row_v = m + seg
    ub_row.append(row_v)
    ub_col.append(ga_ofs + seg)
    ub_val.append(np.ones(m))
    ub_row.append(row_v)
    ub_col.append(v_ofs + seg)
    ub_val.append(-np.ones(m))
    for s in range(S):
        blk = 3 * m + s * 6 * m
        row_cap = 2 * m + s * m + seg
        ub_row.append(row_cap)
        ub_col.append(blk + seg)
        ub_val.append(np.ones(m))
        ub_row.append(row_cap)
        ub_col.append(ga_ofs + seg)
        ub_val.append(-np.ones(m))
    A_ub = coo_matrix((np.concatenate(ub_val), (np.concatenate(ub_row), np.concatenate(ub_col))),
                      shape=(2 * m + S * m, nv)).tocsr()
    return ga_ofs, u_ofs, v_ofs, A_eq, A_ub


_STC = {}


def solve_stage_scen(pvk_mu, load_scen, pv_scen, g_plan, tdec, soc_at, S):
    """情景化阶段 LP：最小化剩余时段的期望费用。"""
    m = N - tdec
    t = np.arange(tdec, N)
    key = tdec
    if key not in _STC:
        _STC[key] = build_stage_scen(tdec, S)
    ga_ofs, u_ofs, v_ofs, A_eq, A_ub = _STC[key]
    nv = 3 * m + 6 * S * m
    cost = np.zeros(nv)
    cost[ga_ofs + idx0(m)] = L.BETA_DEF * price[t]
    cost[v_ofs + idx0(m)] = (L.BETA_EXC - L.BETA_DEF) * price[t]
    b_eq = []
    for s in range(S):
        b_eq.append(load_scen[s][t] * L.DT - pv_scen[s][t] * L.DT)
        b_eq.append(np.r_[soc_at, np.zeros(m - 1)])
    b_eq = np.concatenate(b_eq)
    b_ub = np.concatenate([-g_plan[t], g_plan[t], np.zeros(S * m)])
    lb = np.zeros(nv)
    ub = np.empty(nv)
    ub[ga_ofs:ga_ofs + m] = np.inf
    ub[u_ofs:u_ofs + m] = np.inf
    ub[v_ofs:v_ofs + m] = np.inf
    for s in range(S):
        blk = 3 * m + s * 6 * m
        ub[blk:blk + m] = np.inf
        ub[blk + m:blk + 2 * m] = L.E_MAX
        ub[blk + 2 * m:blk + 3 * m] = L.E_MAX
        ub[blk + 3 * m:blk + 4 * m] = pv_scen[s][t] * L.DT
        lb[blk + 4 * m:blk + 5 * m] = L.SOC_MIN
        ub[blk + 4 * m:blk + 5 * m] = L.SOC_MAX
        ub[blk + 5 * m:blk + 6 * m] = np.inf
    # E2 期望目标系数按情景数摊分
    for s in range(S):
        blk = 3 * m + s * 6 * m
        cost[blk + 5 * m + np.arange(m)] = (L.BETA_EMG / S) * price[t]
    return linprog(cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                   bounds=list(zip(lb, ub)), method="highs"), ga_ofs, m


def idx0(m):
    return np.arange(m)


def settle_day(g_plan, load_kw, pv_kw, soc_start):
    n = N
    ofs = {name: i * n for i, name in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    seg = np.arange(n)
    cost = np.zeros(6 * n)
    cost[ofs["E2"] + seg] = L.BETA_EMG * price
    cost[ofs["C"] + seg] = L.THRU_PEN
    cost[ofs["D"] + seg] = L.THRU_PEN
    eq_row, eq_col, eq_val, eq_rhs = [], [], [], []
    for name, coef in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        eq_row.append(seg)
        eq_col.append(ofs[name] + seg)
        eq_val.append(coef * np.ones(n))
    eq_rhs.append(load_kw * L.DT - pv_kw * L.DT)
    soc_row = n + seg
    eq_row.append(soc_row)
    eq_col.append(ofs["E"] + seg)
    eq_val.append(np.ones(n))
    eq_row.append(soc_row)
    eq_col.append(ofs["C"] + seg)
    eq_val.append(-L.ETA_C * np.ones(n))
    eq_row.append(soc_row)
    eq_col.append(ofs["D"] + seg)
    eq_val.append((1.0 / L.ETA_D) * np.ones(n))
    eq_row.append(soc_row[1:])
    eq_col.append(ofs["E"] + seg[1:] - 1)
    eq_val.append(-np.ones(n - 1))
    eq_rhs.append(np.r_[soc_start, np.zeros(n - 1)])
    A_eq = coo_matrix((np.concatenate(eq_val), (np.concatenate(eq_row), np.concatenate(eq_col))),
                      shape=(2 * n, 6 * n)).tocsr()
    lb = np.zeros(6 * n)
    ub = np.empty(6 * n)
    ub[ofs["GD"]:ofs["GD"] + n] = g_plan
    ub[ofs["C"]:ofs["C"] + n] = L.E_MAX
    ub[ofs["D"]:ofs["D"] + n] = L.E_MAX
    ub[ofs["S"]:ofs["S"] + n] = pv_kw * L.DT
    lb[ofs["E"]:ofs["E"] + n] = L.SOC_MIN
    ub[ofs["E"]:ofs["E"] + n] = L.SOC_MAX
    ub[ofs["E2"]:ofs["E2"] + n] = np.inf
    return linprog(cost, A_eq=A_eq, b_eq=np.concatenate(eq_rhs),
                   bounds=list(zip(lb, ub)), method="highs")


def cost_of(gp, ga, ee_):
    """与 问题3求解.py 同一计费口径：基准 + 违约 + 超额 + 紧急。

    传进来的 gp/ga/ee 已经是报告期 334 天的数组，这里再切一次 sl 就会整体错位。
    """
    under = np.maximum(gp - ga, 0)
    over = np.maximum(ga - gp, 0)
    return {"base": float((price * np.minimum(gp, ga)).sum()),
            "def": float((price * L.BETA_DEF * under).sum()),
            "exc": float((price * L.BETA_EXC * over).sum()),
            "emg": float(L.BETA_EMG * (price * ee_).sum()),
            "u": float(under.sum()), "v": float(over.sum()), "e": float(ee_.sum()),
            "tot": (float((price * np.minimum(gp, ga)).sum())
                    + float((price * L.BETA_DEF * under).sum())
                    + float((price * L.BETA_EXC * over).sum())
                    + float(L.BETA_EMG * (price * ee_).sum()))}


def run(mode, vint, active, label):
    """mode='det' 走确定性阶段 LP（交付口径），'scen' 走情景化阶段随机 LP。"""
    rng = np.random.default_rng(42)
    soc = L.SOC_INIT
    plan_days, adj_days, emg_days = [], [], []
    for d in range(L.N_DAY):
        load_fc = ST.load_forecast(load_2d, DTYPE, d) if d >= 5 else load_2d[d]
        # 日前计划用 0:00 版光伏预报
        pv_fc = FC10[d, 0] if d >= 5 else pv_2d[d]
        if d >= 5:
            load_pool = ST.load_residual_pool(load_2d, DTYPE, d)
            pv_pool = ST.fc_residual_pool(FC10[:, 0], pv_2d, d, 0, eps=50.0)
            load_scen0 = ST.load_scenarios(load_pool, load_fc, N_SCEN_DAY, rng)
            pv_scen0 = np.clip(ST.sample_scenarios(pv_pool, N_SCEN_DAY, rng) * pv_fc, 0, None)
            res0 = ST.solve_two_stage(price, load_fc, pv_fc, load_scen0, pv_scen0, soc, None)
            plan = (res0["g_plan"] if res0 is not None
                    else np.maximum(load_fc * L.DT - pv_fc * L.DT, 0))
        else:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0)
        committed = plan.copy()
        for k in active:
            t_dec = DECIDE_T[k]
            soc_path = ST.soc_trajectory(price, committed, load_2d[d], pv_2d[d], soc)
            soc_at = float(soc_path[0][t_dec - 1]) if soc_path is not None else soc
            ver = vint.get(k, k)
            pv_stage = np.clip(FC10[d, ver] - BIAS[d, ver], 0.0, None) if d >= 5 else pv_2d[d]
            if mode == "det":
                rk, ofs, m = L.solve_stage_lp(price, load_fc, pv_stage, plan, t_dec, soc_at)
                if rk.success:
                    committed[t_dec:] = rk.x[ofs["GA"]:ofs["GA"] + m]
            else:
                if d >= 5:
                    if (d, ver) not in PVD:
                        PVD[(d, ver)] = ST.fc_residual_pool(FC10[:, ver], pv_2d, d, ver, eps=50.0)
                    pv_pool_stage = PVD[(d, ver)]
                    pv_scen = np.clip(np.clip(FC10[d, ver] - BIAS[d, ver], 0.0, None)[None, :]
                                      * np.clip(ST.sample_scenarios(pv_pool_stage, S_STAGE, rng),
                                                0.0, None), 0, None)
                    load_scen = ST.load_scenarios(ST.load_residual_pool(load_2d, DTYPE, d),
                                                  load_fc, S_STAGE, rng)
                else:
                    pv_scen = np.tile(pv_2d[d], (S_STAGE, 1))
                    load_scen = np.tile(load_2d[d], (S_STAGE, 1))
                rk, ofs, m = solve_stage_scen(pv_stage, load_scen, pv_scen, plan, t_dec,
                                              soc_at, S_STAGE)
                if rk.success:
                    committed[t_dec:] = rk.x[ofs:ofs + m]
        res = settle_day(committed, load_2d[d], pv_2d[d], soc)
        soc = res.x[4 * N + N - 1]
        if d >= REP0:
            plan_days.append(plan.copy())
            adj_days.append(committed.copy())
            emg_days.append(res.x[5 * N:6 * N].copy())
    cost = cost_of(np.array(plan_days), np.array(adj_days), np.array(emg_days))
    print(label, "合计 %.2f 元" % cost["tot"])
    sys.stdout.flush()
    return cost["tot"]


print("问3 情景化滚动验证（均值电价，报告期 334 天）")
t0 = time.time()
CA = run("det", {}, (), "A 不调整（两种模式相同）")
CBd = run("det", {1: 0, 2: 0, 3: 0}, (1, 2, 3), "B 有改签·无新信息（确定性，交付口径）")
CCd = run("det", {1: 1, 2: 2, 3: 3}, (1, 2, 3), "C 有改签·各用当期预报（确定性，交付口径）")
CBs = run("scen", {1: 0, 2: 0, 3: 0}, (1, 2, 3), "B' 有改签·无新信息（情景化）")
CCs = run("scen", {1: 1, 2: 2, 3: 3}, (1, 2, 3), "C' 有改签·各用当期预报（情景化）")
print("总用时 %.0f s" % (time.time() - t0))
print("决策机会价值 C(A)-C(B)：确定性 %.2f 元，情景化 %.2f 元" % (CA - CBd, CA - CBs))
print("信息更新价值 C(B)-C(C)：确定性 %.2f 元，情景化 %.2f 元" % (CBd - CCd, CBs - CCs))
print("情景化后决策机会价值若仍为负，说明该结论不依赖阶段模型的形式。")
np.savez("results/问3情景化滚动_结果.npz", CA=CA, CBd=CBd, CCd=CCd, CBs=CBs, CCs=CCs, S_stage=S_STAGE)
print("已保存 results/问3情景化滚动_结果.npz")
