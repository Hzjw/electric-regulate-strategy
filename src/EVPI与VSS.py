"""EVPI：报告期 334 天，SP 费用减 WS 费用"""
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
N_SCEN = 20
REP0 = 31
data = L.load_all(verbose=False)
price_mean = data["typical"]["price"]
price_rt = data["price_rt"]["price"]
load_2d = data["actual"]["load_kw"]
pv_2d = data["actual"]["pv_kw"]
DTYPE = ST.day_type(data["actual"]["dates"])


def stage_cost(price, g_plan, load_kw, pv_kw, soc_start):
    """给定计划量 g_plan，求第二阶段最优补救费用（含紧急购电）。"""
    n = N
    blk = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(n)
    cost_vec = np.zeros(6 * n)
    cost_vec[blk["E2"] + idx] = L.BETA_EMG * price
    cost_vec[blk["C"] + idx] = L.THRU_PEN
    cost_vec[blk["D"] + idx] = L.THRU_PEN
    rows, cols, vals, rhs = [], [], [], []
    for name, co in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        rows.append(idx); cols.append(blk[name] + idx); vals.append(co * np.ones(n))
    rhs.append(load_kw * L.DT - pv_kw * L.DT)
    soc_row = n + idx
    rows.append(soc_row); cols.append(blk["E"] + idx); vals.append(np.ones(n))
    rows.append(soc_row); cols.append(blk["C"] + idx); vals.append(-L.ETA_C * np.ones(n))
    rows.append(soc_row); cols.append(blk["D"] + idx); vals.append((1.0 / L.ETA_D) * np.ones(n))
    rows.append(soc_row[1:]); cols.append(blk["E"] + idx[1:] - 1); vals.append(-np.ones(n - 1))
    soc_rhs = np.zeros(n); soc_rhs[0] = soc_start
    rhs.append(soc_rhs)
    A_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * n, 6 * n)).tocsr()
    lb = np.zeros(6 * n); ub = np.empty(6 * n)
    ub[blk["GD"]:blk["GD"] + n] = g_plan
    ub[blk["C"]:blk["C"] + n] = L.E_MAX
    ub[blk["D"]:blk["D"] + n] = L.E_MAX
    ub[blk["S"]:blk["S"] + n] = pv_kw * L.DT
    lb[blk["E"]:blk["E"] + n] = L.SOC_MIN
    ub[blk["E"]:blk["E"] + n] = L.SOC_MAX
    ub[blk["E2"]:blk["E2"] + n] = np.inf
    return linprog(cost_vec, A_eq=A_eq, b_eq=np.concatenate(rhs),
                   bounds=list(zip(lb, ub)), method="highs")


def build_scenarios(price):
    """按 04/06 的情景生成顺序重跑一遍，逐日保存 (soc, lf, pf, ls, ps, g_plan)。"""
    rng = np.random.default_rng(42)
    soc = L.SOC_INIT
    rec = []
    for d in range(L.N_DAY):
        if d < 5:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0.0)
            load_scen = np.tile(load_2d[d], (N_SCEN, 1))
            pv_scen = np.tile(pv_2d[d], (N_SCEN, 1))
            load_fc = pv_fc = None
        else:
            if price is price_mean:
                load_fc = ST.point_forecast(load_2d, DTYPE, d)
                pv_fc = ST.point_forecast(pv_2d, DTYPE, d)
                load_pool = ST.residual_pool(load_2d, DTYPE, d, eps=0.0)
                pv_pool = ST.residual_pool(pv_2d, DTYPE, d, eps=50.0)
            else:
                load_fc = ST.load_forecast(load_2d, DTYPE, d)
                pv_fc = ST.pv_forecast_persistence(pv_2d, DTYPE, d)
                load_pool = ST.load_residual_pool(load_2d, DTYPE, d)
                pv_pool = ST.pv_residual_pool_persist(pv_2d, DTYPE, d)
            if price is price_mean:
                # 问2 残差池为相对曲线，乘点预测
                load_scen = ST.sample_scenarios(load_pool, N_SCEN, rng) * load_fc
                pv_scen = np.clip(ST.sample_scenarios(pv_pool, N_SCEN, rng) * pv_fc, 0.0, None)
            else:
                # 4-2 负载残差池为分解式，走还原公式
                load_scen = ST.load_scenarios(load_pool, load_fc, N_SCEN, rng)
                pv_scen = np.clip(ST.sample_scenarios(pv_pool, N_SCEN, rng) * pv_fc, 0.0, None)
            two_stage = ST.solve_two_stage(price[d] if price.ndim > 1 else price,
                                           load_fc, pv_fc, load_scen, pv_scen, soc, None)
            if two_stage is not None:
                plan = two_stage["g_plan"]
            else:
                plan = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0.0)
        res = stage_cost(price[d] if price.ndim > 1 else price, plan, load_2d[d], pv_2d[d], soc)
        soc = res.x[4 * N + N - 1]
        rec.append(dict(soc0=None, lf=load_fc, pf=pv_fc, ls=load_scen, ps=pv_scen,
                        plan=plan, soc_end=soc))
    return rec


def analyse(price, name):
    price_of = lambda d: (price[d] if price.ndim > 1 else price)
    rec = build_scenarios(price)
    ws_sum = sp_sum = eev_sum = 0.0
    for d in range(REP0, L.N_DAY):
        day = rec[d]
        p = price_of(d)
        soc0 = rec[d - 1]["soc_end"]
        # SP：计划量下的期望第二阶段费用
        sp = float(p @ day["plan"])
        for s in range(N_SCEN):
            emg_res = stage_cost(p, day["plan"], day["ls"][s], day["ps"][s], soc0)
            sp += float(L.BETA_EMG * (p @ emg_res.x[5 * N:6 * N])) / N_SCEN
        # WS：情景已知时的最优费用
        ws = 0.0
        for s in range(N_SCEN):
            ws_res = L.solve_day_lp(p, day["ls"][s], day["ps"][s], soc_start=soc0, soc_end=None)
            ws += ws_res["cost"] / N_SCEN
        ws_sum += ws
        sp_sum += sp
    print(name, "报告期 334 天")
    print("  WS 完美信息期望费用 %.2f 元" % ws_sum)
    print("  SP 两阶段随机规划 %.2f 元" % sp_sum)
    print("  EVPI = SP - WS = %.2f 元（占 SP 的 %.3f%%）"
          % (sp_sum - ws_sum, (sp_sum - ws_sum) / sp_sum * 100))
    return dict(WS=ws_sum, SP=sp_sum, EVPI=sp_sum - ws_sum)


print("EVPI（报告期 334 天，同一情景分布）")
t0 = time.time()
result_mean = analyse(price_mean, "问2 · 均值电价")
print("  用时 %.0f s" % (time.time() - t0))
t1 = time.time()
result_rt = analyse(price_rt, "4-2 · 实时电价")
print("  用时 %.0f s" % (time.time() - t1))
np.savez("results/EVPI与VSS_结果.npz",
         mn_WS=result_mean["WS"], mn_SP=result_mean["SP"], mn_EVPI=result_mean["EVPI"],
         rt_WS=result_rt["WS"], rt_SP=result_rt["SP"], rt_EVPI=result_rt["EVPI"])
print("说明：VSS 已在 基准线与后悔值.py 中按统一完整结算口径给出，本脚本只补 EVPI")
print("已保存 results/EVPI与VSS_结果.npz")
