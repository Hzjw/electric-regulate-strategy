"""问4 后悔值对紧急购电惩罚倍数 beta 的敏感性"""
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
SL = slice(REP0, L.N_DAY)

data = L.load_all(verbose=False)
price_mean = data["typical"]["price"]
price_rt = data["price_rt"]["price"]
load_2d = data["actual"]["load_kw"]
pv_2d = data["actual"]["pv_kw"]
dates = data["actual"]["dates"]
DTYPE = ST.day_type(dates)


def settle(price, g_plan, load_kw, pv_kw, soc_start):
    """固定计划量 g_plan，按实际负载/光伏结算；目标仅含紧急购电（计划费用为沉没成本）。"""
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


def plan_mean_price():
    """问2 内核：均值电价下的两阶段随机规划，取 365 天计划量。"""
    rng = np.random.default_rng(42)
    soc = L.SOC_INIT
    plans = np.zeros((L.N_DAY, N))
    for d in range(L.N_DAY):
        if d < 5:
            plans[d] = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0.0)
        else:
            load_fc = ST.point_forecast(load_2d, DTYPE, d)
            pv_fc = ST.point_forecast(pv_2d, DTYPE, d)
            load_pool = ST.residual_pool(load_2d, DTYPE, d, eps=0.0)
            pv_pool = ST.residual_pool(pv_2d, DTYPE, d, eps=50.0)
            load_scen = ST.sample_scenarios(load_pool, N_SCEN, rng) * load_fc
            pv_scen = np.clip(ST.sample_scenarios(pv_pool, N_SCEN, rng) * pv_fc, 0.0, None)
            # 显式传入 beta_emg
            two_stage = ST.solve_two_stage(price_mean, load_fc, pv_fc, load_scen, pv_scen,
                                           soc, None, beta_emg=beta)
            if two_stage is not None:
                plans[d] = two_stage["g_plan"]
            else:
                plans[d] = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0.0)
        res = settle(price_mean, plans[d], load_2d[d], pv_2d[d], soc)
        assert res.success, "方案A 第 %d 天结算失败" % d
        soc = res.x[4 * N + N - 1]
    return plans


def plan_rt():
    """4-2 内核：实时电价下的两阶段随机规划，取 365 天计划量与紧急购电量。"""
    rng = np.random.default_rng(42)
    soc = L.SOC_INIT
    plans = np.zeros((L.N_DAY, N)); emg = np.zeros((L.N_DAY, N))
    for d in range(L.N_DAY):
        pk = price_rt[d]
        load_fc = ST.load_forecast(load_2d, DTYPE, d) if d >= 5 else load_2d[d]
        pv_fc = ST.pv_forecast_persistence(pv_2d, DTYPE, d) if d >= 5 else pv_2d[d]
        if d >= 5:
            load_pool = ST.load_residual_pool(load_2d, DTYPE, d)
            pv_pool = ST.pv_residual_pool_persist(pv_2d, DTYPE, d)
            load_scen = ST.load_scenarios(load_pool, load_fc, N_SCEN, rng)
            pv_scen = np.clip(ST.sample_scenarios(pv_pool, N_SCEN, rng) * pv_fc, 0, None)
            two_stage = ST.solve_two_stage(pk, load_fc, pv_fc, load_scen, pv_scen,
                                           soc, None, beta_emg=beta)
            if two_stage is not None:
                plan = two_stage["g_plan"]
            else:
                plan = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0)
        else:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0)
        plans[d] = plan
        res = settle(pk, plan, load_2d[d], pv_2d[d], soc)
        assert res.success, "方案B 第 %d 天结算失败" % d
        emg[d] = res.x[5 * N:6 * N]
        soc = res.x[4 * N + N - 1]
    return plans, emg


def cost_A(plan_a):
    """方案A 按实时电价完整结算（计划购电费 + 紧急购电费）。"""
    soc = L.SOC_INIT
    buy_cost = emg_cost = 0.0
    for d in range(REP0, L.N_DAY):
        res = settle(price_rt[d], plan_a[d], load_2d[d], pv_2d[d], soc)
        soc = res.x[4 * N + N - 1]
        buy_cost += float(price_rt[d] @ plan_a[d])
        emg_cost += float(L.BETA_EMG * (price_rt[d] @ res.x[5 * N:6 * N]))
    return buy_cost, emg_cost, buy_cost + emg_cost


BETAS = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                            ["1.5", "2", "3", "5", "8", "12"])]

print("问4 后悔值结论对紧急购电惩罚倍数 beta 的敏感性")
print("方案A 用均值电价做计划、按实时电价完整结算；方案B 直接用实时电价优化")
print("%5s %16s %12s %18s %12s %8s %9s"
      % ("beta", "方案A(均值电价计划)", "方案A紧急费", "方案B(实时电价优化)", "后悔值 A-B",
         "占B比", "报童分位"))
scan = []
for beta in BETAS:
    t0 = time.time()
    L.BETA_EMG = beta
    plan_a = plan_mean_price()
    buy_a, emg_a, cost_a = cost_A(plan_a)
    plan_b, emg_b = plan_rt()
    cost_b = (float((price_rt[SL] * plan_b[SL]).sum())
              + beta * float((price_rt[SL] * emg_b[SL]).sum()))
    regret = cost_a - cost_b
    scan.append((beta, cost_a, buy_a, emg_a, cost_b, regret, regret / cost_b * 100))
    # 相对后悔率分母取方案B 费用
    print("%5.1f  %16.2f  %12.2f  %18.2f  %12.2f  %7.2f%%  %8.2f   (%.0f s)"
          % (beta, cost_a, emg_a, cost_b, regret, regret / cost_b * 100, (beta - 1) / beta,
             time.time() - t0))

neg = [r for r in scan if r[5] <= 0]
pos = [r for r in scan if r[5] > 0]
if neg and pos:
    lo = max(neg, key=lambda r: r[0]); hi = min(pos, key=lambda r: r[0])
    frac = -lo[5] / (hi[5] - lo[5])
    print("符号翻转临界 beta* ≈ %.2f" % (lo[0] + frac * (hi[0] - lo[0])),
          "（插值位置 %.0f%%）" % (frac * 100))
elif pos and not neg:
    print("所有测试 beta 下后悔值均 > 0，结论在低惩罚区不成立")
elif neg and not pos:
    print("所有测试 beta 下后悔值均 <= 0，结论对 beta 稳健（至少覆盖 [%.1f, %.1f]）"
          % (min(r[0] for r in scan), max(r[0] for r in scan)))
