"""问2 稳健性扫描：SAA 情景数、惩罚倍数 beta、情景缩放系数 rho"""
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
REP0 = 31
data = L.load_all(verbose=False)
price_mean = data["typical"]["price"]
load_2d = data["actual"]["load_kw"]
pv_2d = data["actual"]["pv_kw"]
dates = data["actual"]["dates"]
DTYPE = ST.day_type(dates)


def settle(g_plan, load_kw, pv_kw, soc_start):
    """固定计划量，按当日实际负载/光伏结算，只最小化紧急购电。"""
    n = N
    blk = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(n)
    cost_vec = np.zeros(6 * n)
    cost_vec[blk["E2"] + idx] = L.BETA_EMG * price_mean
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


def run_q2(S=20, beta=5.0, rho=1.0):
    L.BETA_EMG = beta
    rng = np.random.default_rng(42)
    soc = L.SOC_INIT
    c_plan = c_emg = 0.0
    tot_plan = tot_emg = 0.0
    ndays_emg = 0
    for d in range(L.N_DAY):
        if d < 5:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0.0)
        else:
            load_fc = ST.point_forecast(load_2d, DTYPE, d)
            pv_fc = ST.point_forecast(pv_2d, DTYPE, d)
            load_pool = ST.residual_pool(load_2d, DTYPE, d, eps=0.0)
            pv_pool = ST.residual_pool(pv_2d, DTYPE, d, eps=50.0)
            load_scen = ST.sample_scenarios(load_pool, S, rng) * load_fc
            pv_scen = np.clip(ST.sample_scenarios(pv_pool, S, rng) * pv_fc, 0.0, None)
            if rho != 1.0:
                # rho 缩放情景偏差
                load_scen = load_fc[None, :] + rho * (load_scen - load_fc[None, :])
                pv_scen = np.clip(pv_fc[None, :] + rho * (pv_scen - pv_fc[None, :]), 0.0, None)
            # 显式传入 beta_emg
            two_stage = ST.solve_two_stage(price_mean, load_fc, pv_fc, load_scen, pv_scen,
                                           soc, None, beta_emg=beta)
            if two_stage is not None:
                plan = two_stage["g_plan"]
            else:
                plan = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0.0)
        res = settle(plan, load_2d[d], pv_2d[d], soc)
        soc = res.x[4 * N + N - 1]
        if d >= REP0:
            c_plan += float(price_mean @ plan)
            emg_kwh = res.x[5 * N:6 * N]
            c_emg += float(beta * (price_mean @ emg_kwh))
            tot_plan += float(plan.sum())
            tot_emg += float(emg_kwh.sum())
            ndays_emg += int(emg_kwh.sum() > 1e-6)
    return dict(S=S, beta=beta, rho=rho, cost=c_plan + c_emg, plan=c_plan, emg=c_emg,
                kwh_plan=tot_plan, kwh_emg=tot_emg, days_emg=ndays_emg)


CONFIGS = [
    ("基准（S=20, β=5, ρ=1.0）", dict(S=20, beta=5.0, rho=1.0)),
    ("SAA：S=10", dict(S=10, beta=5.0, rho=1.0)),
    ("SAA：S=50", dict(S=50, beta=5.0, rho=1.0)),
    ("惩罚倍数：β=3", dict(S=20, beta=3.0, rho=1.0)),
    ("惩罚倍数：β=8", dict(S=20, beta=8.0, rho=1.0)),
    ("保守度：ρ=0.5（情景收窄）", dict(S=20, beta=5.0, rho=0.5)),
    ("保守度：ρ=1.5（情景放宽）", dict(S=20, beta=5.0, rho=1.5)),
]

print("问2 稳健性扫描（均值电价，报告期 334 天，严格样本外）")
print("%-26s %14s %14s %13s %11s %6s" %
      ("配置", "报告期费用(元)", "其中紧急费", "紧急购电(kWh)", "计划购电(kWh)", "触发天数"))
scan = []
for label, cfg in CONFIGS:
    t0 = time.time()
    res = run_q2(**cfg)
    scan.append((label, res))
    print("  %-26s %14.2f %14.2f %13.0f %11.0f %6d   (%4.0fs)"
          % (label, res["cost"], res["emg"], res["kwh_emg"], res["kwh_plan"],
             res["days_emg"], time.time() - t0))
    sys.stdout.flush()

base = scan[0][1]["cost"]
print("相对基准的偏差：")
for label, res in scan[1:]:
    print("  %-26s %+12.2f 元 (%+.3f%%)"
          % (label, res["cost"] - base, (res["cost"] - base) / base * 100))

np.savez("results/问2稳健性扫描_结果.npz",
         labels=np.array([label for label, _ in scan]),
         cost=np.array([res["cost"] for _, res in scan]),
         plan=np.array([res["plan"] for _, res in scan]),
         emg=np.array([res["emg"] for _, res in scan]),
         kwh_plan=np.array([res["kwh_plan"] for _, res in scan]),
         kwh_emg=np.array([res["kwh_emg"] for _, res in scan]),
         days_emg=np.array([res["days_emg"] for _, res in scan]))
print("已保存 results/问2稳健性扫描_结果.npz")
