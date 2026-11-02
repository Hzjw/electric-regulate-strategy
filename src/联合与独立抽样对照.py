"""联合 vs 独立抽样对照：同一历史日成对抽取残差"""
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
price_rt = data["price_rt"]["price"]
load_2d = data["actual"]["load_kw"]
pv_2d = data["actual"]["pv_kw"]
dates = data["actual"]["dates"]
DTYPE = ST.day_type(dates)


def settle_day(price, g_plan, load_kw, pv_kw, soc_start):
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


def load_scen_from_rows(pool_rows, load_fc):
    """与 stoch.load_scenarios 相同的还原公式，但作用于指定的残差行。"""
    total_pt = load_fc.sum() * L.DT
    shape_pt = load_fc / max(total_pt / 24.0, 1e-9)
    return (total_pt * pool_rows[:, 0:1] / 24.0) * (shape_pt[None, :] * pool_rows[:, 1:])


def run42(paired):
    """paired=False：现行独立抽样；paired=True：按同一历史日成对抽样。"""
    rng = np.random.default_rng(42)
    soc = L.SOC_INIT
    cost = 0.0
    tot_plan = tot_emg = 0.0
    corrs = []
    for d in range(L.N_DAY):
        pk = price_rt[d]
        load_fc = ST.load_forecast(load_2d, DTYPE, d) if d >= 5 else load_2d[d]
        pv_fc = ST.pv_forecast_persistence(pv_2d, DTYPE, d) if d >= 5 else pv_2d[d]
        if d >= 5:
            load_pool = ST.load_residual_pool(load_2d, DTYPE, d)
            pv_pool = ST.pv_residual_pool_persist(pv_2d, DTYPE, d)
            if paired:
                pairs = min(len(load_pool) - 1, len(pv_pool))
                if pairs <= 0:
                    load_scen = np.tile(load_fc, (N_SCEN, 1))
                    pv_scen = np.tile(pv_fc, (N_SCEN, 1))
                else:
                    pair_idx = rng.integers(0, pairs, size=N_SCEN)
                    load_scen = load_scen_from_rows(load_pool[pair_idx + 1], load_fc)
                    pv_scen = np.clip(pv_pool[pair_idx] * pv_fc, 0.0, None)
            else:
                load_scen = ST.load_scenarios(load_pool, load_fc, N_SCEN, rng)
                pv_scen = np.clip(ST.sample_scenarios(pv_pool, N_SCEN, rng) * pv_fc, 0.0, None)
            # 情景内负载—光伏相关系数
            if pv_scen.std() > 1e-6 and load_scen.std() > 1e-6:
                corrs.append(float(np.corrcoef(load_scen.reshape(-1), pv_scen.reshape(-1))[0, 1]))
            two_stage = ST.solve_two_stage(pk, load_fc, pv_fc, load_scen, pv_scen, soc, None)
            if two_stage is not None:
                plan = two_stage["g_plan"]
            else:
                plan = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0)
        else:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0)
        res = settle_day(pk, plan, load_2d[d], pv_2d[d], soc)
        soc = res.x[4 * N + N - 1]
        if d >= REP0:
            cost += float(pk @ plan) + float(L.BETA_EMG * (pk @ res.x[5 * N:6 * N]))
            tot_plan += float(plan.sum())
            tot_emg += float(res.x[5 * N:6 * N].sum())
    return dict(cost=cost, plan=tot_plan, emg=tot_emg, corr=float(np.mean(corrs)))


print("联合 vs 独立抽样对照（问题 4-2，实时电价，报告期 334 天）")
t0 = time.time()
ind_res = run42(False)
print("独立抽样（现行）完成 %.0f s" % (time.time() - t0))
t1 = time.time()
pair_res = run42(True)
print("成对抽样（改进）完成 %.0f s" % (time.time() - t1))
print("%-22s %16s %14s %14s %14s"
      % ("口径", "报告期费用(元)", "计划购电(kWh)", "紧急购电(kWh)", "情景内相关系数"))
for name, res in [("独立抽样（现行）", ind_res), ("成对抽样（保留联合结构）", pair_res)]:
    print("  %-22s %16.2f %14.0f %14.0f %14.4f"
          % (name, res["cost"], res["plan"], res["emg"], res["corr"]))
print("费用之差 %+.2f 元（%+.3f%%）"
      % (pair_res["cost"] - ind_res["cost"],
         (pair_res["cost"] - ind_res["cost"]) / ind_res["cost"] * 100),
      "计划购电之差 %+.0f kWh" % (pair_res["plan"] - ind_res["plan"]),
      "紧急购电之差 %+.0f kWh" % (pair_res["emg"] - ind_res["emg"]))
print("若独立抽样的情景内相关系数接近 0、成对抽样显著为正，则现行做法确实丢了源荷联合结构")
print("费用差异若小于 0.5%，说明该假设对结论无实质影响，属可声明的方法学细节")

np.savez("results/联合与独立抽样_结果.npz",
         ind_cost=ind_res["cost"], par_cost=pair_res["cost"],
         ind_plan=ind_res["plan"], par_plan=pair_res["plan"],
         ind_emg=ind_res["emg"], par_emg=pair_res["emg"],
         ind_corr=ind_res["corr"], par_corr=pair_res["corr"])
print("已保存 results/联合与独立抽样_结果.npz")
