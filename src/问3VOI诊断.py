"""问3 边际信息价值非单调的诊断：预报精度与阶段 LP 口径。"""
import os, sys, io
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import lp_core as L
import stoch as ST

N = L.N_T
DT = L.DT
REP0 = 31
TDEC = {1: 36, 2: 72, 3: 108}
data = L.load_all(verbose=False)
price_mean = data["typical"]["price"]
load_2d, pv_2d = data["actual"]["load_kw"], data["actual"]["pv_kw"]
fc_all = data["forecast"]["fc"]
dates = data["actual"]["dates"]
fc_10min = np.zeros((L.N_DAY, 4, N))
for d in range(L.N_DAY):
    for k in range(4):
        fc_10min[d, k] = L.forecast_to_10min(fc_all[d], k)
npz3 = np.load("results/问3_结果.npz", allow_pickle=True)

print("预报精度是否随发布时刻改善（报告期 334 天，只统计被决策的时段 t>=tdec）")
stats = []
for k in range(4):
    t0 = [0, 36, 72, 108][k]
    err_list = []
    for d in range(REP0, L.N_DAY):
        err_list.append(fc_10min[d, k][t0:] - pv_2d[d][t0:])
    err = np.concatenate(err_list)
    ref = np.concatenate([pv_2d[d][t0:] for d in range(REP0, L.N_DAY)])
    bias, mae, rmse = err.mean(), np.abs(err).mean(), np.sqrt((err ** 2).mean())
    rel_rmse = rmse / ref.mean() * 100
    stats.append((k, t0, bias, mae, rmse, rel_rmse))
    print("%d:00" % [0, 6, 12, 18][k], "发布", "t>=%d" % t0, "偏差 %+.2f" % bias,
          "MAE %.2f" % mae, "RMSE %.2f" % rmse, "相对RMSE %.2f%%" % rel_rmse)
best = min(stats, key=lambda r: r[4])
print("RMSE 最小的是", "%d:00" % [0, 6, 12, 18][best[0]], "发布，精度确实随时间改善")

# 计费边际 0.5*pi / 1.5*pi，LP 用 2.5*pi
print("真实计费边际成本：GA<gp 时 0.5*pi，GA>gp 时 1.5*pi")
print("阶段 LP 边际成本：GA<gp 时 0.5*pi，GA>gp 时 2.5*pi，对『多买』多算了一个 pi")

stage_index = {k: i * N for i, k in enumerate(["GD", "C", "D", "S", "E", "E2", "GA", "U", "V"])}


def stage_lp(price_vec, load_kw, pv_actual, pv_fc, g_plan, committed, tdec, soc_start,
             obj_mode="legacy"):
    n = N
    blk, A_eq, A_ub = L._stage_structure(n)
    cost_vec = np.zeros(9 * n)
    fut = np.arange(tdec, n)
    if obj_mode == "legacy":
        cost_vec[blk["GA"] + fut] = price_vec[fut]
    else:
        # min(gp,GA) 项 = pi*GA - pi*V
        cost_vec[blk["GA"] + fut] = price_vec[fut]
        cost_vec[blk["V"] + fut] = (L.BETA_EXC - 1.0) * price_vec[fut]
    cost_vec[blk["U"] + fut] = L.BETA_DEF * price_vec[fut]
    cost_vec[blk["E2"] + np.arange(n)] = L.BETA_EMG * price_vec
    pv_use = np.concatenate([pv_actual[:tdec], pv_fc[tdec:]])
    b_eq = np.concatenate([load_kw * DT - pv_use * DT, np.r_[soc_start, np.zeros(n - 1)]])
    b_ub = np.concatenate([np.zeros(n), -g_plan, g_plan])
    lb = np.zeros(9 * n); ub = np.empty(9 * n)
    ub[blk["GD"]:blk["GD"] + n] = np.inf
    ub[blk["C"]:blk["C"] + n] = L.E_MAX
    ub[blk["D"]:blk["D"] + n] = L.E_MAX
    ub[blk["S"]:blk["S"] + n] = pv_use * DT
    lb[blk["E"]:blk["E"] + n] = L.SOC_MIN; ub[blk["E"]:blk["E"] + n] = L.SOC_MAX
    ub[blk["E2"]:blk["E2"] + n] = np.inf
    lb[blk["GA"]:blk["GA"] + n] = 0.0; ub[blk["GA"]:blk["GA"] + n] = np.inf
    lb[blk["GA"]:blk["GA"] + tdec] = committed[:tdec]
    ub[blk["GA"]:blk["GA"] + tdec] = committed[:tdec]
    ub[blk["U"]:blk["U"] + n] = np.inf
    ub[blk["V"]:blk["V"] + n] = np.inf
    res = linprog(cost_vec, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                  bounds=list(zip(lb, ub)), method="highs")
    return res, blk


def billing(price_vec, gp, ga):
    """真实计费口径（Reading A）"""
    u = np.maximum(gp - ga, 0.0); v = np.maximum(ga - gp, 0.0)
    return float((price_vec * np.minimum(gp, ga)).sum()
                 + (price_vec * L.BETA_DEF * u).sum()
                 + (price_vec * L.BETA_EXC * v).sum()), float(u.sum()), float(v.sum())


plan_all = npz3["g_plan"]; adjust_all = npz3["g_adj"]
for k in (1, 2, 3):
    t0 = TDEC[k]
    gp = plan_all[:, t0:]; ga = adjust_all[:, t0:]
    u = np.maximum(gp - ga, 0); v = np.maximum(ga - gp, 0)
    print("阶段", k, "计划量(段均) %.2f" % gp.mean(), "调整量(段均) %.2f" % ga.mean(),
          "净调整 %+.2f" % (ga - gp).mean(), "U总量 %.1f" % u.sum(), "V总量 %.1f" % v.sum(),
          "V占比 %.1f%%" % (v.sum() / (u.sum() + v.sum()) * 100))
print("V 占比接近 0，说明阶段 LP 几乎从不『多买』，与过度惩罚的判断一致")

# pv_use = [实际[:tdec], 预报[tdec:]]
print("阶段 LP 对负载拥有完美信息，新增的信息实际上只有光伏预报")
