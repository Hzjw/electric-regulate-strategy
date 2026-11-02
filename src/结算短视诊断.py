"""结算层短视诊断：逐日 vs 7 天滚动联合结算。"""
import os, sys, io, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import lp_core as L

# 逐日独立结算，储能不可跨日传递
N = L.N_T
DT = L.DT
REP0 = 31
data = L.load_all(verbose=False)
price_mean = data["typical"]["price"]
price_rt = data["price_rt"]["price"]
load_2d, pv_2d = data["actual"]["load_kw"], data["actual"]["pv_kw"]
npz2 = np.load("results/问2_结果.npz", allow_pickle=True)
npz4 = np.load("results/问4_结果.npz", allow_pickle=True)


def settle_joint(pflat, gplan, load, pvv, soc_start):
    """展平后 m 段联合结算，末端储电量自由（无价值）。"""
    m = len(gplan)
    blk = {k: i * m for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(m)
    cost_vec = np.zeros(6 * m)
    cost_vec[blk["E2"] + idx] = L.BETA_EMG * pflat
    cost_vec[blk["C"] + idx] = L.THRU_PEN
    cost_vec[blk["D"] + idx] = L.THRU_PEN
    rows, cols, vals, rhs = [], [], [], []
    for name, co in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        rows.append(idx); cols.append(blk[name] + idx); vals.append(co * np.ones(m))
    rhs.append(load * DT - pvv * DT)
    soc_row = m + idx
    rows.append(soc_row); cols.append(blk["E"] + idx); vals.append(np.ones(m))
    rows.append(soc_row); cols.append(blk["C"] + idx); vals.append(-L.ETA_C * np.ones(m))
    rows.append(soc_row); cols.append(blk["D"] + idx); vals.append((1.0 / L.ETA_D) * np.ones(m))
    rows.append(soc_row[1:]); cols.append(blk["E"] + idx[1:] - 1); vals.append(-np.ones(m - 1))
    soc_rhs = np.zeros(m); soc_rhs[0] = soc_start
    rhs.append(soc_rhs)
    A_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * m, 6 * m)).tocsr()
    lb = np.zeros(6 * m); ub = np.empty(6 * m)
    ub[blk["GD"]:blk["GD"] + m] = gplan
    ub[blk["C"]:blk["C"] + m] = L.E_MAX
    ub[blk["D"]:blk["D"] + m] = L.E_MAX
    ub[blk["S"]:blk["S"] + m] = pvv * DT
    lb[blk["E"]:blk["E"] + m] = L.SOC_MIN
    ub[blk["E"]:blk["E"] + m] = L.SOC_MAX
    ub[blk["E2"]:blk["E2"] + m] = np.inf
    return linprog(cost_vec, A_eq=A_eq, b_eq=np.concatenate(rhs),
                   bounds=list(zip(lb, ub)), method="highs")


def compare(G2, price2d, label, K=60):
    price_flat = np.concatenate([price2d[REP0 + d] for d in range(K)])
    plan_flat = G2[:K].reshape(-1)
    load_flat = load_2d[REP0:REP0 + K].reshape(-1)
    pv_flat = pv_2d[REP0:REP0 + K].reshape(-1)
    t0 = time.time()
    soc = L.SOC_INIT
    emg_daily = 0.0
    for d in range(K):
        res = settle_joint(price_flat[d * N:(d + 1) * N], plan_flat[d * N:(d + 1) * N],
                           load_flat[d * N:(d + 1) * N], pv_flat[d * N:(d + 1) * N], soc)
        soc = res.x[4 * N + N - 1]
        emg_daily += float(res.x[5 * N:6 * N].sum())
    soc = L.SOC_INIT
    emg_window = 0.0
    for w0 in range(0, K, 7):
        w1 = min(w0 + 7, K)
        res = settle_joint(price_flat[w0 * N:w1 * N], plan_flat[w0 * N:w1 * N],
                           load_flat[w0 * N:w1 * N], pv_flat[w0 * N:w1 * N], soc)
        m = w1 - w0
        emg_window += float(res.x[5 * m * N:6 * m * N].sum())
        soc = float(res.x[4 * m * N + m * N - 1])
    cost = L.BETA_EMG * float(price_flat @ np.zeros_like(price_flat))
    save = (emg_daily - emg_window) * L.BETA_EMG * float(price_flat.mean())
    print(label, "逐日 %10.2f kWh" % emg_daily, "7天滚动 %10.2f kWh" % emg_window,
          "降低 %8.2f kWh" % (emg_daily - emg_window),
          "(%+.2f%%)" % (-(emg_daily - emg_window) / emg_daily * 100))
    print("折算费用影响（按平均价×5）%.0f 元 / %d 天，外推 334 天约 %.0f 元"
          % (save, K, save / K * 334))
    print("用时 %.1f s" % (time.time() - t0))
    return emg_daily, emg_window


print("结算层短视诊断（前 60 个报告日，计划购电量固定为各问结果）")
compare(npz2["g"], np.tile(price_mean, (365, 1)), "问2(均值电价)")
compare(npz4["g42"], price_rt, "4-2(实时电价)")
print("npz 紧急购电 前60日", "问2 %.2f" % npz2["e"][:60].sum(),
      "4-2 %.2f" % npz4["e42"][:60].sum(), "kWh")
print("逐日结算相对 7 天滚动多出的紧急购电若不足总费用 1%，只算可记录的局限，不改变结论")
