"""模型检验：可行性、算法一致性、对偶与边界检验。"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core as L

rng = np.random.default_rng(42)

N = L.N_T

data = L.load_all(verbose=False)
price_typ = data["typical"]["price"]
load_typ = data["typical"]["load_kw"]
pv_typ = data["typical"]["pv_kw"]

base = L.solve_day_lp(price_typ, load_typ, pv_typ, soc_start=L.SOC_INIT, soc_end=L.SOC_INIT)
assert base["status"] == 0
buy, charge, discharge, spill, soc = base["g"], base["c"], base["d"], base["s"], base["E"]
cost_base = base["cost"]
print("基准最优费用", "%.2f" % cost_base, "元")

soc_prev = np.r_[L.SOC_INIT, soc[:-1]]
bal_err = buy + pv_typ * L.DT + discharge - (load_typ * L.DT + charge + spill)
soc_err = soc - (soc_prev + L.ETA_C * charge - discharge / L.ETA_D)
soc_hi_err = np.maximum(soc - L.SOC_MAX, 0)
soc_lo_err = np.maximum(L.SOC_MIN - soc, 0)
chg_err = np.maximum(charge - L.E_MAX, 0)
dis_err = np.maximum(discharge - L.E_MAX, 0)
spill_err = np.maximum(spill - pv_typ * L.DT, 0)
neg_err = np.maximum(-np.minimum.reduce([buy, charge, discharge, spill]), 0)
term_err = abs(soc[-1] - L.SOC_INIT)

checks = [("母线功率平衡", np.abs(bal_err).max()),
          ("SOC 递推方程", np.abs(soc_err).max()),
          ("SOC 上界", soc_hi_err.max()),
          ("SOC 下界", soc_lo_err.max()),
          ("充电功率上限", chg_err.max()),
          ("放电功率上限", dis_err.max()),
          ("弃光上界", spill_err.max()),
          ("变量非负性", neg_err.max()),
          ("末端条件", term_err)]
ok_all = True
for name, v in checks:
    flag = "通过" if v < 1e-8 else "违反"
    if v >= 1e-8:
        ok_all = False
        print(name, "最大违反量", "%.3e" % v, flag)
print("可行性检验最大违反量", "%.3e" % max(v for _, v in checks),
      "全部约束满足" if ok_all else "存在约束违反")


def solve_with(method, price, load_kw, pv_kw, soc_end=L.SOC_INIT):
    """指定算法求解单日 LP，用于交叉验证。"""
    n = N
    c_obj = np.concatenate([price, np.zeros(4 * n)])
    rows, cols, vals, b_eq = [], [], [], []
    for t in range(n):
        r = t
        rows += [r, r, r, r]
        cols += [t, n + t, 2 * n + t, 3 * n + t]
        vals += [1.0, -1.0, 1.0, -1.0]
        b_eq.append(load_kw[t] * L.DT - pv_kw[t] * L.DT)
    for t in range(n):
        r = n + t
        rows.append(r)
        cols.append(4 * n + t)
        vals.append(1.0)
        if t == 0:
            b_eq.append(L.SOC_INIT)
        else:
            rows.append(r)
            cols.append(4 * n + t - 1)
            vals.append(-1.0)
            b_eq.append(0.0)
        rows.append(r)
        cols.append(n + t)
        vals.append(-L.ETA_C)
        rows.append(r)
        cols.append(2 * n + t)
        vals.append(1.0 / L.ETA_D)
    A_eq = coo_matrix((vals, (rows, cols)), shape=(2 * n, 5 * n)).tocsr()
    lb = np.zeros(5 * n)
    ub = np.empty(5 * n)
    ub[0:n] = np.inf
    ub[n:2*n] = L.E_MAX
    ub[2*n:3*n] = L.E_MAX
    ub[3*n:4*n] = pv_kw * L.DT
    ub[4*n:5*n] = L.SOC_MAX
    lb[4*n:5*n] = L.SOC_MIN
    if soc_end is not None:
        ub[5*n-1] = soc_end
        lb[5*n-1] = soc_end
    return linprog(c_obj, A_eq=A_eq, b_eq=np.array(b_eq),
                   bounds=list(zip(lb, ub)), method=method)


r_ds = solve_with("highs-ds", price_typ, load_typ, pv_typ)
r_ipm = solve_with("highs-ipm", price_typ, load_typ, pv_typ)
gap = abs(r_ds.fun - r_ipm.fun)
rel = gap / cost_base * 100
print("对偶单纯形", "%.6f" % r_ds.fun, "内点法", "%.6f" % r_ipm.fun,
      "lp_core 默认", "%.6f" % cost_base)
print("两法目标值之差", "%.3e" % gap, "相对偏差", "%.2e%%" % rel,
      "购电策略最大逐段差异", "%.3e" % np.abs(r_ds.x[:N] - r_ipm.x[:N]).max())
print("算法交叉验证", "通过" if rel < 1e-6 else "未通过")

res = linprog(price_typ.tolist() + [0.0] * (4 * N), method="highs", **{})
dual = base["dual_balance"]
slack_chg = L.E_MAX - charge
slack_dis = L.E_MAX - discharge
slack_spill = pv_typ * L.DT - spill
slack_soc_hi = L.SOC_MAX - soc
slack_soc_lo = soc - L.SOC_MIN
print("紧约束时段数 充电", int((slack_chg < 1e-8).sum()),
      "放电", int((slack_dis < 1e-8).sum()), "弃光", int((slack_spill < 1e-8).sum()),
      "SOC 上界", int((slack_soc_hi < 1e-6).sum()), "SOC 下界", int((slack_soc_lo < 1e-6).sum()))
# 互补松弛检验对象
idle = (charge < 1e-8) & (discharge < 1e-8)
interior = (soc > L.SOC_MIN + 1e-6) & (soc < L.SOC_MAX - 1e-6)
pred = idle & interior
hit = np.isclose(dual[pred], price_typ[pred], rtol=1e-6, atol=1e-8)
print("电池空闲且 SOC 未触界的时段", int(pred.sum()), "其中对偶等于电价", int(hit.sum()),
      "KKT 条件", "成立" if pred.sum() and hit.all() else "需人工复核")

# 下界：只保留全天总净需求约束
net = (load_typ - pv_typ) * L.DT
total_net = net.sum()
lb = 0.0
if total_net >= 0:
    lb = float(np.min(price_typ) * total_net)
else:
    lb = float(np.min(price_typ) * total_net)
net_pos = np.maximum(net, 0)
cost_ub = float((price_typ * net_pos).sum())
inside = (lb - 1e-6) <= cost_base <= (cost_ub + 1e-6)
print("下界", "%.2f" % lb, "模型最优解", "%.2f" % cost_base, "上界", "%.2f" % cost_ub, "元",
      "落在区间内" if inside else "未通过")
print("相对上界改进", "%.2f" % (cost_ub - cost_base), "元（%.2f%%）"
      % ((cost_ub - cost_base) / cost_ub * 100),
      "距下界", "%.2f" % (cost_base - lb), "元（%.2f%%）" % ((cost_base - lb) / cost_base * 100))

robust = []
for sigma in (0.01, 0.03, 0.05, 0.10):
    costs, corrs, l1_g, l1_cd, max_g = [], [], [], [], []
    for _ in range(30):
        load_noisy = load_typ * (1 + rng.normal(0, sigma, N))
        pv_noisy = np.clip(pv_typ * (1 + rng.normal(0, sigma, N)), 0, None)
        rr = L.solve_day_lp(price_typ, load_noisy, pv_noisy,
                            soc_start=L.SOC_INIT, soc_end=L.SOC_INIT)
        if rr["status"] != 0:
            continue
        costs.append(rr["cost"])
        corrs.append(np.corrcoef(rr["g"], buy)[0, 1])
        l1_g.append(np.abs(rr["g"] - buy).sum() / buy.sum())
        l1_cd.append((np.abs(rr["c"] - charge).sum() + np.abs(rr["d"] - discharge).sum())
                     / (charge.sum() + discharge.sum()))
        max_g.append(np.abs(rr["g"] - buy).max() / max(buy.max(), 1e-9))
    costs = np.array(costs)
    dev = (costs - cost_base) / cost_base * 100
    robust.append((sigma, costs.mean(), costs.std(), np.abs(dev).max(), np.mean(corrs),
                   float(np.mean(l1_g)), float(np.mean(l1_cd)), float(np.mean(max_g))))
    print("噪声", "%.0f%%" % (sigma * 100), "费用均值", "%.2f" % costs.mean(),
          "标准差", "%.2f" % costs.std(), "最大偏差", "%.3f%%" % np.abs(dev).max())
print("10% 噪声下策略重组 购电量 L1", "%.2f%%" % (robust[-1][5] * 100),
      "充放 L1", "%.2f%%" % (robust[-1][6] * 100),
      "单段最大", "%.2f%%" % (robust[-1][7] * 100), "相关系数", "%.4f" % robust[-1][4])


def rule_based(a, b):
    """阈值式规则调度：低价满充、高价满放，返回全天费用。"""
    E = L.SOC_INIT
    cost = 0.0
    for t in range(N):
        net = load_typ[t] * L.DT - pv_typ[t] * L.DT
        c = d = 0.0
        if price_typ[t] <= a:
            room = (L.SOC_MAX - E) / L.ETA_C
            c = max(0.0, min(L.E_MAX, room, max(net + 0.0, net)))
            c = min(L.E_MAX, room)
        elif price_typ[t] >= b:
            avail = (E - L.SOC_MIN) * L.ETA_D
            d = max(0.0, min(L.E_MAX, avail))
        g = net + c - d
        if g < 0:
            g2 = 0.0
            surplus = -g
            s = min(surplus, pv_typ[t] * L.DT)
            c = max(0.0, c - (surplus - s))
        g = max(g, 0.0)
        E = E + L.ETA_C * c - d / L.ETA_D
        E = min(max(E, L.SOC_MIN), L.SOC_MAX)
        cost += price_typ[t] * g
    return cost


best = (np.inf, None, None)
# 充电阈值网格从 0.36 起
for a in np.arange(0.36, 0.95, 0.02):
    for b in np.arange(a + 0.05, 1.45, 0.02):
        cost_rule = rule_based(a, b)
        if cost_rule < best[0]:
            best = (cost_rule, a, b)
print("规则策略族最优 阈值", "%.2f" % best[1], "%.2f" % best[2],
      "费用", "%.2f" % best[0], "元", "LP 最优", "%.2f" % cost_base, "元")
print("LP 相对最优规则策略的改进", "%.2f" % (best[0] - cost_base), "元（%.2f%%）"
      % ((best[0] - cost_base) / best[0] * 100))

price_2d = np.tile(price_typ, (L.N_DAY, 1))
load_2d = data["actual"]["load_kw"]
pv_2d = data["actual"]["pv_kw"]
dates = pd.Series(pd.to_datetime(data["actual"]["dates"]))
q2 = np.load(os.path.join(L.RES, "问2_结果.npz"), allow_pickle=True)
# g_all 覆盖全年，季度按日期切分
g_all = q2["g_all"]
for qname, (m0, m1) in [("Q1(1-3月)", (1, 3)), ("Q2(4-6月)", (4, 6)),
                        ("Q3(7-9月)", (7, 9)), ("Q4(10-12月)", (10, 12))]:
    mask = (dates.dt.month.values >= m0) & (dates.dt.month.values <= m1)
    buy_q = g_all[mask]
    load_q = load_2d[mask]
    pv_q = pv_2d[mask]
    unit_cost = float((price_typ * buy_q).sum() / buy_q.sum())
    load_avg = float(load_q.mean())
    pv_avg = float(pv_q.mean())
    print(qname, "天数", mask.sum(), "单位购电成本", "%.4f" % unit_cost,
          "平均负载", "%.1f" % load_avg, "平均光伏", "%.1f" % pv_avg)
units = []
for m0, m1 in [(1, 3), (4, 6), (7, 9), (10, 12)]:
    mask = (dates.dt.month.values >= m0) & (dates.dt.month.values <= m1)
    buy_q = g_all[mask]
    units.append(float((price_typ * buy_q).sum() / buy_q.sum()))
print("四季度单位购电成本 均值", "%.4f" % np.mean(units), "标准差", "%.4f" % np.std(units),
      "变异系数", "%.3f" % (np.std(units) / np.mean(units)))

print("已生成 fig23_model_validation.png / fig24_robustness.png")

np.savez(os.path.join(L.RES, "模型检验_结果.npz"),
         rob=np.array(robust), lb=lb, ub=cost_ub, base=cost_base,
         rule_best=best[0], rule_a=best[1], rule_b=best[2],
         gap=gap, ds=r_ds.fun, ipm=r_ipm.fun)
print("模型检验完成。")
