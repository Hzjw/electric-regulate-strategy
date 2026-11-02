"""四问结果统计描述与关键参数敏感性分析。"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core as L

N = L.N_T


def desc(name, x, unit=""):
    x = np.asarray(x, float)
    print(name, "均值", "%.2f" % x.mean(), "标准差", "%.2f" % x.std(),
          "最小", "%.2f" % x.min(), "最大", "%.2f" % x.max(),
          "变异系数", "%.3f" % (x.std()/x.mean() if x.mean() != 0 else float("nan")), unit)


data = L.load_all(verbose=False)
q1 = np.load(os.path.join(L.RES, "问1_结果.npz"), allow_pickle=True)
q2 = np.load(os.path.join(L.RES, "问2_结果.npz"), allow_pickle=True)
q3 = np.load(os.path.join(L.RES, "问3_结果.npz"), allow_pickle=True)
q4 = np.load(os.path.join(L.RES, "问4_结果.npz"), allow_pickle=True)

buy1, chg1, dis1, spill1, soc1 = q1["g"], q1["c"], q1["d"], q1["s"], q1["E"]
desc("购电量 g_t (kWh)", buy1)
desc("充电量 c_t (kWh)", chg1)
desc("放电量 d_t (kWh)", dis1)
desc("储电量 E_t (kWh)", soc1)
dual = q1["dual"]
price_typ = q1["price"]
same = np.isclose(dual, price_typ, rtol=1e-6, atol=1e-8)
print("问1 不购电时段", int((buy1 < 1e-6).sum()), "充电时段", int((chg1 > 1e-6).sum()),
      "放电时段", int((dis1 > 1e-6).sum()),
      "空闲时段", int(((chg1 < 1e-6) & (dis1 < 1e-6)).sum()),
      "SOC 触上界", int(np.isclose(soc1, L.SOC_MAX).sum()),
      "触下界", int(np.isclose(soc1, L.SOC_MIN).sum()))
print("问1 SOC 利用率", "%.1f%%" % ((soc1.max()-soc1.min())/(L.SOC_MAX-L.SOC_MIN)*100),
      "对偶等于电价时段", int(same.sum()), "对偶低于电价时段", int((dual < price_typ - 1e-8).sum()),
      "其中正在放电", int(((dual < price_typ - 1e-8) & (dis1 > 1e-6)).sum()))

price2 = q2["price"]
buy2 = q2["g"]
soc2 = q2["E"]
# datetime64 包成 Series 后用 .dt 访问器
dates2 = pd.Series(pd.to_datetime(q2["dates"]))
cost_day = (price2 * buy2).sum(axis=1)
buy_day = buy2.sum(axis=1)
chg_day = q2["c"].sum(axis=1)
dis_day = q2["d"].sum(axis=1)
spill_day = q2["s"].sum(axis=1)
desc("日购电费 (元)", cost_day)
desc("日购电量 (kWh)", buy_day)
print("问2 报告期总费用", "%.2f" % cost_day.sum(), "元", "日均", "%.2f" % cost_day.mean(),
      "极差比", "%.2f" % (cost_day.max()/cost_day.min()),
      "总充放", "%.0f/%.0f" % (chg_day.sum(), dis_day.sum()), "总弃光", "%.1f" % spill_day.sum(),
      "分位数 P5/P25/P50/P75/P95",
      "%.0f %.0f %.0f %.0f %.0f" % tuple(np.percentile(cost_day, [5, 25, 50, 75, 95])))
# 等效循环 = 日放电量 / SOC 可用跨度
cycle = dis_day / (L.SOC_MAX - L.SOC_MIN)
soc_dawn = np.r_[L.SOC_INIT, soc2[:-1, -1]]
# 实时结算按当日求解，末端取下限
mth = dates2.dt.month.values
print("问2 日等效循环", "%.3f" % cycle.mean(), "每日0:00储电量 均值", "%.1f" % soc_dawn.mean(),
      "最小", "%.1f" % soc_dawn.min(), "最大", "%.1f" % soc_dawn.max(),
      "恰等于下界 %.0f 的天数" % L.SOC_MIN,
      int((np.abs(soc2[:, -1] - L.SOC_MIN) < 1e-6).sum()), "/", len(soc2),
      "月度费用(万元)", [round(cost_day[mth == m].sum()/10000, 1) for m in range(2, 13)],
      "月度购电量(MWh)", [round(buy_day[mth == m].sum()/100, 0) for m in range(2, 13)])

g_plan, g_adj, emg = q3["g_plan"], q3["g_adj"], q3["e"]
short_buy = np.maximum(g_plan - g_adj, 0)
over_buy = np.maximum(g_adj - g_plan, 0)
desc("紧急购电量 e (kWh/段)", emg)
print("问3 非零少买段", int((short_buy > 1e-6).sum()), "非零多买段", int((over_buy > 1e-6).sum()),
      "非零紧急段", int((emg > 1e-6).sum()), "/", short_buy.size,
      "紧急天数", int((emg.sum(axis=1) > 1e-6).sum()), "/", emg.shape[0],
      "调整量占计划量", "%.3f%%" % (g_adj.sum()/g_plan.sum()*100),
      "日均总偏差", "%.1f" % np.abs(g_adj - g_plan).sum(axis=1).mean())

g42 = q4["g42"]
price_rt = q4["price_rt"]
cost42_day = (price_rt * g42).sum(axis=1)
buy_all = q2["g_all"]
cost_a_day = (price_rt * buy_all[31:]).sum(axis=1)
regret_day = cost_a_day - cost42_day
desc("4-2 日购电费 (元)", cost42_day)
desc("逐日后悔值 (元)", regret_day)
# 0 维 object 数组用 .item() 取出
cost43 = q4["cb43"].item()
print("问4 后悔值为正", int((regret_day > 0).sum()), "天", "为负", int((regret_day < 0).sum()), "天",
      "中位数", "%.2f" % np.median(regret_day), "总和", "%.2f" % regret_day.sum(),
      "均值/标准差", "%.2f / %.2f" % (regret_day.mean(), regret_day.std()),
      "4-3 费用分解 基准/违约/超额/紧急",
      "%.2f / %.2f / %.2f / %.2f" % (cost43["base"], cost43["def"], cost43["exc"], cost43["emg"]),
      "紧急购电", "%.2f -> %.2f kWh" % (emg.sum(), q4["e43"].sum()),
      "下降", "%.1f%%" % ((1 - q4["e43"].sum()/emg.sum())*100))

price2_flat = price2
net = None
g1_base = buy1.copy()


def q1_solve(eta=None, price=None, pmax=None, soc_half=None):
    """按指定参数重解问1，返回 (费用, 购电量, 放电量)。"""
    if eta is not None:
        L.ETA_C = eta
        L.ETA_D = eta
    old_pm, old_lo, old_hi = L.E_MAX, L.SOC_MIN, L.SOC_MAX
    if pmax is not None:
        L.E_MAX = pmax * L.DT
    if soc_half is not None:
        L.SOC_MIN = L.SOC_INIT - soc_half
        L.SOC_MAX = L.SOC_INIT + soc_half
    price_run = price if price is not None else q1["price"]
    res = L.solve_day_lp(price_run, q1["load_kw"], q1["pv_kw"],
                         soc_start=L.SOC_INIT, soc_end=L.SOC_INIT)
    out = (res["cost"], res["g"].sum(), res["d"].sum())
    L.ETA_C = 0.9 ** 0.5
    L.ETA_D = 0.9 ** 0.5
    L.E_MAX = old_pm
    L.SOC_MIN = old_lo
    L.SOC_MAX = old_hi
    return out


base_cost1, base_g1, base_d1 = q1_solve()
price0 = q1["price"]
price_mean = price0.mean()
cost_a = q1_solve(eta=0.9)
print("S1 充放电效率 往返0.90（本文采用）", "%.2f" % base_cost1, "元", "往返0.81", "%.2f" % cost_a[0],
      "元", "相对偏差", "%+.2f%%" % ((cost_a[0] - base_cost1) / base_cost1 * 100))
q1_solve()

s2 = []
for k in (0.5, 0.75, 1.0, 1.25, 1.5):
    price_s = price_mean + k * (price0 - price_mean)
    cost_s, buy_s, dis_s = q1_solve(price=price_s)
    s2.append((k, cost_s, buy_s, dis_s))
print("S2 价差倍数 0.5/1.0/1.5 的单日费用", "%.2f" % s2[0][1], "%.2f" % s2[2][1], "%.2f" % s2[4][1],
      "元", "相对偏差 %+.2f%%" % ((s2[0][1] - base_cost1) / base_cost1 * 100))
q1_solve()

s3 = []
for k in (0.25, 0.5, 0.75, 1.0, 1.5):
    cost_s, buy_s, dis_s = q1_solve(pmax=L.P_MAX * k)
    s3.append((k, cost_s, buy_s, dis_s))
print("S3 电池功率 0.5x/1.0x 的单日费用", "%.2f" % s3[1][1], "%.2f" % s3[3][1], "元",
      "相对偏差 %+.2f%%" % ((s3[1][1] - base_cost1) / base_cost1 * 100))
q1_solve()

s4 = []
for k in (0.25, 0.5, 0.75, 1.0):
    cost_s, buy_s, dis_s = q1_solve(soc_half=4800 * k)
    s4.append((k, cost_s, buy_s, dis_s))
print("S4 可用区间半宽 0.5x/1.0x 的单日费用", "%.2f" % s4[1][1], "%.2f" % s4[3][1], "元",
      "相对偏差 %+.2f%%" % ((s4[1][1] - base_cost1) / base_cost1 * 100))
q1_solve()

for k in (0.8, 1.0, 1.2):
    cost_s, buy_s, dis_s = q1_solve(price=price0 * k)
    print("S6 电价倍数", "%.2f" % k, "费用", "%.2f" % cost_s, "购电量", "%.0f" % buy_s,
          "费用/倍数", "%.2f" % (cost_s / k))
q1_solve()
# 价格齐次性：费用随电价等比，购电量不变

# 每次全年求解约 6 s
price_2d = np.tile(q2["price"], (L.N_DAY, 1))
load_2d = data["actual"]["load_kw"]
pv_2d = data["actual"]["pv_kw"]
res2 = []
t0 = time.time()
res_base = L.solve_multiday_lp(price_2d, load_2d, pv_2d)
res2.append(("往返 0.90（本文采用）", res_base["cost"]))
L.ETA_C = L.ETA_D = 0.9
t0 = time.time()
res_a = L.solve_multiday_lp(price_2d, load_2d, pv_2d)
res2.append(("往返 0.81", res_a["cost"]))
L.ETA_C = L.ETA_D = 0.9 ** 0.5
# 截断到最后 11 天前，估计边界效应
sl_last = slice(0, 354)
r_trunc = L.solve_multiday_lp(price_2d[:354], load_2d[:354], pv_2d[:354])
print("问2 全年 往返0.90", "%.2f" % res_base["cost"], "往返0.81", "%.2f" % res_a["cost"],
      "元", "相对偏差", "%+.2f%%" % ((res_a["cost"] - res_base["cost"]) / res_base["cost"] * 100),
      "截到 354 天", "%.2f" % r_trunc["cost"], "元",
      "年末可提前变现", "%.0f" % (L.SOC_MAX - L.SOC_MIN), "kWh")

print("敏感性与统计分析完成，已生成 fig21_parameter_sensitivity.png / fig22_result_distribution.png")
