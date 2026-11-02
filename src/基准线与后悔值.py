"""基准线、VSS 与后悔值：按实时电价完整结算（含紧急费）"""
import os, sys, io
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import lp_core as L
import stoch as ST

REP0 = 31
data = L.load_all(verbose=False)
load_2d, pv_2d = data["actual"]["load_kw"], data["actual"]["pv_kw"]
dates = data["actual"]["dates"]
price_mean = data["typical"]["price"]
price_rt = data["price_rt"]["price"]


def settle_full(price_2d, g_plan_2d, days):
    """给定计划购电量与电价，按当日实际做全天结算，返回 (购电费, 紧急费, 合计)。"""
    buy_cost = emg_cost = 0.0
    soc = L.SOC_INIT
    for d in days:
        traj = ST.soc_trajectory(price_2d[d], g_plan_2d[d], load_2d[d], pv_2d[d], soc)
        if traj is None:
            continue
        soc = traj[0][-1]
        buy_cost += float(price_2d[d] @ g_plan_2d[d])
        emg_cost += float(L.BETA_EMG * (price_2d[d] @ traj[4]))
    return buy_cost, emg_cost, buy_cost + emg_cost


days = list(range(REP0, L.N_DAY))
report_slice = slice(REP0, L.N_DAY)
price_rt_2d = price_rt
price_mean_2d = np.tile(price_mean, (L.N_DAY, 1))
npz2 = np.load("results/问2_结果.npz", allow_pickle=True)
npz3 = np.load("results/问3_结果.npz", allow_pickle=True)
npz4 = np.load("results/问4_结果.npz", allow_pickle=True)

# 完美信息多日 LP 最优值
perfect = L.solve_multiday_lp(np.tile(price_mean, (L.N_DAY, 1)), load_2d, pv_2d,
                              soc_init=L.SOC_INIT, use_emg=True, verbose=False)
assert perfect["status"] == 0, "完美信息多日 LP 求解失败"
cost_perfect = float((price_mean[None, :] * perfect["g"][report_slice]).sum())
cost_perfect_daily = 0.0
for d in days:
    daily = L.solve_day_lp(price_mean, load_2d[d], pv_2d[d],
                           soc_start=L.SOC_INIT, soc_end=L.SOC_INIT)
    cost_perfect_daily += daily["cost"]

plan_eev = np.zeros((L.N_DAY, L.N_T))
for d in range(5, L.N_DAY):
    load_fc = ST.load_forecast(load_2d, ST.day_type(dates), d) if d >= 5 else load_2d[d]
    pv_fc = ST.pv_forecast_persistence(pv_2d, ST.day_type(dates), d) if d >= 5 else pv_2d[d]
    plan_eev[d] = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0)
buy_eev, emg_eev, cost_eev_rt = settle_full(price_rt_2d, plan_eev, days)
cost_no_store = float((price_mean[None, :] * np.maximum(
    load_2d[report_slice] * L.DT - pv_2d[report_slice] * L.DT, 0)).sum())

print("三条基准线（报告期 334 天，统一完整结算口径）")
print("完美信息多日LP", "%.2f" % cost_perfect, "元",
      "紧急购电 %.3e kWh" % float(perfect["e"][report_slice].sum()))
print("逐日闭环变体", "%.2f" % cost_perfect_daily, "元，更高，不构成下界")
print("期望值策略(实时价结算)", "%.2f" % cost_eev_rt, "元")
print("无储能", "%.2f" % cost_no_store, "元")
cost_sp2 = float(npz2["cost_plan"]) + float(npz2["cost_emg"])
print("问2 两阶段SP", "%.2f" % cost_sp2, "元")
print("问3 滚动决策", "%.2f" % float(npz3["C_tot"]), "元")
print("4-2 两阶段SP", "%.2f" % float(npz4["C42"]), "元")
print("4-3 滚动决策", "%.2f" % float(npz4["C43"]), "元")

cost42 = float(npz4["C42"])
cost43 = float(npz4["C43"])
print("VSS(4-2)", "%.2f - %.2f = %+.2f 元" % (cost_eev_rt, cost42, cost_eev_rt - cost42),
      "%+.2f%%" % ((cost_eev_rt - cost42) / cost_eev_rt * 100))
print("VSS(4-3)", "%.2f - %.2f = %+.2f 元" % (cost_eev_rt, cost43, cost_eev_rt - cost43),
      "%+.2f%%" % ((cost_eev_rt - cost43) / cost_eev_rt * 100))

buy_a, emg_a, cost_a = settle_full(price_rt_2d, npz2["g_all"], days)
print("方案A 均值电价计划+实时价完整结算", "%.2f" % cost_a, "元",
      "购电 %.2f + 紧急 %.2f" % (buy_a, emg_a))
print("方案B 实时电价直接优化(4-2)", "%.2f" % cost42, "元")
print("后悔值", "%.2f" % (cost_a - cost42), "元", "%.2f%%" % ((cost_a - cost42) / cost42 * 100))

# 回写统一口径后悔值
out4 = {k: npz4[k] for k in npz4.files}
out4["C_det_in_rt"] = cost_a
out4["regret"] = cost_a - cost42
np.savez(os.path.join(ROOT, "results", "问4_结果.npz"), **out4)
print("已回写 results/问4_结果.npz 的 C_det_in_rt / regret")

print("完美信息下储能可节省", "%.2f" % (cost_no_store - cost_perfect), "元",
      "%.2f%%" % ((cost_no_store - cost_perfect) / cost_no_store * 100))
print("两阶段SP 下实际节省", "%.2f" % (cost_no_store - cost_sp2), "元",
      "%.2f%%" % ((cost_no_store - cost_sp2) / cost_no_store * 100))
print("不确定性的代价", "%.2f" % (cost_sp2 - cost_perfect), "元",
      "占完美信息费用的 %.1f%%" % ((cost_sp2 - cost_perfect) / cost_perfect * 100))
