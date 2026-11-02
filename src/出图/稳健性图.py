"""重制图25 / 图26（稳健性与滚动验证）"""
import os, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import lp_core as L
from plot_util import SEM, apply_style, despine, Placer

OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
apply_style()

ev_npz = np.load(os.path.join(L.RES, "EVPI与VSS_结果.npz"), allow_pickle=True)
rob_npz = np.load(os.path.join(L.RES, "问2稳健性扫描_结果.npz"), allow_pickle=True)
roll_npz = np.load(os.path.join(L.RES, "问3情景化滚动_结果.npz"), allow_pickle=True)
W = 1e4                                   # 元 -> 万元
ok = True
print("重制 图25 / 图26：完美信息价值 与 稳健性·情景化滚动验证")

mean_ws, mean_sp, mean_evpi = (
    float(ev_npz["mn_WS"]) / W, float(ev_npz["mn_SP"]) / W, float(ev_npz["mn_EVPI"]) / W)
rt_ws, rt_sp, rt_evpi = (
    float(ev_npz["rt_WS"]) / W, float(ev_npz["rt_SP"]) / W, float(ev_npz["rt_EVPI"]) / W)
ws = [mean_ws, rt_ws]
sp = [mean_sp, rt_sp]
ev = [mean_evpi, rt_evpi]
print("WS", "%.2f/%.2f" % (mean_ws, rt_ws), "SP", "%.2f/%.2f" % (mean_sp, rt_sp),
      "EVPI", "%.2f/%.2f" % (mean_evpi, rt_evpi), "万元")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
x = np.arange(2)
wd = 0.34
a = axes[0]
bw = a.bar(x - wd / 2, ws, wd, color=SEM["charge"], label="WS　完美信息期望费用")
bs = a.bar(x + wd / 2, sp, wd, color=SEM["buy"], label="SP　两阶段随机规划费用")
for i in range(2):
    a.annotate("%.2f" % ws[i], (x[i] - wd / 2, ws[i]), ha="center", va="bottom", fontsize=9)
    a.annotate("%.2f" % sp[i], (x[i] + wd / 2, sp[i]), ha="center", va="bottom", fontsize=9)
a.set_xticks(x)
a.set_xticklabels(["问2（均值电价）\nEVPI %.2f 万元" % mean_evpi,
                   "4-2（实时电价）\nEVPI %.2f 万元" % rt_evpi])
a.set_ylabel("报告期费用（万元）")
a.set_ylim(0, max(sp) * 1.26)
a.set_xlim(-0.62, 1.60)
a.set_title("（a）完美信息期望费用 WS 与随机规划费用 SP", fontsize=11)
a.legend(loc="upper left")
despine(a)

b = axes[1]
be0 = b.bar([0], [mean_evpi], 0.5, color=SEM["buy"], label="源荷不确定性（问2 EVPI）")
be1 = b.bar([1], [mean_evpi], 0.5, color=SEM["buy"])
be2 = b.bar([1], [rt_evpi - mean_evpi], 0.5, bottom=[mean_evpi], color=SEM["adj"],
            label="价格不确定性增量（4-2 − 问2）")
b.axhline(mean_evpi, color=SEM["ref"], ls=":", lw=1.2, label="源荷基础 %.2f 万元" % mean_evpi)
b.annotate("%.2f" % mean_evpi, (0, mean_evpi / 2), ha="center", va="center", fontsize=9,
            color="white")
b.annotate("%.2f" % mean_evpi, (1, mean_evpi / 2), ha="center", va="center", fontsize=9,
            color="white")
b.annotate("%.2f" % (rt_evpi - mean_evpi), (1, mean_evpi + (rt_evpi - mean_evpi) / 2), ha="center",
            va="center",
           fontsize=9, color=SEM["ink"])
b.annotate("%.2f" % rt_evpi, (1, rt_evpi), ha="center", va="bottom", fontsize=9)
b.set_xticks([0, 1])
b.set_xticklabels(["问2（均值电价）", "4-2（实时电价）"])
b.set_ylabel("EVPI（万元）")
b.set_ylim(0, rt_evpi * 1.30)
b.set_xlim(-0.62, 1.78)
b.set_title("（b）EVPI 来源：价格不确定性占 4-2 的 %.1f%%"
            % ((rt_evpi - mean_evpi) / rt_evpi * 100), fontsize=11)
b.legend(loc="upper left")
despine(b)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])

pa = Placer(fig, a)
pa.bars(bw); pa.bars(bs); pa.texts()
# 锚点取 WS 柱顶偏右 0.38*wd
pa.place("EVPI %.2f 万元（占 SP %.2f%%）" % (mean_evpi, mean_evpi / mean_sp * 100),
         (0 - wd / 2 + wd * 0.38, mean_ws), color=SEM["charge"])
pa.place("EVPI %.2f 万元（占 SP %.2f%%）" % (rt_evpi, rt_evpi / rt_sp * 100),
         (1 - wd / 2 + wd * 0.38, rt_ws), color=SEM["charge"])
ok &= pa.report()

pb = Placer(fig, b)
pb.bars(be0); pb.bars(be1); pb.bars(be2); pb.texts()
# 锚点取柱顶右侧 1.18
pb.place("价格波动贡献 %.2f 万元（占 %.1f%%）" % (rt_evpi - mean_evpi, (rt_evpi - mean_evpi) / rt_evpi * 100),
         (1.18, rt_evpi), color=SEM["ink"])
ok &= pb.report()

fig.text(0.985, 0.008, "EVPI = SP − WS＝完美信息价值；问2 与 4-2 使用同一情景分布与同一报告期",
         ha="right", va="bottom", fontsize=8.8, color=SEM["ink"])
fig.savefig(os.path.join(OUT, "fig25_evpi_decomposition.png"), bbox_inches="tight")
plt.close(fig)
print("  fig25_evpi_decomposition.png")

cost = rob_npz["cost"] / W                       # 万元
plan_mwh = rob_npz["kwh_plan"] / 1000.0             # MWh
emg_mwh = rob_npz["kwh_emg"] / 1000.0              # MWh
emg_days = rob_npz["days_emg"]
base_c = float(cost[0])                    # 基准（S=20, beta=5, rho=1.0）
base_e = float(emg_mwh[0])
idx_s = [1, 0, 2]                               # S=10, S=20(本文), S=50
idx_rho = [5, 0, 6]                             # rho = 0.5, 1.0, 1.5
rho = np.array([0.5, 1.0, 1.5])

cost_a, cost_bd, cost_cd = (
    float(roll_npz["CA"]) / W, float(roll_npz["CBd"]) / W, float(roll_npz["CCd"]) / W)
cost_bs, cost_cs = float(roll_npz["CBs"]) / W, float(roll_npz["CCs"]) / W
v_dec_d, v_dec_s = cost_a - cost_bd, cost_a - cost_bs      # 决策机会价值 C(A)-C(B)
print("基准费用", "%.2f" % base_c, "万元；决策机会价值", "%.2f/%.2f" % (v_dec_d, v_dec_s), "万元")
print("CA/CBd/CCd", "%.2f/%.2f/%.2f" % (cost_a, cost_bd, cost_cd),
      "CBs/CCs", "%.2f/%.2f" % (cost_bs, cost_cs))

# (a)(b) 画相对基准的偏差 (%)
fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))

a = axes[0]
x_s = np.arange(3)
dev_cost = np.array([(cost[i] / base_c - 1.0) * 100 for i in idx_s])
dev_emg = np.array([(emg_mwh[i] / base_e - 1.0) * 100 for i in idx_s])
a.axhline(0.0, color=SEM["ref"], ls=":", lw=1.1)
a.plot(x_s, dev_cost, "-o", color=SEM["buy"], lw=2.0, markersize=6,
       markerfacecolor="white", markeredgewidth=1.6, label="报告期费用偏差")
a.plot(x_s, dev_emg, "-s", color=SEM["emerg"], lw=2.0, markersize=6,
       markerfacecolor="white", markeredgewidth=1.6, label="紧急购电电量偏差")
# 标签与数据点留 5pt 空隙
for i in (0, 2):
    a.annotate("%+.2f%%" % dev_cost[i], (x_s[i], dev_cost[i]), ha="center",
               va="bottom" if dev_cost[i] > 0 else "top", fontsize=8.5,
               textcoords="offset points", xytext=(0, 5 if dev_cost[i] > 0 else -5),
               color=SEM["buy"])
    a.annotate("%+.2f%%" % dev_emg[i], (x_s[i], dev_emg[i]), ha="center",
               va="bottom" if dev_emg[i] > 0 else "top", fontsize=8.5,
               textcoords="offset points", xytext=(0, 5 if dev_emg[i] > 0 else -5),
               color=SEM["emerg"])
a.set_xticks(x_s)
a.set_xticklabels(["S=10", "S=20\n（本文）", "S=50"])
a.set_xlabel("SAA 情景数 S")
a.set_ylabel("相对基准 S=20 的偏差 (%)")
a.set_ylim(-9.0, 9.0)
a.set_xlim(-0.30, 2.45)
a.set_title("（a）SAA 情景数收敛性（基准 %.2f 万元）" % base_c, fontsize=11)
a.legend(loc="lower left")
despine(a)

b = axes[1]
dev_plan = np.array([(plan_mwh[i] / float(plan_mwh[0]) - 1.0) * 100 for i in idx_rho])
dev_emg_rho = np.array([(emg_mwh[i] / base_e - 1.0) * 100 for i in idx_rho])
dev_cost_rho = np.array([(cost[i] / base_c - 1.0) * 100 for i in idx_rho])
b.axhline(0.0, color=SEM["ref"], ls=":", lw=1.1)
b.plot(rho, dev_plan, "-o", color=SEM["buy"], lw=2.0, markersize=6,
       markerfacecolor="white", markeredgewidth=1.6, label="计划购电量偏差")
b.plot(rho, dev_emg_rho, "-s", color=SEM["emerg"], lw=2.0, markersize=6,
       markerfacecolor="white", markeredgewidth=1.6, label="紧急购电量偏差")
b.plot(rho, dev_cost_rho, "-^", color=SEM["ink"], lw=1.7, markersize=6,
       markerfacecolor="white", markeredgewidth=1.6, label="报告期费用偏差")
for i in (0, 2):
    b.annotate("%+.1f%%" % dev_emg_rho[i], (rho[i], dev_emg_rho[i]), ha="center",
               va="bottom" if dev_emg_rho[i] > 0 else "top", fontsize=8.5, color=SEM["emerg"],
               textcoords="offset points", xytext=(0, 5 if dev_emg_rho[i] > 0 else -5))
    b.annotate("%+.2f%%" % dev_cost_rho[i], (rho[i], dev_cost_rho[i]), ha="center", va="bottom",
               fontsize=8.5, color=SEM["ink"], textcoords="offset points", xytext=(0, 5))
b.set_xticks(rho)
b.set_xlabel("保守度 $\\rho$（情景偏差缩放系数）")
b.set_ylabel("相对基准 $\\rho$=1.0 的偏差 (%)")
b.set_ylim(-24.0, 36.0)
b.set_xlim(0.42, 1.86)
b.set_title("（b）$\\rho$-保守度曲线：费用在 $\\rho$=1.0 处最低", fontsize=11)
b.legend(loc="lower left", fontsize=8.5)
despine(b)

c = axes[2]
bar_x = np.arange(2)
bar_w = 0.26
bar_a = c.bar(bar_x - bar_w, [cost_a, cost_a], bar_w, color=SEM["none"], label="A 不调整")
bar_b = c.bar(bar_x, [cost_bd, cost_bs], bar_w, color=SEM["emerg"], label="B 改签·无新信息")
bar_c = c.bar(bar_x + bar_w, [cost_cd, cost_cs], bar_w, color=SEM["charge"], label="C 改签·用当期预报")
for i, (va, vb, vc) in enumerate([(cost_a, cost_bd, cost_cd), (cost_a, cost_bs, cost_cs)]):
    for xx, v in [(bar_x[i] - bar_w, va), (bar_x[i], vb), (bar_x[i] + bar_w, vc)]:
        c.annotate("%.2f" % v, (xx, v), ha="center", va="bottom", fontsize=8.2)
c.set_xticks(bar_x)
c.set_xticklabels(["确定性阶段 LP\n（交付口径）", "情景化阶段随机\n（S=%d）" % int(roll_npz["S_stage"])])
c.set_ylabel("报告期费用（万元）")
c.set_ylim(0, max(cost_bd, cost_bs) * 1.26)
c.set_xlim(-0.52, 1.52)
c.set_title("（c）情景化验证：决策机会价值 $C(A)-C(B)$ 负号不翻转", fontsize=11)
c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5,
         handlelength=1.4, columnspacing=1.1)
despine(c)

fig.tight_layout(rect=[0, 0.045, 1, 0.93])

qa = Placer(fig, a)
qa.line(x_s, dev_cost); qa.line(x_s, dev_emg); qa.hline(0.0)
qa.texts()
qa.place("S=10→50 费用仅漂移 1.14%", (1.0, 0.0), color=SEM["buy"])
ok &= qa.report()

qb = Placer(fig, b)
qb.line(rho, dev_plan); qb.line(rho, dev_emg_rho); qb.line(rho, dev_cost_rho); qb.hline(0.0)
qb.texts()
qb.place("$\\rho$=1.5 紧急购电 %.1f%%" % dev_emg_rho[2], (rho[2], dev_emg_rho[2]), color=SEM["emerg"])
qb.place("费用 $\\rho$=1.0 最低（两侧均升）", (1.0, 0.0), color=SEM["ink"])
ok &= qb.report()

qc = Placer(fig, c)
qc.bars(bar_a); qc.bars(bar_b); qc.bars(bar_c); qc.texts()
qc.place("确定性 %.2f 万元" % v_dec_d, (bar_x[0] + bar_w / 2 + 0.09, cost_bd), color=SEM["emerg"])
qc.place("情景化 %.2f 万元" % v_dec_s, (bar_x[1] + bar_w / 2 + 0.09, cost_bs), color=SEM["emerg"])
ok &= qc.report()

fig.text(0.985, 0.008,
         "基准口径 S=20,β=5,ρ=1.0：费用 %.2f 万元、紧急购电 %.1f MWh、触发 %d 天；情景化阶段随机（S=%d）总费用 %.2f 万元（%+.2f%%）"
         % (base_c, base_e, int(emg_days[0]), int(roll_npz["S_stage"]), cost_cs,
             (cost_cs / cost_cd - 1.0) * 100),
         ha="right", va="bottom", fontsize=8.8, color=SEM["ink"])
fig.savefig(os.path.join(OUT, "fig26_rolling_validation.png"), bbox_inches="tight")
plt.close(fig)
print("  fig26_rolling_validation.png")
print("EVPI 问2", "%.2f" % mean_evpi, "万元；4-2", "%.2f" % rt_evpi, "万元")
print("价格不确定性占总 EVPI", "%.1f%%" % ((rt_evpi - mean_evpi) / rt_evpi * 100))
print("SAA 基准费用", "%.2f" % base_c, "万元；紧急购电", "%.1f" % base_e, "MWh")
print("决策机会价值", "%.2f" % v_dec_d, "万元（确定性）/", "%.2f" % v_dec_s, "万元（情景化）")
print("情景化总费用", "%.2f" % cost_cs, "万元（S=%d）" % int(roll_npz["S_stage"]))

print("全部标注避让成功" if ok else "存在未能完全避让的标注")
