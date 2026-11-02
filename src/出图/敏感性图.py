"""重制图21 / 图22（敏感性与分布）"""
import os, sys, io, time
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

# 同色相深浅两级区分「族」
TINT = "#8FB8D8"

q1 = np.load(os.path.join(L.RES, "问1_结果.npz"), allow_pickle=True)
q2 = np.load(os.path.join(L.RES, "问2_结果.npz"), allow_pickle=True)
data = L.load_all(verbose=False)
hrs = np.arange(1, L.N_T + 1) * L.DT
ok = True

print("重制敏感性/分布图：图21 / 图22")

def q1_solve(eta=None, price=None, pmax=None, soc_half=None):
    """按指定参数重解问1，返回 (费用, 购电量, 放电量)"""
    eta_c0, eta_d0 = L.ETA_C, L.ETA_D
    e_max0, soc_lo0, soc_hi0 = L.E_MAX, L.SOC_MIN, L.SOC_MAX
    try:
        if eta is not None:
            L.ETA_C = L.ETA_D = eta
        if pmax is not None:
            L.E_MAX = pmax * L.DT
        if soc_half is not None:
            L.SOC_MIN = L.SOC_INIT - soc_half
            L.SOC_MAX = L.SOC_INIT + soc_half
        price_used = q1["price"] if price is None else price
        res = L.solve_day_lp(price_used, q1["load_kw"], q1["pv_kw"],
                             soc_start=L.SOC_INIT, soc_end=L.SOC_INIT)
    finally:
        L.ETA_C, L.ETA_D = eta_c0, eta_d0
        L.E_MAX, L.SOC_MIN, L.SOC_MAX = e_max0, soc_lo0, soc_hi0
    return float(res["cost"]), float(res["g"].sum()), float(res["d"].sum())


t_start = time.time()
base_cost, base_g, base_d = q1_solve()
price_day = q1["price"]
avg_price = price_day.mean()
print("基准 单日购电费", "%.2f" % base_cost, "元；购电量", "%.1f" % base_g, "kWh")

# S2 峰谷价差幅度
scan_price = []
for k in (0.5, 0.75, 1.0, 1.25, 1.5):
    cost_k, buy_k, dischg_k = q1_solve(price=avg_price + k * (price_day - avg_price))
    scan_price.append((k, cost_k, buy_k, dischg_k))
# S3 电池最大充放电功率
scan_pmax = []
for k in (0.25, 0.5, 0.75, 1.0, 1.5):
    cost_k, buy_k, dischg_k = q1_solve(pmax=L.P_MAX * k)
    scan_pmax.append((L.P_MAX * k, cost_k, buy_k, dischg_k))
# S4 电池可用容量区间（半宽）
scan_soc = []
for k in (0.25, 0.5, 0.75, 1.0):
    cost_k, buy_k, dischg_k = q1_solve(soc_half=4800.0 * k)
    scan_soc.append((4800.0 * k, cost_k, buy_k, dischg_k))

# S1 效率口径：全年模型
price_2d = np.tile(q2["price"], (L.N_DAY, 1))
load_2d = data["actual"]["load_kw"]
pv_2d = data["actual"]["pv_kw"]
eta_keep = (L.ETA_C, L.ETA_D)
try:
    cost_round90 = L.solve_multiday_lp(price_2d, load_2d, pv_2d, verbose=False)["cost"]
    L.ETA_C = L.ETA_D = 0.9                      # 口径A：充放各 0.9（往返 0.81）
    cost_round81 = L.solve_multiday_lp(price_2d, load_2d, pv_2d, verbose=False)["cost"]
finally:
    L.ETA_C, L.ETA_D = eta_keep
print("全年费用 往返0.90", "%.2f" % cost_round90,
      "元；往返0.81", "%.2f" % cost_round81, "元", "%.1f s" % (time.time() - t_start))
for name, tab, unit in (("S2 价差", scan_price, "k"), ("S3 功率", scan_pmax, "kW"),
    ("S4 容量", scan_soc, "kWh")):
    print("  %s:" % name, "  ".join("%.0f:%.2f元" % (a, b) for a, b, _, _ in tab))

fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))


def base_marker(ax, x, y, label="基准工况"):
    """基准工况用空心菱形标出"""
    ax.plot([x], [y], marker="D", ms=6.5, mfc="white", mec=SEM["emerg"],
            mew=1.6, ls="none", label=label, zorder=5)


a = axes[0, 0]
ka = np.array([r[0] for r in scan_price])
ca = np.array([r[1] for r in scan_price]) / 1000.0
a.plot(ka, ca, marker="o", ms=5.5, color=SEM["buy"], lw=2.0, label="单日购电费")
base_marker(a, 1.0, base_cost / 1000.0)
a.set_xlabel("峰谷价差倍数 k")
a.set_ylabel("单日购电费 (千元)")
a.set_title("（a）费用对峰谷价差的敏感性", fontsize=11)
a.set_ylim(27.2, ca.max() * 1.055)
a.legend(loc="lower left")
despine(a)
coef = np.polyfit(ka, ca * 1000.0, 1)
fit = np.polyval(coef, ka)
r2 = 1 - ((ca * 1000.0 - fit) ** 2).sum() / ((ca * 1000.0 - (ca * 1000.0).mean()) ** 2).sum()
txt_a1 = "价差 ×%.1f：费用 −%.2f%%" % (ka[-1], (1 - ca[-1] * 1000 / base_cost) * 100)
txt_a2 = "购电量几乎不变（%+.2f%%）" % ((scan_price[-1][2] - base_g) / base_g * 100)

b = axes[0, 1]
kb = np.array([r[0] for r in scan_pmax])
cb = np.array([r[1] for r in scan_pmax]) / 1000.0
b.plot(kb, cb, marker="s", ms=5.0, color=SEM["buy"], lw=2.0, label="单日购电费")
base_marker(b, L.P_MAX, base_cost / 1000.0, label="基准 %.0f kW" % L.P_MAX)
b.set_xlabel("最大充放电功率 (kW)")
b.set_ylabel("单日购电费 (千元)")
b.set_title("（b）费用对电池功率的敏感性", fontsize=11)
b.set_ylim(cb.min() * 0.965, cb.max() * 1.06)
b.legend(loc="lower left")
despine(b)
i_half = int(np.argmin(np.abs(kb - L.P_MAX * 0.5)))
txt_b1 = "功率减半：费用 %+.2f%%" % ((cb[i_half] * 1000 - base_cost) / base_cost * 100)
txt_b2 = "≥%.0f kW 后饱和（仅再降 %.2f%%）" % (L.P_MAX, (1 - cb[-1] * 1000 / base_cost) * 100)

c = axes[1, 0]
kc = np.array([r[0] for r in scan_soc])
cost_c = np.array([r[1] for r in scan_soc]) / 1000.0
c.plot(kc, cost_c, marker="^", ms=5.5, color=SEM["buy"], lw=2.0, label="单日购电费")
base_marker(c, 4800.0, base_cost / 1000.0, label="基准半宽 4 800 kWh")
c.set_xlabel("SOC 可用区间半宽 (kWh)")
c.set_ylabel("单日购电费 (千元)")
c.set_title("（c）费用对电池容量的敏感性", fontsize=11)
c.set_ylim(cost_c.min() * 0.95, cost_c.max() * 1.08)
c.legend(loc="upper right")
despine(c)
txt_c1 = "半宽降至 2 400 / 1 200 kWh：%+.2f%% / %+.2f%%" % (
    (cost_c[1] * 1000 - base_cost) / base_cost * 100,
    (cost_c[0] * 1000 - base_cost) / base_cost * 100)

d = axes[1, 1]
vals = np.array([cost_round90, cost_round81]) / 1e4
xbar = np.arange(2)
bc = d.bar(xbar, vals, width=0.5, color=[SEM["buy"], TINT])
for i, v in enumerate(vals):
    d.annotate("%.1f" % v, (i, v), ha="center", va="bottom", fontsize=9)
d.set_xticks(xbar)
d.set_xticklabels(["往返 0.90\n（本文采用）", "往返 0.81"])
d.set_ylabel("全年费用 (万元)")
d.set_xlim(-0.6, 1.6)
d.set_ylim(0, vals.max() * 1.30)
d.set_title("（d）全年费用对效率口径的敏感性", fontsize=11)
despine(d)
txt_d = "往返效率降 0.09：多花 %.1f 万元（%+.2f%%）" % (
    (cost_round81 - cost_round90) / 1e4, (cost_round81 - cost_round90) / cost_round90 * 100)

fig.tight_layout(rect=[0, 0, 1, 0.955])

pa = Placer(fig, a)
pa.line(ka, ca); pa.texts()
pa.place(txt_a1, (ka[3], ca[3]))
pa.place(txt_a2, (ka[1], ca[1]))
ok &= pa.report()

pb1 = Placer(fig, b)
pb1.line(kb, cb); pb1.texts()
pb1.place(txt_b1, (kb[i_half], cb[i_half]))
pb1.place(txt_b2, (L.P_MAX, base_cost / 1000.0))
ok &= pb1.report()

pc = Placer(fig, c)
pc.line(kc, cost_c); pc.texts()
pc.place(txt_c1, (kc[1], cost_c[1]))
ok &= pc.report()

pd = Placer(fig, d)
pd.bars(bc); pd.texts()
pd.place(txt_d, (1.26, vals[1]), color=SEM["charge"])
ok &= pd.report()

fig.savefig(os.path.join(OUT, "fig21_parameter_sensitivity.png"), bbox_inches="tight")
plt.close(fig)
print("  fig21_parameter_sensitivity.png")
print("    (a) %s | %s（线性拟合 R²=%.3f）" % (txt_a1, txt_a2, r2))
print("    (b) %s | %s" % (txt_b1, txt_b2))
print("    (c) %s" % txt_c1)
print("    (d) %s" % txt_d)

g1 = q1["g"]; dual = q1["dual"]; price1 = q1["price"]
g2 = q2["g"]
months = (q2["dates"].astype("datetime64[M]").astype(int) % 12) + 1
day_cost = (q2["price"][None, :] * g2).sum(axis=1) / 1000.0      # 日购电费 (千元)
n_zero = int((g1 < 1e-6).sum())

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

a = axes[0]
cnt, bins, patches = a.hist(g1, bins=40, color=SEM["buy"], edgecolor="white", lw=0.6)
a.set_xlabel("分段购电量 (kWh)")
a.set_ylabel("频数")
a.set_title("（a）分段购电量分布", fontsize=11)
a.set_ylim(0, cnt.max() * 1.18)
despine(a)
txt_22a1 = "%.1f%% 的时段不购电（%d / 144 段）" % (n_zero / float(len(g1)) * 100, n_zero)
txt_22a2 = "购电量均值 %.0f kWh，最大 %.0f kWh" % (g1.mean(), g1.max())

b = axes[1]
gap = price1 - dual
b.fill_between(hrs, dual, price1, where=(gap > 1e-9), color=SEM["emerg"],
               alpha=0.18, lw=0, label="缺口（电池机会成本）")
b.plot(hrs, price1, color=SEM["price"], lw=1.7, label="实际电价")
b.plot(hrs, dual, color=SEM["buy"], lw=1.5, label="对偶变量（边际供电成本）")
b.set_xlabel("时刻 (小时)")
b.set_ylabel("元/kWh")
b.set_title("（b）边际电价与购电价的缺口\n（%d / 144 段对偶低于电价）" % int((gap > 1e-9).sum()),
            fontsize=11)
b.set_xlim(0, 24)
b.set_xticks(np.arange(0, 25, 4))
b.set_ylim(float(min(dual.min(), price1.min())) * 0.94, float(price1.max()) * 1.13)
b.legend(loc="upper left")
despine(b)
ig = int(np.argmax(gap))
txt_22b = "最大缺口 %.3f 元/kWh" % gap.max()

c = axes[2]
groups = [day_cost[months == m] for m in range(2, 13)]
bp = c.boxplot(groups, patch_artist=True, widths=0.6,
               boxprops=dict(facecolor=TINT, edgecolor=SEM["buy"], linewidth=0.9, alpha=0.85),
               medianprops=dict(color=SEM["ink"], linewidth=1.4),
               whiskerprops=dict(color=SEM["ref"], linewidth=0.9),
               capprops=dict(color=SEM["ref"], linewidth=0.9),
               flierprops=dict(marker="o", markersize=3, alpha=0.55,
                               markerfacecolor=SEM["emerg"], markeredgecolor="none"))
c.set_xticklabels(["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
c.set_xlabel("月份")
c.set_ylabel("日购电费 (千元)")
c.set_title("（c）日购电费的分月分布\n（全期日费用极差 %.1f 倍）" % (day_cost.max() / day_cost.min()),
            fontsize=11)
c.set_ylim(0, max(float(np.max(gr)) for gr in groups) * 1.18)
despine(c)
med = np.array([float(np.median(gr)) for gr in groups])
imax, imin = int(med.argmax()), int(med.argmin())
top_day = float(np.max(groups[imax]))
txt_22c1 = "%d 月 / %d 月中位数差 %.1f 倍" % (imax + 2, imin + 2, med[imax] / med[imin])

fig.tight_layout(rect=[0, 0, 1, 0.94])

pa2 = Placer(fig, a)
pa2.bars(patches); pa2.texts()
pa2.place(txt_22a1, (float(bins[1]), float(cnt[0])))
pa2.place(txt_22a2, (float(bins[-1]), float(cnt[-1])))
ok &= pa2.report()

pb2 = Placer(fig, b)
pb2.line(hrs, price1); pb2.line(hrs, dual); pb2.texts()
pb2.place(txt_22b, (hrs[ig], float(price1[ig])))
ok &= pb2.report()

pc2 = Placer(fig, c)
pc2.bars(bp["boxes"])
for grp in ("whiskers", "caps", "medians"):
    for ln in bp[grp]:
        pc2.line(ln.get_xdata(), ln.get_ydata())
pc2.texts()
pc2.place(txt_22c1, (imax + 1 + 0.32, top_day))
ok &= pc2.report()

fig.savefig(os.path.join(OUT, "fig22_result_distribution.png"), bbox_inches="tight")
plt.close(fig)
print("  fig22_result_distribution.png")
print("    (a) %s | %s" % (txt_22a1, txt_22a2))
print("    (b) %s" % txt_22b)
print("    (c) %s" % txt_22c1)

print("全部标注避让成功" if ok else "存在未能完全避让的标注")
