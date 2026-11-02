"""重制图28（四口径对照与 CVaR 前沿）"""
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

res = np.load(os.path.join(L.RES, "电价不确定性与CVaR_结果.npz"), allow_pickle=True)
cost = [float(res["cost_A"]), float(res["cost_B"]), float(res["cost_C"]), float(res["cost_D"])]
lam, cost_scan, cvar_scan = res["lam"], res["cost_scan"], res["cvar_scan"]
ok = True
print("重制 图28：四口径对照与 CVaR 风险前沿")

fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
a = axes[0]
labels = ["A 电价已知\n独立抽样", "B 电价已知\n三源联合抽样",
          "C 电价不确定\n独立抽样", "D 电价不确定\n三源联合抽样"]
# 按电价处理方式分组上色
cols = ["#8FB8D8", "#0072B2", "#F2B33D", "#C67C00"]
bc = a.bar(np.arange(4), np.array(cost) / 1e4, 0.55, color=cols)
for i, v in enumerate(cost):
    a.annotate("%.2f" % (v / 1e4), (i, v / 1e4), ha="center", va="bottom", fontsize=9)
a.set_xticks(np.arange(4))
a.set_xticklabels(labels, fontsize=8.5)
a.set_ylabel("报告期费用 (万元)")
a.set_ylim(min(cost) / 1e4 * 0.985, max(cost) / 1e4 * 1.06)
a.set_title("（a）四种口径的报告期费用", fontsize=11)
despine(a)

b = axes[1]
b.plot(cvar_scan / 1e4, cost_scan / 1e4, "-", color=SEM["buy"], lw=1.9, marker="o", markersize=6,
       markerfacecolor="white", markeredgewidth=1.5)
offs = {0.0: (0, 12), 0.05: (0, 12), 0.1: (0, 12), 0.2: (-4, -17)}
for lam_v, cx, cy in zip(lam, cvar_scan / 1e4, cost_scan / 1e4):
    if lam_v in offs:
        b.annotate("λ=%.2f" % lam_v, (cx, cy), textcoords="offset points", xytext=offs[lam_v],
                   ha="center", fontsize=8.5, color=SEM["ink"])
b.set_xlabel("日均 CVaR$_{95}$ (万元)   ← 尾部风险越低")
b.set_ylabel("报告期费用 (万元)")
b.set_title("（b）期望费用–CVaR 风险前沿", fontsize=11)
despine(b)
b.invert_xaxis()
fig.tight_layout()

pc = Placer(fig, a)
pc.bars(bc)
pc.texts()
# B 水平参考刻度
a.plot([3 - 0.34, 3 + 0.34], [cost[1] / 1e4] * 2, color=SEM["ref"], ls="--", lw=1.0, zorder=2)
pc.line([3 - 0.34, 3 + 0.34], [cost[1] / 1e4] * 2)
pc.place("相对 B：%+.2f 万元（%+.2f%%）"
         % ((cost[3] - cost[1]) / 1e4, (cost[3] - cost[1]) / cost[1] * 100),
         (3.24, cost[3] / 1e4), color=SEM["emerg"])
ok &= pc.report()

pd = Placer(fig, b)
pd.line(np.asarray(cvar_scan / 1e4), np.asarray(cost_scan / 1e4))
pd.texts()
i_lam02 = list(lam).index(0.2)
pd.place("λ 0→0.2：CVaR 降 %.1f%%，期望费用升 %.1f%%"
         % ((1 - cvar_scan[i_lam02] / cvar_scan[0]) * 100,
             (cost_scan[i_lam02] / cost_scan[0] - 1) * 100),
         (float(cvar_scan[i_lam02]) / 1e4, float(cost_scan[i_lam02]) / 1e4), color=SEM["ink"])
ok &= pd.report()

fig.savefig(os.path.join(OUT, "fig28_cvar.png"), bbox_inches="tight")
plt.close(fig)
print("  fig28_cvar.png")
print("四口径报告期费用（万元）", " / ".join("%.2f" % (v / 1e4) for v in cost))
print("口径 A→D 变化", "%+.2f%%" % ((cost[3] / cost[0] - 1) * 100),
      "；D 相对 B", "%+.2f%%" % ((cost[3] - cost[1]) / cost[1] * 100))
print("λ 0→0.2：CVaR 降", "%.1f%%" % ((1 - cvar_scan[i_lam02] / cvar_scan[0]) * 100),
      "期望费用升", "%.1f%%" % ((cost_scan[i_lam02] / cost_scan[0] - 1) * 100))
print("日均 CVaR95 最低", "%.2f" % (cvar_scan.min() / 1e4), "万元；最高",
      "%.2f" % (cvar_scan.max() / 1e4), "万元")
print("λ 扫描点", len(lam), "个，范围", "%.2f~%.2f" % (lam.min(), lam.max()))
print("四口径均值", "%.2f" % (sum(cost) / len(cost) / 1e4), "万元")
print("四口径费用极差", "%.2f" % ((max(cost) - min(cost)) / 1e4), "万元")
print("全部标注避让成功" if ok else "存在未能完全避让的标注")
