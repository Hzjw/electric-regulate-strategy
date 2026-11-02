"""重制图23 / 图24（模型检验章）"""
import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import lp_core as L
from plot_util import SEM, apply_style, despine, Placer, tag

OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
apply_style()

chk = np.load(os.path.join(L.RES, "模型检验_结果.npz"), allow_pickle=True)
q1 = np.load(os.path.join(L.RES, "问1_结果.npz"), allow_pickle=True)
q2 = np.load(os.path.join(L.RES, "问2_结果.npz"), allow_pickle=True)

rob = np.asarray(chk["rob"], float)          # [sigma, 均值, 标准差, max|偏差|%, corr, rlg, rcd, mxg]
sigma = rob[:, 0] * 100.0                      # 噪声标准差（%）
cost_mean = rob[:, 1] / 1000.0                      # 费用均值（千元）
cost_sd = rob[:, 2] / 1000.0                      # 费用标准差（千元）
corr = rob[:, 4]
l1_rel = rob[:, 5] * 100.0                      # 购电量 L1 相对变化（%）
max_rel = rob[:, 7] * 100.0                      # 单时段最大相对变化（%）

LB = float(chk["lb"])                        # 理论下界
UB = float(chk["ub"])                        # 无储能朴素策略上界
BASE = float(chk["base"])                    # LP 最优解
RULE, RA, RB = float(chk["rule_best"]), float(chk["rule_a"]), float(chk["rule_b"])
GAP, DS, IPM = float(chk["gap"]), float(chk["ds"]), float(chk["ipm"])
DV_LAST = (cost_mean[-1] * 1000.0 - BASE) / BASE * 100.0          # sigma=10% 费用相对偏差（%）

# 检验7：全年购电量 × 问1 均值电价
price1 = q1["price"]
g_all = q2["g_all"]
month_idx = pd.Series(pd.to_datetime(L.load_all(verbose=False)["actual"]["dates"])).dt.month.values
quarter_unit = np.array([float((price1 * g_all[(month_idx >= m0) & (month_idx <= m1)]).sum()
                        / g_all[(month_idx >= m0) & (month_idx <= m1)].sum())
                  for m0, m1 in ((1, 3), (4, 6), (7, 9), (10, 12))])

xs = np.arange(4.0)
x_labels = ["%d%%" % s for s in sigma]
ok = True
print("重制模型检验图：图23 / 图24")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

a = axes[0]
a.errorbar(xs, cost_mean, yerr=cost_sd, marker="o", ms=6, lw=1.8, capsize=4, zorder=4,
           color=SEM["buy"], label="最优费用（30 次均值 ± 标准差）")
a.axhline(BASE / 1000.0, color=SEM["ref"], ls="--", lw=1.3,
          label="基准费用 %.0f 元" % BASE)
a.set_xticks(xs); a.set_xticklabels(x_labels)
a.set_xlim(-0.45, 3.75); a.set_ylim(33.25, 35.45)
a.set_xlabel("输入负载 / 光伏的噪声标准差 σ")
a.set_ylabel("单日购电费 (千元)")
a.set_title("（a）费用对输入噪声的鲁棒性", fontsize=11)
a.legend(loc="upper left")
despine(a)

b = axes[1]
alg_names = ["highs-ds\n对偶单纯形", "highs-ipm\n内点法", "lp_core\n默认 highs"]
alg_costs = [DS, IPM, BASE]
bc = b.bar(np.arange(3.0), alg_costs, color=SEM["buy"], width=0.55)
for i, v in enumerate(alg_costs):
    b.annotate("%.2f 元" % v, (i, v), ha="center", va="bottom", fontsize=9)
b.set_xticks(np.arange(3.0)); b.set_xticklabels(alg_names)
b.set_ylim(0, max(alg_costs) * 1.30)
b.set_ylabel("目标值 (元)")
b.set_title("（b）两条独立算法路径的一致性", fontsize=11)
despine(b)

c = axes[2]
bound_names = ["忽略电池物理限制\n(理论下界)", "LP 最优解", "无储能朴素策略\n(上界)"]
bound_costs = [LB, BASE, UB]
cc = c.bar(np.arange(3.0), np.array(bound_costs) / 1000.0, width=0.55,
           color=[SEM["ref"], SEM["buy"], SEM["none"]])
for i, v in enumerate(bound_costs):
    c.annotate("%.2f" % (v / 1000.0), (i, v / 1000.0), ha="center", va="bottom", fontsize=9)
c.set_xticks(np.arange(3.0)); c.set_xticklabels(bound_names)
c.set_ylim(0, max(bound_costs) / 1000.0 * 1.22)
c.set_ylabel("单日购电费 (千元)")
c.set_title("（c）边界检验：最优值落在上下界之间", fontsize=11)
despine(c)

fig.tight_layout(rect=[0, 0.035, 1, 0.93])

p0 = Placer(fig, a)
p0.line(xs, cost_mean)
for i in range(len(xs)):                      # 误差棒竖段也登记为障碍
    p0.line(np.array([xs[i], xs[i]]),
             np.array([cost_mean[i] - cost_sd[i], cost_mean[i] + cost_sd[i]]))
p0.hline(BASE / 1000.0)
p0.texts()
p0.place("σ=10%% 费用仅偏 %+.2f%%（%+.0f 元）" % (DV_LAST, cost_mean[-1] * 1000.0 - BASE),
         (xs[-1] + 0.07, cost_mean[-1]), color=SEM["buy"])
ok &= p0.report()

p1 = Placer(fig, b)
p1.bars(bc); p1.texts()
p1.place("两法目标值之差 = %.1e 元" % GAP, (1.30, alg_costs[1]), color=SEM["charge"])
ok &= p1.report()

p2 = Placer(fig, c)
p2.bars(cc); p2.texts()
p2.place("较无储能上界省 %.0f 元（−%.1f%%）" % (UB - BASE, (UB - BASE) / UB * 100),
         (2.29, UB / 1000.0), color=SEM["charge"])
ok &= p2.report()

fig.text(0.995, 0.008,
         "最优值 %.2f 元 ∈ [%.2f, %.2f] 元（相对下界 +%.1f%%、相对上界 −%.1f%%）；"
         "σ=10%% 时费用标准差 %.0f 元"
         % (BASE, LB, UB, (BASE - LB) / BASE * 100, (UB - BASE) / UB * 100,
             float(chk["rob"][-1, 2])),
         ha="right", va="bottom", fontsize=8.6, color=SEM["ink"])
fig.savefig(os.path.join(OUT, "fig23_model_validation.png"), bbox_inches="tight")
plt.close(fig)
print("  fig23_model_validation.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

a = axes[0]
a.plot(xs, corr, marker="s", ms=7, lw=2.0, zorder=4, color=SEM["buy"])
for xv, yv in zip(xs, corr):
    # 标签放锚点右上
    a.annotate("%.4f" % yv, (xv + 0.09, yv + 0.0010), ha="left", va="bottom", fontsize=9)
a.set_xticks(xs); a.set_xticklabels(x_labels)
a.set_xlim(-0.45, 3.75); a.set_ylim(0.960, 1.012)
a.set_xlabel("输入负载 / 光伏的噪声标准差 σ")
a.set_ylabel("最优购电量与基准的相关系数")
a.set_title("（a）最优策略的稳定性", fontsize=11)
despine(a)

b = axes[1]
cmp_costs = [BASE, RULE]
bb = b.bar([0, 1], [v / 1000.0 for v in cmp_costs], width=0.5,
           color=[SEM["buy"], SEM["none"]])
for i, v in enumerate(cmp_costs):
    b.annotate("%.2f 千元" % (v / 1000.0), (i, v / 1000.0), ha="center", va="bottom", fontsize=9.5)
b.set_xticks([0, 1]); b.set_xticklabels(["LP 最优解", "最优阈值规则策略"])
b.set_ylim(0, RULE / 1000.0 * 1.34)
b.set_ylabel("单日购电费 (千元)")
b.set_title("（b）与朴素阈值策略的费用对比", fontsize=11)
despine(b)

fig.tight_layout(rect=[0, 0.035, 1, 0.93])

pa = Placer(fig, a)
pa.line(xs, corr); pa.texts()
pa.place("σ=10%% 相关系数仍 %.4f" % corr[-1], (xs[-1] + 0.06, corr[-1]), color=SEM["buy"])
ok &= pa.report()

pb = Placer(fig, b)
pb.bars(bb); pb.texts()
pb.place("较规则策略省 %.0f 元（−%.1f%%）" % (RULE - BASE, (RULE - BASE) / RULE * 100),
         (0.28, BASE / 1000.0), color=SEM["charge"])
ok &= pb.report()

fig.text(0.995, 0.008,
         "σ=10%% 时购电量 L1 变化 %.1f%%、单段最大 %.1f%%（费用仅偏 %+.2f%%）；"
         "规则阈值 a=%.2f / b=%.2f；四季度单位购电成本 %.4f~%.4f 元/kWh（变异系数 %.3f）"
         % (l1_rel[-1], max_rel[-1], DV_LAST, RA, RB, quarter_unit.min(), quarter_unit.max(),
            quarter_unit.std() / quarter_unit.mean()),
         ha="right", va="bottom", fontsize=8.6, color=SEM["ink"])
fig.savefig(os.path.join(OUT, "fig24_robustness.png"), bbox_inches="tight")
plt.close(fig)
print("  fig24_robustness.png")
print("理论下界", "%.2f" % LB, "元；LP 最优", "%.2f" % BASE, "元；无储能上界", "%.2f" % UB, "元")
print("两条算法路径目标值之差", "%.1e" % GAP, "元")
print("σ=10% 费用偏差", "%+.2f%%" % DV_LAST, "；购电量 L1 变化", "%.1f%%" % l1_rel[-1])
print("σ=10% 相关系数", "%.4f" % corr[-1], "；单段最大变化", "%.1f%%" % max_rel[-1])
print("四季度单位购电成本", "%.4f~%.4f" % (quarter_unit.min(), quarter_unit.max()), "元/kWh")
print("规则阈值 a=%.2f b=%.2f" % (RA, RB), "；较规则策略省", "%.0f" % (RULE - BASE), "元")

print("全部标注避让成功" if ok else "存在未能完全避让的标注")
