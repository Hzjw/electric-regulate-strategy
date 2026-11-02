"""重制图9~图11（问题一）"""
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
from plot_util import SEM, apply_style, despine, Placer, tag

OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
apply_style()

q1 = np.load(os.path.join(L.RES, "问1_结果.npz"), allow_pickle=True)
hrs = np.arange(1, L.N_T + 1) * L.DT
price, load_kw, pv_kw = q1["price"], q1["load_kw"], q1["pv_kw"]
buy, charge, dischg, curt, soc = q1["g"], q1["c"], q1["d"], q1["s"], q1["E"]
dual = q1["dual"]
cost1, base1 = float(q1["cost"]), float(q1["cost_base"])
ok = True
print("重制问题一图：图9 / 图10 / 图11")

fig, ax = plt.subplots(3, 1, figsize=(12.5, 8.8), sharex=True)
a = ax[0]
a.step(hrs, price, where="post", color=SEM["price"], lw=1.9, label="电价（附件1 均值日）", zorder=3)
a.fill_between(hrs, 0, price, where=(charge > 1e-6), color=SEM["charge"], alpha=0.20, step="post",
                label="充电时段")
a.fill_between(hrs, 0, price, where=(dischg > 1e-6), color=SEM["dischg"], alpha=0.20, step="post",
                label="放电时段")
a.set_ylabel("电价 (元/kWh)")
a.legend(ncol=3, loc="upper left")
despine(a)

b = ax[1]
b.fill_between(hrs, 0, pv_kw, color=SEM["pv"], alpha=0.55, label="光伏出力")
b.plot(hrs, pv_kw, color=SEM["pv_edge"], lw=0.9)
b.plot(hrs, load_kw, color=SEM["load"], lw=1.7, label="小区负载")
b.plot(hrs, buy / L.DT, color=SEM["buy"], lw=1.7, label="计划购电功率")
b.set_ylabel("功率 (kW)")
b.legend(ncol=3, loc="upper left")
despine(b)

cc = ax[2]
cc.plot(hrs, soc, color=SEM["soc"], lw=2.0, label="储电量 SOC")
cc.axhline(L.SOC_MAX, color=SEM["emerg"], ls="--", lw=1.0, label="运行上界 10 800 kWh")
cc.axhline(L.SOC_MIN, color=SEM["emerg"], ls="--", lw=1.0, label="运行下界 1 200 kWh")
cc.axhline(6000, color=SEM["ref"], ls=":", lw=1.1, label="初值 / 末端 6 000 kWh")
cc.set_ylabel("储电量 (kWh)")
cc.set_xlabel("时刻 (小时)")
cc.set_xticks(np.arange(0, 25, 2))
cc.legend(ncol=4, loc="lower left")
despine(cc)
fig.tight_layout()

p0 = Placer(fig, a)
p0.line(np.repeat(hrs, 2)[1:], np.repeat(price, 2)[:-1])
p0.texts()
i_hi, i_lo = int(price.argmax()), int(price.argmin())
p0.place("峰段 %.4f 放电" % price[i_hi], (hrs[i_hi], price[i_hi]), color=SEM["dischg"])
p0.place("谷段 %.4f 充电" % price[i_lo], (hrs[i_lo], price[i_lo]), color=SEM["charge"])
ok &= p0.report()

p1 = Placer(fig, b)
p1.line(hrs, pv_kw); p1.line(hrs, load_kw); p1.line(hrs, buy / L.DT)
p1.texts()
i_pv = int(pv_kw.argmax())
p1.place("光伏峰值 %.0f kW" % pv_kw[i_pv], (hrs[i_pv], pv_kw[i_pv]))
ok &= p1.report()

p2 = Placer(fig, cc)
p2.line(hrs, soc)
for yv in (L.SOC_MAX, L.SOC_MIN, 6000):
    p2.hline(yv)
p2.texts()
p2.place("最低放至 %.0f kWh" % soc.min(), (hrs[int(soc.argmin())], soc.min()))
p2.place("0:00 与 24:00 均为 6 000 kWh", (24.0, 6000.0))
ok &= p2.report()

fig.text(0.985, 0.012,
         "全天购电费 %.2f 元；无储能基准 %.2f 元，节省 %.2f%%（弃光由 %.0f kWh 降至 %.0f kWh）"
         % (cost1, base1, (base1 - cost1) / base1 * 100,
            float(np.maximum(pv_kw * L.DT - load_kw * L.DT, 0).sum()), float(curt.sum())),
         ha="right", va="bottom", fontsize=8.8, color=SEM["ink"])
fig.savefig(os.path.join(OUT, "fig09_q1_optimal_dispatch.png"), bbox_inches="tight")
plt.close(fig)
print("  fig09_q1_optimal_dispatch.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
a = axes[0]
a.plot(hrs, price, color=SEM["price"], lw=1.8, label="实际电价")
a.plot(hrs, dual, color=SEM["buy"], lw=1.5, ls="--", label="对偶变量（边际供电成本）")
a.set_xlabel("时刻 (小时)"); a.set_ylabel("元/kWh")
a.set_title("（a）对偶变量与实际电价", fontsize=11)
a.legend()
despine(a)

b = axes[1]
b.scatter(price, dual, s=16, color=SEM["buy"], alpha=0.7)
lim = [min(price.min(), dual.min()) * 0.98, max(price.max(), dual.max()) * 1.02]
b.plot(lim, lim, color=SEM["ref"], ls="--", lw=1.2)
b.set_xlabel("实际电价 (元/kWh)"); b.set_ylabel("对偶变量 (元/kWh)")
b.set_title("（b）两者吻合度（越贴近对角线越一致）", fontsize=11)
despine(b)
fig.tight_layout()

pa = Placer(fig, a)
pa.line(hrs, price); pa.line(hrs, dual); pa.texts()
pa.place("对偶变量与实际电价逐段重合", (hrs[int(np.argmax(np.abs(dual - price)))],
                                       float(dual[int(np.argmax(np.abs(dual - price)))])))
ok &= pa.report()

pb = Placer(fig, b)
pb.line(np.array(lim), np.array(lim)); pb.texts()
dev_max = float(np.max(np.abs(dual - price)))
pb.place("最大偏差 %.2e 元/kWh" % dev_max, (float(price[int(np.argmax(np.abs(dual - price)))]),
                                        float(dual[int(np.argmax(np.abs(dual - price)))])))
ok &= pb.report()
fig.savefig(os.path.join(OUT, "fig10_q1_marginal_price.png"), bbox_inches="tight")
plt.close(fig)
print("  fig10_q1_marginal_price.png")

net_load = np.maximum(load_kw * L.DT - pv_kw * L.DT, 0.0)
base_curt = float(np.maximum(pv_kw * L.DT - load_kw * L.DT, 0).sum())
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
a = axes[0]
bc0 = a.bar(["无储能基准", "有储能优化"], [base1, cost1],
            color=[SEM["none"], SEM["buy"]], width=0.5)
for i, v in enumerate([base1, cost1]):
    a.annotate("%.0f 元" % v, (i, v), ha="center", va="bottom", fontsize=9)
a.set_ylabel("全天购电费 (元)")
a.set_ylim(0, max(base1, cost1) * 1.16)
a.set_title("（a）储能带来的费用下降", fontsize=11)
despine(a)

b = axes[1]
bar_x = np.arange(2); w = 0.36
# 购电量与弃光量统一为 kWh
base_vals = [float(net_load.sum()), base_curt]
opt_vals = [float(buy.sum()), float(curt.sum())]
b.bar(bar_x - w / 2, base_vals, w, label="无储能", color=SEM["none"])
b.bar(bar_x + w / 2, opt_vals, w, label="有储能", color=SEM["buy"])
for i in range(2):
    b.annotate("%.0f" % base_vals[i], (i - w / 2, base_vals[i]), ha="center", va="bottom",
                fontsize=8.5)
    b.annotate("%.0f" % opt_vals[i], (i + w / 2, opt_vals[i]), ha="center", va="bottom",
                fontsize=8.5)
b.set_xticks(bar_x); b.set_xticklabels(["购电量 (kWh)", "弃光量 (kWh)"])
b.set_ylim(0, max(base_vals + opt_vals) * 1.22)
b.set_title("（b）电量结构对比", fontsize=11)
b.legend()
despine(b)
fig.tight_layout()

pa = Placer(fig, a)
pa.bars(bc0); pa.texts()
pa.place("节省 %.0f 元（−%.2f%%）" % (base1 - cost1, (base1 - cost1) / base1 * 100),
         (1.26, cost1), color=SEM["charge"])
ok &= pa.report()

pb = Placer(fig, b)
pb.texts()
pb.place("弃光全部被储能回收（%.0f → %.0f kWh）" % (base_curt, float(curt.sum())),
         (1.18, max(base_vals[1], opt_vals[1])), color=SEM["charge"])
ok &= pb.report()
fig.savefig(os.path.join(OUT, "fig11_q1_storage_value.png"), bbox_inches="tight")
plt.close(fig)
print("  fig11_q1_storage_value.png")
print("全天购电费", "%.2f" % cost1, "元；无储能基准", "%.2f" % base1, "元")
print("储能节省", "%.1f%%" % ((base1 - cost1) / base1 * 100))
print("弃光", "%.0f" % base_curt, "kWh →", "%.0f" % float(curt.sum()), "kWh")
print("对偶变量与实际电价最大偏差", "%.2e" % dev_max, "元/kWh")
print("已生成：图9 / 图10 / 图11")

print("全部标注避让成功" if ok else "存在未能完全避让的标注")
