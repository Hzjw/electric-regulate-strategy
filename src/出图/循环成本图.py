"""重制图27（循环成本敏感性）"""
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
RES = L.RES


def load_csv(name):
    arr = np.genfromtxt(os.path.join(RES, name), delimiter=",", skip_header=1,
                        encoding="utf-8-sig")
    return arr[np.isfinite(arr[:, 0])]


# csv 列序
layer_day = load_csv("循环成本_单日.csv")     # c_cyc,购电费,循环成本,总成本,EFC,…
layer_year = load_csv("循环成本_年度.csv")     # ... 年弃光量,紧急购电,节省,节省率%,日均EFC
layer_sp = load_csv("循环成本_问2.csv")      # ... 相对无储能节省,节省率%
day_grid, day_cyc, day_total, day_efc, day_curt = (
    layer_day[:, 0], layer_day[:, 2], layer_day[:, 3], layer_day[:, 4], layer_day[:, 7])
year_grid, year_total, year_efc, year_save = (
    layer_year[:, 0], layer_year[:, 3], layer_year[:, 4], layer_year[:, 9])
year_curt, year_efc_daily = layer_year[:, 7], layer_year[:, 11]
sp_grid, sp_buy, sp_emg, sp_total = layer_sp[:, 0], layer_sp[:, 1], layer_sp[:, 2], layer_sp[:, 4]
sp_save = layer_sp[:, 11]

npz_main = np.load(os.path.join(RES, "循环成本_结果.npz"), allow_pickle=True)
npz_crit = np.load(os.path.join(RES, "循环成本_临界点.npz"), allow_pickle=True)
val1 = lambda k: float(npz_main[k])
val2 = lambda k: float(npz_crit[k])

C_ARB = val1("c_arb")
C_STOP_D = val1("c_stop_day")
C_STOP_Y = val1("c_stop_year")
C50_D, C10_D = val1("c50_day"), val1("c10_day")
C50_Y, C10_Y = val1("c50_year"), val1("c10_year")
C_NG_D, C_ZC_D = val1("c_no_grid"), val1("c_zero_curt")
C_NG_Y, C_ZC_Y = val1("c_no_grid_year"), val1("c_zero_curt_year")
SURP, SURP_Y = val1("surplus"), val1("surplus_year")
SAV_D0, SAV_Y0 = val1("sav_day0"), val1("sav_year0")
NS_DAY, NS_YEAR = val1("NS_DAY"), val1("NS_YEAR")
PI_MAX, PI_MIN = val2("pi_max"), val2("pi_min")
C_STOP_CP = val2("c_stop")

print("重制 图27 · 循环成本敏感性（只读 npz + csv，未重跑扫描）")
print("  解析上界 pi_max=%.4f，纯套利失效下界 c_arb=%.4f" % (PI_MAX, C_ARB))
print("  单日临界：套利停 %.4f / 弃光重现 %.4f / 电池闲置 %.6f" % (C_NG_D, C_ZC_D, C_STOP_D))
print("  年度临界：套利停 %.4f / 弃光重现 %s / 电池闲置 %.6f"
      % (C_NG_Y, ("nan（年度弃光从不归零）" if not np.isfinite(C_ZC_Y) else "%.4f" % C_ZC_Y), C_STOP_Y))
print("  基准：单日无储能 %.2f 元（节省 %.2f），年度无储能 %.2f 元（节省 %.2f）"
      % (NS_DAY, SAV_D0, NS_YEAR, SAV_Y0))

fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

a = axes[0, 0]
a.plot(day_grid, day_efc, color=SEM["dischg"], lw=2.2, label="等效全循环次数 EFC（次/日）")
a.set_xlabel("单位循环成本 $c_{cyc}$（元/kWh 放电量）")
a.set_ylabel("EFC（次/日）")
a.set_ylim(0.0, float(day_efc.max()) * 1.18)
a2 = a.twinx()
a2.plot(day_grid, day_curt, color=SEM["pv_edge"], lw=1.8, ls="--", label="日内弃光量（kWh）")
a2.set_ylabel("日内弃光量（kWh）")
a2.set_ylim(0.0, max(float(day_curt.max()), 1.0) * 1.35)
a2.grid(False)
despine(a)
a2.spines["top"].set_visible(False)
VLO, VHI = 0.60, 0.985          # 临界值引导线只画上部
a_lim = a.get_ylim(); a_lo, a_hi = a_lim[1] * VLO, a_lim[1] * VHI
# 三条临界线：前两条贯穿到底
for xv in (C_NG_D, C_ZC_D):
    a.vlines(xv, a_lim[0], a_hi, colors=SEM["ref"], linestyles=":", lw=1.4, zorder=1)
a.vlines(C_STOP_D, a_lo, a_hi, colors=SEM["ref"], linestyles=":", lw=1.4, zorder=1)
a.annotate("%.4f\n电池闲置" % C_STOP_D, (C_STOP_D, a_lo), textcoords="offset points",
           xytext=(0, -2), ha="center", va="top", fontsize=7.5,
           color=SEM["ref"], linespacing=1.15, zorder=2)
h1, l1 = a.get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.02, 0.02))
a.set_title("（a）典型日：EFC 阶梯下降，%.4f 元/kWh 之后弃光重新出现" % C_ZC_D, fontsize=11)

b = axes[0, 1]
b.plot(day_grid, day_total, color=SEM["ink"], lw=2.4, label="总成本（购电费＋循环成本）")
b.plot(day_grid, layer_day[:, 1], color=SEM["buy"], lw=1.8, label="其中：购电费")
b.plot(day_grid, day_cyc, color=SEM["dischg"], lw=1.7, label="其中：循环成本")
b.axhline(NS_DAY, color=SEM["none"], ls="--", lw=1.5, label="无储能购电费 %.0f 元" % NS_DAY)
b.set_xlabel("单位循环成本 $c_{cyc}$（元/kWh 放电量）")
b.set_ylabel("日内费用（元）")
b.set_ylim(0.0, NS_DAY * 1.20)          # y 轴自 0 起
b_lim = b.get_ylim()
b.vlines(C_STOP_D, b_lim[1] * VLO, b_lim[1] * VHI, colors=SEM["ref"], linestyles=":", lw=1.4,
          zorder=1)
b.legend(loc="center right")
b.set_title("（b）典型日：总成本始终低于无储能，直至电池完全闲置", fontsize=11)
despine(b)

c = axes[1, 0]
c.plot(year_grid, year_efc, color=SEM["dischg"], lw=2.2, label="年等效全循环次数（次/年）")
c.set_xlabel("单位循环成本 $c_{cyc}$（元/kWh 放电量）")
c.set_ylabel("年 EFC（次/年）")
c.set_ylim(0.0, float(year_efc.max()) * 1.18)
c2 = c.twinx()
c2.plot(year_grid, year_total / 1e4, color=SEM["ink"], lw=2.0, ls="--", label="年总成本（万元）")
c2.axhline(NS_YEAR / 1e4, color=SEM["none"], ls=":", lw=1.6,
           label="无储能年基准 %.0f 万元" % (NS_YEAR / 1e4))
c2.set_ylabel("年总成本（万元）")
c2.set_ylim(float(year_total.min()) / 1e4 * 0.93, NS_YEAR / 1e4 * 1.12)
c2.grid(False)
despine(c)
c2.spines["top"].set_visible(False)
c_lim = c.get_ylim()
for xv, line_label in ((C_NG_Y, "套利停"), (C_STOP_Y, "电池闲置")):
    c.vlines(xv, c_lim[1] * VLO, c_lim[1] * VHI, colors=SEM["ref"], linestyles=":", lw=1.4,
              zorder=1)
    # 挂在线段下端
    c.annotate("%.4f\n%s" % (xv, line_label), (xv, c_lim[1] * VLO), textcoords="offset points",
               xytext=(0, -2), ha="center", va="top", fontsize=7.5,
               color=SEM["ref"], linespacing=1.15, zorder=2)
h1, l1 = c.get_legend_handles_labels()
h2, l2 = c2.get_legend_handles_labels()
c.legend(h1 + h2, l1 + l2, loc="center left")
c.set_title("（c）报告期 334 天：年循环强度与年总成本同步退化", fontsize=11)

d = axes[1, 1]
d.plot(sp_grid, sp_total / 1e4, "o-", color=SEM["ink"], lw=2.2, ms=5,
        label="问2 报告期总费用（万元）")
d.plot(sp_grid, sp_buy / 1e4, "s--", color=SEM["buy"], lw=1.6, ms=4,
        label="其中：计划购电费（万元）")
d.plot(sp_grid, sp_emg / 1e4, "^--", color=SEM["emerg"], lw=1.6, ms=4,
        label="其中：紧急购电费（万元）")
d.axhline(NS_YEAR / 1e4, color=SEM["none"], ls=":", lw=1.7,
           label="无储能基准 %.0f 万元" % (NS_YEAR / 1e4))
for xi, yi in zip(sp_grid, sp_total / 1e4):
    d.annotate("%.0f" % yi, (xi, yi), ha="center", va="bottom", fontsize=9)
d.set_xlabel("单位循环成本 $c_{cyc}$（元/kWh 放电量）")
d.set_ylabel("报告期费用（万元）")
d.set_ylim(float(sp_total.min()) / 1e4 * 0.90, float(sp_total.max()) / 1e4 * 1.16)
d.legend(loc="center left")
d.set_title("（d）问2 两阶段随机规划：头条费用对循环成本的稳健性", fontsize=11)
despine(d)

# 与无储能基准的交点
x_cross, y_cross = float("nan"), float("nan")
for i in range(1, len(sp_grid)):
    if sp_save[i] <= 0.0 <= sp_save[i - 1]:
        x_cross = sp_grid[i - 1] + (0.0 - sp_save[i - 1]) * (
            sp_grid[i] - sp_grid[i - 1]) / (sp_save[i] - sp_save[i - 1])
        y_cross = float(np.interp(x_cross, sp_grid, sp_total / 1e4))
        break

fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))

ok = True


def efc_at(x):
    return float(np.interp(x, day_grid, day_efc))


def twin_pt(ax_to, ax_from, x, y):
    """从轴数据坐标换算为主轴坐标"""
    xy = ax_from.transData.transform(np.column_stack([np.asarray(x, float),
                                                      np.asarray(y, float)]))
    return ax_to.transData.inverted().transform(xy)


def twin_line(p, x, y, ax_to, ax_from):
    """登记从轴曲线（换算到主轴坐标）"""
    xy = twin_pt(ax_to, ax_from, x, y)
    p.line(xy[:, 0], xy[:, 1])


pa = Placer(fig, a)
pa.line(day_grid, day_efc)
twin_line(pa, day_grid, day_curt, a, a2)
for xv in (C_NG_D, C_ZC_D):                     # 前两条：贯穿到底
    pa.line(np.array([xv, xv]), np.array([a_lim[0], a_hi]))
pa.line(np.array([C_STOP_D, C_STOP_D]), np.array([a_lo, a_hi]))
pa.texts()
pa.place("① %.4f 停止电网套利" % C_NG_D, (C_NG_D, efc_at(C_NG_D)), color=SEM["buy"])
# 锚在 EFC 曲线中段 a_lo
pa.place("② %.4f 起弃光重现" % C_ZC_D, (C_ZC_D, a_lo), color=SEM["pv_edge"])
ok &= pa.report()

pb = Placer(fig, b)
pb.line(day_grid, day_total)
pb.line(day_grid, layer_day[:, 1])
pb.line(day_grid, day_cyc)
pb.hline(NS_DAY)
pb.line(np.array([C_STOP_D, C_STOP_D]), np.array([b_lim[1] * VLO, b_lim[1] * VHI]))
pb.texts()
pb.place("c=0 节省 %.0f 元（%.1f%%）" % (SAV_D0, SAV_D0 / NS_DAY * 100.0),
         (0.0, float(day_total[0])), color=SEM["charge"])
pb.place("③ c≥%.4f=无储能 %.0f 元" % (C_STOP_D, NS_DAY), (C_STOP_D, NS_DAY), color=SEM["none"])
ok &= pb.report()

pc = Placer(fig, c)
pc.line(year_grid, year_efc)
twin_line(pc, year_grid, year_total / 1e4, c, c2)
pc.hline(NS_YEAR / 1e4)
for xv in (C_NG_Y, C_STOP_Y):
    pc.line(np.array([xv, xv]), np.array([c_lim[1] * VLO, c_lim[1] * VHI]))
pc.texts()
pc.place("c=0 年循环 %.0f 次" % float(year_efc[0]), (float(year_grid[0]), float(year_efc[0])),
         color=SEM["dischg"])
b50 = float(np.interp(0.5, year_grid, year_save)) / 1e4
c2_at_05 = twin_pt(c, c2, [0.5], [float(np.interp(0.5, year_grid, year_total / 1e4))])[0]
pc.place("c=0.5 年节省仍 %.0f 万元" % b50, (0.5, float(c2_at_05[1])),
         color=SEM["charge"])
ok &= pc.report()

pd = Placer(fig, d)
pd.line(sp_grid, sp_total / 1e4)
pd.line(sp_grid, sp_buy / 1e4)
pd.line(sp_grid, sp_emg / 1e4)
pd.hline(NS_YEAR / 1e4)
pd.texts()
pd.place("c=0 总费用 %.0f 万元" % (float(sp_total[0]) / 1e4),
          (float(sp_grid[0]), float(sp_total[0]) / 1e4),
         color=SEM["ink"])
pd.place("c≈%.2f 起反超无储能 %.0f 万元" % (x_cross, NS_YEAR / 1e4), (x_cross, y_cross),
         color=SEM["emerg"])
ok &= pd.report()

fig.text(0.985, 0.008,
         "c_cyc < %.4f 元/kWh 时弃光仍全额消纳；c_cyc ≥ %.4f 元/kWh（=最高电价 %.4f）电池完全闲置，"
         "退化为无储能基准（单日 %.0f 元 / 年 %.0f 万元）"
         % (C_ZC_D, C_STOP_D, PI_MAX, NS_DAY, NS_YEAR / 1e4),
         ha="right", va="bottom", fontsize=8.8, color=SEM["ink"])
fig.savefig(os.path.join(OUT, "fig27_cycle_cost_sensitivity.png"), bbox_inches="tight")
plt.close(fig)
print("  fig27_cycle_cost_sensitivity.png")
print("  单日：c=0 节省 %.2f 元；c=0.5 总成本 %.2f 元（节省 %.2f 元）"
      % (SAV_D0, float(np.interp(0.5, day_grid, day_total)),
          float(NS_DAY - np.interp(0.5, day_grid, day_total))))
print("  年度：c=0 年 EFC %.1f（日均 %.4f）；c=0.5 年节省 %.2f 元"
      % (float(year_efc[0]), float(year_efc_daily[0]), np.interp(0.5, year_grid, year_save)))
print("  问2：c=0 总费用 %.2f 元；c≈%.3f 处反超无储能基准 %.2f 元"
      % (float(sp_total[0]), x_cross, NS_YEAR))
print("全部标注避让成功" if ok else "存在未能完全避让的标注")
