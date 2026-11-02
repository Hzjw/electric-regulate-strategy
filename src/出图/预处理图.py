"""重制图1~图8（数据预处理章）"""
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

D = L.load_all(verbose=False)
typical = D["typical"]; actual = D["actual"]; rt = D["price_rt"]
price1, load1, pv1 = typical["price"], typical["load_kw"], typical["pv_kw"]
load2, pv2, price4 = actual["load_kw"], actual["pv_kw"], rt["price"]
dates = actual["dates"]; month = dates.dt.month.values
N_T, N_DAY, DT = L.N_T, L.N_DAY, L.DT
MONTHS = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
hrs = np.arange(1, N_T + 1) * DT                  # 每段的时刻（小时，区间右端）

ok = True
def rep(p, name):
    global ok
    r = p.report()
    if not r:
        print("避让未通过", name)
    ok &= r
    return r

def boxplot_style(ax, data, face, edge, widths=0.6, flier_ms=2.2):
    return ax.boxplot(
        data, patch_artist=True, widths=widths, showfliers=True,
        boxprops=dict(facecolor=face, edgecolor=edge, linewidth=1.1, alpha=0.85),
        medianprops=dict(color=SEM["ink"], linewidth=1.6),
        whiskerprops=dict(color=SEM["ref"], linewidth=1.0),
        capprops=dict(color=SEM["ref"], linewidth=1.0),
        flierprops=dict(marker="o", markerfacecolor=SEM["emerg"], markeredgecolor="none",
                        markersize=flier_ms, alpha=0.55))

def reg_box(p, bp):
    """箱线图各元素登记为障碍物"""
    p.bars(bp["boxes"])
    for ln in list(bp["whiskers"]) + list(bp["caps"]) + list(bp["medians"]):
        p.line(ln.get_xdata(), ln.get_ydata())
    for f in bp["fliers"]:
        p.line(f.get_xdata(), f.get_ydata())

print("重制 图1 ~ 图8（数据预处理章）")

fig, axs = plt.subplots(3, 1, figsize=(12.2, 8.6), sharex=True)
a0 = axs[0]
a0.step(hrs, price1, where="post", color=SEM["price"], lw=1.7, label="分时电价")
a0.set_ylabel("电价 (元/kWh)")
a0.set_ylim(0.28, 1.68)
a0.legend(loc="upper left")
despine(a0)

a1 = axs[1]
a1.plot(hrs, load1, color=SEM["load"], lw=1.7, label="小区负载")
a1.set_ylabel("小区负载 (kW)")
a1.set_ylim(2800, 6600)
a1.legend(loc="upper left")
despine(a1)

a2 = axs[2]
a2.fill_between(hrs, 0, pv1, color=SEM["pv"], alpha=0.75, label="光伏预测")
a2.plot(hrs, pv1, color=SEM["pv_edge"], lw=1.1)
a2.set_ylabel("光伏预测 (kW)")
a2.set_xlabel("时刻 (小时)")
a2.set_xticks(np.arange(0, 25, 2))
a2.set_ylim(0, pv1.max() * 1.30)
a2.legend(loc="upper left")
despine(a2)
fig.tight_layout()

p = Placer(fig, a0)
p.line(np.repeat(hrs, 2)[1:], np.repeat(price1, 2)[:-1])   # 加密 step 折线
p.texts()
p.place("峰段 1.3952 元/kWh，峰谷比 3.76", (20.6667, 1.3952), color=SEM["price"])
p.place("谷段最低 0.3713 元/kWh", (5.6667, 0.3713), color=SEM["price"])
rep(p, "图1a")

p = Placer(fig, a1)
p.line(hrs, load1); p.texts()
p.place("负载峰值 5 959 kW（09:10）", (9.1667, 5958.9696))
rep(p, "图1b")

p = Placer(fig, a2)
p.line(hrs, pv1); p.texts()
p.place("光伏峰值 7 612 kW（12:10）", (12.1667, 7612.316))
rep(p, "图1c")

fig.savefig(os.path.join(OUT, "fig01_typical_day_curves.png"), bbox_inches="tight")
plt.close(fig)
print("  fig01_typical_day_curves.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
specs = [("（a）电价 (元/kWh)", price1, SEM["price"], SEM["price"], (0.20, 1.62),
          "峰谷极差 1.024 元/kWh，无离群点"),
         ("（b）小区负载 (kW)", load1, SEM["load"], SEM["load"], (2600, 7100),
          "箱体跨度 2 127 kW（中位 4 378 kW）"),
         ("（c）光伏预测 (kW)", pv1, SEM["pv"], SEM["pv_edge"], (0, 9600),
          "Q1 = 0 kW：逾 1/4 时段出力为零")]
boxes = []
for ax, (label, series, face, edge, ylim, note) in zip(axes, specs):
    boxes.append(boxplot_style(ax, series, face, edge, widths=0.5, flier_ms=5))
    ax.set_title(label, fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    despine(ax)
axes[0].set_ylabel("变量取值")
fig.tight_layout()

peaks = [float(price1.max()), float(load1.max()), float(pv1.max())]
for ax, bp, (label, series, face, edge, ylim, note), top_y in zip(axes, boxes, specs, peaks):
    p = Placer(fig, ax)
    reg_box(p, bp)
    p.texts()
    p.place(note, (1.0, top_y))
    rep(p, "图2 " + label)
fig.savefig(os.path.join(OUT, "fig02_boxplot_three_vars.png"), bbox_inches="tight")
plt.close(fig)
print("  fig02_boxplot_three_vars.png")

def monthly_box(year, title, ylab, fname, face, edge, ylim, tags):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    data = [year[month == m].ravel() for m in range(1, 13)]
    bp = boxplot_style(ax, data, face, edge, widths=0.6, flier_ms=2.2)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel(ylab)
    ax.set_xlim(0.4, 13.0)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    despine(ax)
    fig.tight_layout()
    p = Placer(fig, ax)
    reg_box(p, bp); p.texts()
    for note_txt, xy, colour in tags:
        p.place(note_txt, xy, color=colour)
    rep(p, title)
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print("  " + fname)

load_med = np.array([float(np.median(load2[month == m])) for m in range(1, 13)])
monthly_box(load2, "2025 年小区负载分月箱线图（季节规律清晰，无异常尖峰）", "负载 (kW)",
            "fig03_monthly_load_boxplot.png", SEM["load"], SEM["load"], (1000, 8800),
            [("7 月负载全年最高（中位 4 886 kW，较 5 月 +29.7%）", (7.0, 7978.8849), SEM["load"]),
             ("10 月最低 1 996 kW，为全年最小负荷", (10.0, 1995.7176), SEM["load"])])

monthly_box(pv2, "2025 年光伏实际功率分月箱线图（夜间恒零 → 各月 Q1 均为 0）", "光伏功率 (kW)",
            "fig04_monthly_pv_boxplot.png", SEM["pv"], SEM["pv_edge"], (0, 12600),
            [("4 月峰值 10 216 kW，为全年最大值", (4.0, 10216.2), SEM["ink"]),
             ("12 月最高 6 856 kW，仅为 4 月的 67%", (12.0, 6856.3), SEM["ink"])])

price_med = np.array([float(np.median(price4[month == m])) for m in range(1, 13)])
monthly_box(price4, "2025 年实时电价分月箱线图（冬高夏低，12 个月均无离群点）", "电价 (元/kWh)",
            "fig05_monthly_price_boxplot.png", SEM["price"], SEM["price"], (-0.05, 2.10),
            [("7 月峰值 1.794 元/kWh，为全年最大值", (7.0, 1.7936), SEM["price"]),
             ("12 月中位 0.877（较 5 月 +38.1%）", (12.0, 1.6698), SEM["price"])])

pv_hour = pv2.reshape(N_DAY, 24, 6).mean(axis=2)           # (365, 24) 逐小时均值
hours = np.arange(24)
hour_mean = pv_hour.mean(axis=0)
hour_p95 = np.percentile(pv_hour, 95, axis=0)
hour_max = pv2.reshape(N_DAY, 24, 6).max(axis=(0, 2))          # 各小时全年最大功率
night = [0, 1, 2, 3, 22, 23]                              # 22:00-04:00

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
a = axes[0]
a.fill_between(hours, 0, hour_p95, color=SEM["pv"], alpha=0.55, label="95 分位")
a.plot(hours, hour_mean, color=SEM["pv_edge"], lw=2.0, marker="o", markersize=4, label="日均值")
a.set_xticks(hours[::2])
a.set_xlabel("小时"); a.set_ylabel("光伏功率 (kW)")
a.set_title("（a）全年光伏日内平均曲线（标准钟形）", fontsize=11)
a.set_ylim(0, hour_p95.max() * 1.30)
a.legend(loc="upper left")
despine(a)

b = axes[1]
bc = b.bar(hours, hour_max, 0.72, color=SEM["pv"], edgecolor=SEM["pv_edge"], linewidth=0.7)
b.set_xticks(hours[::2])
b.set_xlabel("小时"); b.set_ylabel("全年最大光伏功率 (kW)")
# 夜间 03:00 小时最大功率 0.0266 kW
b.set_title("（b）各小时全年最大光伏功率（夜间 22:00-04:00 最大仅 %.4f kW）"
            % hour_max[night].max(), fontsize=11)
b.set_ylim(0, hour_max.max() * 1.18)
despine(b)
fig.tight_layout()

p = Placer(fig, a)
p.line(hours, hour_mean); p.line(hours, hour_p95); p.texts()
p.place("95 分位峰值 9 188 kW（日均 7 522 kW）", (12, float(hour_p95[12])))
rep(p, "图6a")

p = Placer(fig, b)
p.bars(bc); p.texts()
p.place("夜间最大仅 0.0266 kW", (3.38, float(hour_max[night].max())), color=SEM["emerg"])
p.place("正午 11:00 全年最大 10 216 kW", (11.38, float(hour_max[11])))
rep(p, "图6b")

fig.savefig(os.path.join(OUT, "fig06_pv_daynight_consistency.png"), bbox_inches="tight")
plt.close(fig)
print("  fig06_pv_daynight_consistency.png")

def zscore_flag(x, thr=3.0):
    x = np.asarray(x, float).ravel()
    sd = x.std(ddof=0)
    if sd == 0:
        return np.zeros(x.size, bool)
    return np.abs((x - x.mean()) / sd) > thr

def iqr_fence(x, k=1.5):
    x = np.asarray(x, float).ravel()
    q1v, q3v = np.percentile(x, [25, 75])
    iqr = q3v - q1v
    return q1v - k * iqr, q3v + k * iqr, iqr

def margins(x):
    """极值距判定线余量，单位 IQR"""
    x = np.asarray(x, float).ravel()
    mu, sd = x.mean(), x.std(ddof=0)
    lo, hi, iqr = iqr_fence(x)
    mz = min((mu + 3 * sd - x.max()) / iqr, (x.min() - (mu - 3 * sd)) / iqr)
    mi = min((hi - x.max()) / iqr, (x.min() - lo) / iqr)
    return mz, mi

labels = ["附件1\n电价", "附件1\n负载", "附件1\n光伏", "全年\n负载", "全年\n光伏", "全年\n电价"]
sets = [price1, load1, pv1, load2.ravel(), pv2.ravel(), price4.ravel()]
z_rate = np.array([zscore_flag(x).mean() * 100 for x in sets])
iqr_rate = np.array([(((x < iqr_fence(x)[0]) | (x > iqr_fence(x)[1])).mean() * 100) for x in sets])
margin_series = np.array([margins(x) for x in sets])

fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
xpos = np.arange(6); w = 0.38
a = axes[0]
ba = a.bar(xpos - w / 2, z_rate, w, label="Z-score 法 (|z|>3)", color=SEM["buy"])
bb = a.bar(xpos + w / 2, iqr_rate, w, label="IQR 法 (1.5 倍)", color=SEM["adj"])
for i, v in enumerate(z_rate):          # 0 值标注为 "0"
    a.annotate("0" if v == 0 else "%.4f%%" % v, (i - w / 2, v), ha="center", va="bottom",
                fontsize=8.5)
for i, v in enumerate(iqr_rate):
    a.annotate("0" if v == 0 else "%.4f%%" % v, (i + w / 2, v), ha="center", va="bottom",
                fontsize=8.5)
a.set_xticks(xpos); a.set_xticklabels(labels, fontsize=9)
a.set_xlim(-0.6, 6.4)
a.set_ylabel("检出比例 (%)")
a.set_title("（a）两种方法的检出率：12 组中 11 组为 0", fontsize=11)
a.set_ylim(0, z_rate.max() * 1.30)
a.legend(loc="upper left")
despine(a)

b = axes[1]
bc1 = b.bar(xpos - w / 2, margin_series[:, 0], w, label="Z-score 法", color=SEM["buy"])
bc2 = b.bar(xpos + w / 2, margin_series[:, 1], w, label="IQR 法", color=SEM["adj"])
for i, v in enumerate(margin_series[:, 0]):
    b.annotate("%.3f" % v, (i - w / 2, v), ha="center", va="bottom", fontsize=8.5)
for i, v in enumerate(margin_series[:, 1]):
    b.annotate("%.3f" % v, (i + w / 2, v), ha="center", va="bottom", fontsize=8.5)
b.axhline(0.0, color=SEM["emerg"], ls="--", lw=1.2)
b.set_xticks(xpos); b.set_xticklabels(labels, fontsize=9)
b.set_xlim(-0.6, 6.4)
b.set_ylabel("距判定线余量 (倍 IQR)")
b.set_title("（b）极值距判定线的余量（余量 ≤ 0 即被检出）", fontsize=11)
b.set_ylim(-0.35, 1.62)
b.legend(loc="upper right")
despine(b)
fig.tight_layout()

p = Placer(fig, a)
p.bars(ba); p.bars(bb); p.texts()
p.place("仅全年电价非零：0.0019%（1 点）", (5.19, float(z_rate[5])), color=SEM["emerg"])
rep(p, "图7a")

p = Placer(fig, b)
p.bars(bc1); p.bars(bc2); p.texts(); p.hline(0.0)
p.place("唯一检出点：全年电价（−0.009）", (5.19, float(margin_series[5, 0])), color=SEM["emerg"])
rep(p, "图7b")

fig.savefig(os.path.join(OUT, "fig07_outlier_detection.png"), bbox_inches="tight")
plt.close(fig)
print("  fig07_outlier_detection.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
a = axes[0]
cnt, edges, bc = a.hist(price4.ravel(), bins=80, color=SEM["price"], edgecolor="white",
                        linewidth=0.3)
a.axvline(0.05, color=SEM["emerg"], ls="--", lw=1.5, label="0.05 元/kWh 阈值")
a.set_xlabel("电价 (元/kWh)"); a.set_ylabel("频数")
a.set_title("（a）附件4 电价全局分布（右偏，左尾伸至近 0）", fontsize=11)
a.set_ylim(0, cnt.max() * 1.30)
a.legend(loc="upper right")
despine(a)

b = axes[1]
seg = price4.ravel()[:288]
b.plot(np.arange(288), seg, color=SEM["price"], lw=1.0)
b.set_xlabel("时段序号（前 2 天，共 288 段）"); b.set_ylabel("电价 (元/kWh)")
b.set_ylim(0.25, 1.95)
b.set_title("（b）电价时序片段（日内剧烈波动，夜谷昼峰）", fontsize=11)
despine(b)
fig.tight_layout()

p = Placer(fig, a)
p.bars(bc)
p.line([0.05, 0.05], list(a.get_ylim()))
p.texts()
p.place("0.43-0.45 元/kWh 处点数最多（2 980 点）", (float(edges[20]), float(cnt.max())))
p.place("9 点 < 0.05 元/kWh，全部落在正午", (0.05, float(a.get_ylim()[1]) * 0.90), color=SEM["emerg"])
rep(p, "图8a")

p = Placer(fig, b)
p.line(np.arange(288), seg); p.texts()
p.place("日内峰谷极差 1.25 元/kWh（全年日均 1.11）", (267, float(seg[267])), color=SEM["price"])
rep(p, "图8b")

fig.savefig(os.path.join(OUT, "fig08_price_diagnosis.png"), bbox_inches="tight")
plt.close(fig)
print("  fig08_price_diagnosis.png")

print("全部标注避让成功" if ok else "存在未能完全避让的标注")
