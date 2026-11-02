"""重制图12~图20（结果总览）"""
import os, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import lp_core as L
from plot_util import SEM, apply_style, despine, Placer

OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
apply_style()

MONTHS = ["1月", "2月", "3月", "4月", "5月", "6月",
          "7月", "8月", "9月", "10月", "11月", "12月"]
KEY = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
LIGHT = "#8FB8D8"          # 同色相浅档：计划口径 / 方案A
DEEP = "#B87A00"           # 同色相深档：实时电价
SAMPLES = [0, 60, 120, 180, 240, 300]

q1 = np.load(os.path.join(L.RES, "问1_结果.npz"), allow_pickle=True)
q2 = np.load(os.path.join(L.RES, "问2_结果.npz"), allow_pickle=True)
q3 = np.load(os.path.join(L.RES, "问3_结果.npz"), allow_pickle=True)
q4 = np.load(os.path.join(L.RES, "问4_结果.npz"), allow_pickle=True)

data = L.load_all(verbose=False)
load_full, pv_full = data["actual"]["load_kw"], data["actual"]["pv_kw"]

dates = pd.to_datetime(q2["dates"])
date_num = mdates.date2num(dates)
hrs = np.arange(1, L.N_T + 1) * L.DT
price_q2, buy_q2, soc_q2 = q2["price"], q2["g"], q2["E"]
cost_daily = (price_q2 * buy_q2).sum(axis=1)

cost1, base1 = float(q1["cost"]), float(q1["cost_base"])
cost2 = float(q2["cost_plan"]) + float(q2["cost_emg"])
cost3 = float(q3["C_base"] + q3["C_def"] + q3["C_exc"] + q3["C_emg"])
cost42, cost43 = float(q4["C42"]), float(q4["C43"])

ok = True
print("重制结果总览图：图12 ~ 图20")


def _make_placer(fig, ax, regs):
    """按登记表 regs 构造 Placer"""
    p = Placer(fig, ax)
    for r in regs:
        k = r[0]
        if k == "bars":
            p.bars(r[1])
        elif k == "line":
            p.line(r[1], r[2], ax=(r[3] if len(r) > 3 else None))
        elif k == "hline":
            p.hline(r[1])
        elif k == "texts":
            p.texts()
        elif k == "legend":
            p.legend()
        elif k == "legend_of":
            keep = p.ax
            p.ax = r[1]
            p.legend()
            p.ax = keep
    return p


def place_all(tag, fig, ax, regs, plans, fs=8.6):
    """plans：每个标注的候选列表"""
    placed = []
    for cands in plans:
        done = False
        for c in cands:
            text, xy = c[0], c[1]
            size = c[2] if len(c) > 2 else fs
            p = _make_placer(fig, ax, regs)
            p.placed.extend(placed)          # 避开已放下的标注
            n0 = len(ax.texts)
            box = p.place(text, xy, fs=size)
            if not p.warn:
                placed.append(box)
                done = True
                break
            for t in ax.texts[n0:]:          # 回滚未放好的标注
                t.remove()
        if not done:
            print("严重：", tag, "找不到可完全避让的标注位")
            return False
    return True


def halo(ax, lw=2.4):
    """给标注文字加白色描边"""
    for t in ax.texts:
        if t.get_text().strip():
            t.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])


def day_index(day_key):
    return int(np.where(dates == pd.Timestamp(day_key))[0][0])


df = pd.DataFrame({"date": dates, "cost": cost_daily})
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
pivot = df.pivot_table(index="month", columns="day", values="cost", aggfunc="mean")
cost_grid = np.ma.masked_invalid(pivot.values)

fig, ax = plt.subplots(figsize=(15, 4.6))
cmap = LinearSegmentedColormap.from_list(
    "cost_seq", ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#EF3B2C", "#A50F15"])
cmap.set_bad("#FFFFFF")
im = ax.imshow(cost_grid, aspect="auto", cmap=cmap, interpolation="nearest")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([MONTHS[m - 1] for m in pivot.index])
ax.set_xticks(range(0, 31, 2))
ax.set_xticklabels(range(1, 32, 2))
ax.set_xlabel("日")
ax.set_ylabel("月份")
ax.grid(False)
despine(ax)
cb = fig.colorbar(im, ax=ax, pad=0.012)
cb.set_label("当日购电费 (元)", fontsize=9.5)
cb.outline.set_visible(False)
fig.tight_layout()

r_hi, c_hi = np.unravel_index(int(np.nanargmax(pivot.values)), pivot.values.shape)
r_lo, c_lo = np.unravel_index(int(np.nanargmin(pivot.values)), pivot.values.shape)
v_hi, v_lo = float(pivot.values[r_hi, c_hi]), float(pivot.values[r_lo, c_lo])
ok &= place_all("图12", fig, ax, [], [
    [("%d-%02d 峰值 %.2f 万元" % (pivot.index[r_hi], pivot.columns[c_hi], v_hi / 1e4), (c_hi, r_hi)),
     ("%.2f 万元（峰值）" % (v_hi / 1e4), (c_hi, r_hi))],
    [("%d-%02d 谷值 %.2f 万元（仅为峰值的 1/5）" % (pivot.index[r_lo], pivot.columns[c_lo], v_lo / 1e4),
      (c_lo, r_lo)),
     ("%d-%02d 谷值 %.2f 万元" % (pivot.index[r_lo], pivot.columns[c_lo], v_lo / 1e4), (c_lo, r_lo))],
])
halo(ax)
fig.savefig(os.path.join(OUT, "fig12_annual_cost_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("  fig12_annual_cost_heatmap.png")

month_kwh = np.array([buy_q2[df["month"].values == m].sum() for m in range(1, 13)])           # kWh
month_cost = np.array([cost_daily[df["month"].values == m].sum() for m in range(1, 13)])   # 元
month_price = np.where(month_kwh > 0, month_cost / np.maximum(month_kwh, 1e-9), np.nan)
month_labels = MONTHS[1:]                      # 报告期自 2/1 起
kwh_mwh = month_kwh[1:] / 1000.0                    # MWh
cost_wan = month_cost[1:] / 1e4                       # 万元
price_line = month_price[1:]
xpos = np.arange(len(month_labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
a = axes[0]
bc = a.bar(xpos, kwh_mwh, 0.62, color=SEM["buy"], label="月度购电量 (MWh)")
a.set_xticks(xpos); a.set_xticklabels(month_labels, fontsize=9)
a.set_ylabel("月度购电量 (MWh)")
a.set_ylim(0, float(kwh_mwh.max()) * 1.22)
a.set_title("（a）月度购电量与平均电价", fontsize=11)
despine(a)
a2 = a.twinx()
a2.plot(xpos, price_line, color=SEM["price"], marker="o", ms=4.5, lw=1.8, label="平均电价 (元/kWh)")
a2.set_ylabel("平均电价 (元/kWh)", color=SEM["price"])
a2.set_ylim(float(np.nanmin(price_line)) - 0.03, float(np.nanmax(price_line)) + 0.045)
a2.tick_params(axis="y", colors=SEM["price"])
a2.grid(False)
a2.spines["top"].set_visible(False)
a2.spines["left"].set_visible(False)
a2.spines["right"].set_color(SEM["price"])
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="upper left")

b = axes[1]
bc2 = b.bar(xpos, cost_wan, 0.62, color=LIGHT, label="月度购电费")
for i, v in enumerate(cost_wan):
    b.annotate("%.1f" % v, (i, v), ha="center", va="bottom", fontsize=8.5)
b.set_xticks(xpos); b.set_xticklabels(month_labels, fontsize=9)
b.set_ylabel("月度购电费 (万元)")
b.set_ylim(0, float(cost_wan.max()) * 1.26)
b.set_title("（b）月度购电费（12 月最高）", fontsize=11)
despine(b)
fig.tight_layout()

mi, mj = int(np.argmax(kwh_mwh)), int(np.argmax(cost_wan))
# 从轴曲线须传 ax=a2
ok &= place_all("图13a", fig, a,
                [("bars", bc), ("line", xpos, price_line, a2), ("texts",)], [
    [("%s购电量 %.0f MWh（全年峰值）" % (month_labels[mi], kwh_mwh[mi]), (mi + 0.30, kwh_mwh[mi])),
     ("%s峰值 %.0f MWh" % (month_labels[mi], kwh_mwh[mi]), (mi + 0.30, kwh_mwh[mi]))],
])
ok &= place_all("图13b", fig, b, [("bars", bc2), ("texts",)], [
    [("占全期 %.1f%%（%.1f 万元）" % (cost_wan[mj] / cost_wan.sum() * 100, cost_wan[mj]),
       (mj + 0.30, cost_wan[mj])),
     ("%.1f 万元（峰值月）" % cost_wan[mj], (mj + 0.30, cost_wan[mj]))],
])
fig.savefig(os.path.join(OUT, "fig13_monthly_cost_structure.png"), bbox_inches="tight")
plt.close(fig)
print("  fig13_monthly_cost_structure.png")

fig, axes = plt.subplots(4, 3, figsize=(16.5, 13.5), sharex=True)
for i, day_key in enumerate(KEY):
    i_day = day_index(day_key)
    i_full = i_day + 31                                  # 报告期第 di 天 → 全年第 di+31 天

    p0 = axes[i, 0]
    p0.step(hrs, price_q2, where="post", color=SEM["price"], lw=1.6)
    p0.fill_between(hrs, 0, price_q2, where=(q2["c"][i_day] > 1e-6), color=SEM["charge"],
                    alpha=0.24, step="post", label="充电")
    p0.fill_between(hrs, 0, price_q2, where=(q2["d"][i_day] > 1e-6), color=SEM["dischg"],
                    alpha=0.24, step="post", label="放电")
    p0.set_ylabel("电价\n(元/kWh)", fontsize=9)
    p0.set_title("%s  电价与充放电" % day_key, fontsize=10.5)
    p0.legend(fontsize=8, ncol=2, loc="upper left")
    despine(p0)

    p1 = axes[i, 1]
    p1.fill_between(hrs, 0, pv_full[i_full], color=SEM["pv"], alpha=0.55, label="光伏")
    p1.plot(hrs, pv_full[i_full], color=SEM["pv_edge"], lw=0.9)
    p1.plot(hrs, load_full[i_full], color=SEM["load"], lw=1.6, label="负载")
    p1.plot(hrs, buy_q2[i_day] / L.DT, color=SEM["buy"], lw=1.5, label="购电功率")
    p1.set_ylabel("功率\n(kW)", fontsize=9)
    p1.set_title("光伏 / 负载 / 购电（购电 %.1f MWh）" % (buy_q2[i_day].sum() / 1000), fontsize=10.5)
    p1.legend(fontsize=8, ncol=3, loc="upper left")
    despine(p1)

    p2 = axes[i, 2]
    p2.fill_between(hrs, 0, soc_q2[i_day], color=SEM["soc"], alpha=0.20)
    p2.plot(hrs, soc_q2[i_day], color=SEM["soc"], lw=1.8)
    p2.axhline(L.SOC_MAX, color=SEM["emerg"], ls="--", lw=0.9)
    p2.axhline(L.SOC_MIN, color=SEM["emerg"], ls="--", lw=0.9)
    p2.set_ylim(0, 11800)
    p2.set_ylabel("储电量\n(kWh)", fontsize=9)
    p2.set_title("当日 SOC 轨迹", fontsize=10.5)
    despine(p2)

axes[3, 0].set_xlabel("时刻 (小时)")
axes[3, 1].set_xlabel("时刻 (小时)")
axes[3, 2].set_xlabel("时刻 (小时)")
for ax_ in axes.ravel():
    ax_.set_xticks(np.arange(0, 25, 4))
    ax_.set_xlim(0, 24)
fig.tight_layout(rect=[0, 0, 1, 0.97])

i_solstice, i_summer = day_index("2025-12-21"), day_index("2025-06-21")
ratio = buy_q2[i_solstice].sum() / buy_q2[i_summer].sum()
ok &= place_all("图14冬至col0", fig, axes[3, 0], [("line", hrs, price_q2), ("texts",)], [
    [("冬至购电为夏至的 %.1f 倍" % ratio, (float(hrs[123]), float(price_q2[123]))),
     ("购电为夏至的 %.1f 倍" % ratio, (float(hrs[123]), float(price_q2[123]))),
     ("冬至/夏至 = %.1f 倍" % ratio, (float(hrs[123]), float(price_q2[123])))],
])
iz = int(np.argmax(soc_q2[i_solstice]))
ok &= place_all("图14冬至col2", fig, axes[3, 2],
                [("line", hrs, soc_q2[i_solstice]), ("hline", L.SOC_MAX), ("hline", L.SOC_MIN),
                  ("texts",)], [
    [("冬至 SOC 峰值 %.0f kWh" % soc_q2[i_solstice].max(),
       (float(hrs[iz]), float(soc_q2[i_solstice][iz]))),
     ("SOC 峰值 %.0f kWh" % soc_q2[i_solstice].max(),
       (float(hrs[iz]), float(soc_q2[i_solstice][iz]))),
     ("峰值 %.0f kWh" % soc_q2[i_solstice].max(), (float(hrs[iz]), float(soc_q2[i_solstice][iz])))],
])
fig.savefig(os.path.join(OUT, "fig14_seasonal_dispatch.png"), bbox_inches="tight")
plt.close(fig)
print("  fig14_seasonal_dispatch.png")

soc_lo, soc_hi = soc_q2.min(axis=1), soc_q2.max(axis=1)
charge_kwh, dischg_kwh = q2["c"].sum(axis=1), q2["d"].sum(axis=1)
fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True)

a = axes[0]
a.fill_between(date_num, soc_lo, soc_hi, color=SEM["soc"], alpha=0.26, lw=0,
                label="每日 SOC 区间（最低~最高）")
a.plot(date_num, soc_hi, color=SEM["soc"], lw=1.1, label="每日峰值 SOC")
a.axhline(L.SOC_MAX, color=SEM["emerg"], ls="--", lw=1.0, label="运行上界 10 800 kWh")
a.axhline(L.SOC_MIN, color=SEM["emerg"], ls="--", lw=1.0, label="运行下界 1 200 kWh")
a.set_ylim(0, 12300)
a.set_ylabel("储电量 (kWh)")
a.set_title("（a）全年 SOC 运行区间：日均峰值 %.0f kWh，%.0f%% 的天数触及上界（日末必回下界）"
            % (soc_hi.mean(), (soc_hi > L.SOC_MAX - 1).mean() * 100), fontsize=11)
a.legend(ncol=4, loc="upper left")
a.xaxis.set_major_locator(mdates.MonthLocator())
a.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
despine(a)

b = axes[1]
b.plot(date_num, charge_kwh, color=SEM["charge"], lw=1.1, label="日充电量")
b.plot(date_num, dischg_kwh, color=SEM["dischg"], lw=1.1, label="日放电量")
b.set_ylim(0, float(max(charge_kwh.max(), dischg_kwh.max())) * 1.18)
b.set_ylabel("电量 (kWh)")
b.set_xlabel("日期")
b.set_title("（b）每日充放电量：循环强度与效率校验", fontsize=11)
b.legend(ncol=2, loc="upper left")
b.xaxis.set_major_locator(mdates.MonthLocator())
b.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
despine(b)
fig.tight_layout(rect=[0, 0, 1, 0.96])

k_mid = int(len(date_num) * 0.55)
ok &= place_all("图15a", fig, a,
                [("line", date_num, soc_hi), ("hline", L.SOC_MAX), ("hline", L.SOC_MIN),
                  ("texts",)], [
    [("每日均回落至下界 1 200 kWh", (date_num[k_mid], float(soc_lo[k_mid]))),
     ("每日回到下界 1 200 kWh", (date_num[k_mid], float(soc_lo[k_mid])))],
])
k_dischg = int(np.argmax(dischg_kwh))
ok &= place_all("图15b", fig, b,
                 [("line", date_num, charge_kwh), ("line", date_num, dischg_kwh), ("texts",)], [
    [("放电/充电 = %.2f，与电池单程效率一致" % (dischg_kwh.sum() / charge_kwh.sum()),
       (date_num[k_dischg], float(dischg_kwh[k_dischg]))),
     ("放电/充电 = %.2f" % (dischg_kwh.sum() / charge_kwh.sum()),
       (date_num[k_dischg], float(dischg_kwh[k_dischg])))],
])
fig.savefig(os.path.join(OUT, "fig15_annual_soc.png"), bbox_inches="tight")
plt.close(fig)
print("  fig15_annual_soc.png")

price_rt, price_mean = q4["price_rt"], q4["price_mean"]
xs = np.arange(1, L.N_T + 1)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
a = axes[0]
a.plot(xs, price_mean, color=SEM["price"], lw=2.2, label="附件1 均值电价（问2 / 问3 使用）")
for dd in SAMPLES:
    a.plot(xs, price_rt[dd], color=DEEP, lw=0.8, alpha=0.75)
a.plot([], [], color=DEEP, lw=0.9, label="附件4 实时电价（问4 使用，抽样 6 天）")
a.set_xlabel("时段（10 分钟）")
a.set_ylabel("电价 (元/kWh)")
a.set_ylim(0, float(max(price_mean.max(), price_rt[SAMPLES].max())) * 1.10)
a.set_title("（a）均值电价与实时电价的日内形态", fontsize=11)
a.legend(loc="upper left")
despine(a)

b = axes[1]
bins = np.linspace(0.0, 1.85, 40)
cnt_mean, _, pat_mean = b.hist(price_mean, bins=bins, color=SEM["price"], alpha=0.92,
                               edgecolor="white", lw=0.4, label="附件1 均值电价")
cnt_rt, _, _ = b.hist(price_rt.ravel(), bins=bins, histtype="step", lw=1.7,
                      color=DEEP, label="附件4 实时电价")
b.set_xlabel("电价 (元/kWh)")
b.set_ylabel("频数")
b.set_ylim(0, float(max(cnt_mean.max(), cnt_rt.max())) * 1.30)
b.set_title("（b）电价分布对比：实时电价被显著展宽", fontsize=11)
b.legend(loc="upper right")
despine(b)
fig.tight_layout(rect=[0, 0, 1, 0.94])

i_low = int(np.argmin(price_mean))
ok &= place_all("图16a", fig, a,
                [("line", xs, price_mean)] + [("line", xs,
    price_rt[dd]) for dd in SAMPLES] + [("texts",)], [
    [("谷段 %.2f 元/kWh（日内极差 %.2f）" % (price_mean[i_low], price_mean.max() - price_mean.min()),
      (float(xs[i_low]), float(price_mean[i_low]))),
     ("谷段低至 %.2f 元/kWh" % price_mean[i_low], (float(xs[i_low]), float(price_mean[i_low]))),
     ("谷 %.2f 元/kWh" % price_mean[i_low], (float(xs[i_low]), float(price_mean[i_low])))],
])
i_mode = int(np.argmax(cnt_rt))
ok &= place_all("图16b", fig, b, [("bars", pat_mean), ("texts",)], [
    [("实时电价极差 %.2f 元/kWh（均值的 %.1f 倍）"
      % (price_rt.max() - price_rt.min(),
          (price_rt.max() - price_rt.min()) / (price_mean.max() - price_mean.min())),
      (float(0.5 * (bins[i_mode] + bins[i_mode + 1])), float(cnt_rt[i_mode]))),
     ("实时电价极差 %.2f 元/kWh" % (price_rt.max() - price_rt.min()),
      (float(0.5 * (bins[i_mode] + bins[i_mode + 1])), float(cnt_rt[i_mode])))],
])
fig.savefig(os.path.join(OUT, "fig16_q4_price_shape.png"), bbox_inches="tight")
plt.close(fig)
print("  fig16_q4_price_shape.png")

plan_a = q2["g_all"][31:].mean(axis=0)     # 方案A：均值电价做计划
plan_b = q4["g42"].mean(axis=0)            # 方案B：实时电价做计划
cost_a, cost_b, regret = float(q4["C_det_in_rt"]), float(q4["C42"]), float(q4["regret"])
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
a = axes[0]
bc = a.bar([0, 1], [cost_a / 1e4, cost_b / 1e4], 0.55, color=[LIGHT, SEM["buy"]])
a.set_xticks([0, 1])
a.set_xticklabels(["方案A\n均值电价做计划\n按实时电价结算",
                   "方案B\n实时电价做计划\n（问题4-2）"], fontsize=9.5)
for i, v in enumerate([cost_a, cost_b]):
    a.annotate("%.1f" % (v / 1e4), (i, v / 1e4), ha="center", va="bottom", fontsize=9)
a.set_ylim(0, max(cost_a, cost_b) / 1e4 * 1.30)
a.set_ylabel("报告期总费用 (万元)")
a.set_title("（a）两种规划口径的费用对比", fontsize=11)
despine(a)

b = axes[1]
b.plot(xs, price_mean, color=SEM["price"], lw=1.6, label="均值电价")
b.set_xlabel("时段（10 分钟）")
b.set_ylabel("电价 (元/kWh)")
b.set_ylim(0, float(price_mean.max()) * 1.25)
b2 = b.twinx()
b2.plot(xs, plan_a, color=LIGHT, lw=1.8, label="方案A 平均计划购电量")
b2.plot(xs, plan_b, color=SEM["buy"], lw=1.8, ls="--", label="方案B 平均计划购电量")
b2.set_ylabel("平均计划购电量 (kWh/段)")
b2.set_ylim(0, float(max(plan_a.max(), plan_b.max())) * 1.25)
b2.grid(False)
b2.spines["top"].set_visible(False)
b2.spines["left"].set_visible(False)
h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
b.legend(h1 + h2, l1 + l2, loc="upper left")
b.set_title("（b）两种口径的购电时序几乎重合，差异集中在尾段", fontsize=11)
despine(b)
fig.tight_layout(rect=[0, 0, 1, 0.94])

ok &= place_all("图17a", fig, a, [("bars", bc), ("texts",)], [
    [("后悔值 %.2f 万元：方案A 反而更省" % (regret / 1e4), (1.28, cost_b / 1e4)),
     ("后悔值 %.2f 万元" % (regret / 1e4), (1.28, cost_b / 1e4))],
])
i_peak_a = int(np.argmax(plan_a))
i_diff = int(np.argmax(np.abs(plan_b - plan_a)))
time_label = "%02d:%02d" % (int(i_diff * L.DT), int(round((i_diff * L.DT % 1) * 60)))
# 从轴曲线须传 ax=b2；主轴电价曲线传 ax=b
ok &= place_all("图17b", fig, b2,
                [("line", xs, price_mean, b), ("line", xs, plan_a, b2), ("line", xs, plan_b, b2),
                 ("legend_of", b), ("texts",)], [
    [("%s 谷段 B %.0f vs A %.0f kWh/段" % (time_label, plan_b[i_diff], plan_a[i_diff]),
      (float(xs[i_diff]), float(plan_b[i_diff]))),
     ("%s B %.0f / A %.0f kWh/段" % (time_label, plan_b[i_diff], plan_a[i_diff]),
      (float(xs[i_diff]), float(plan_b[i_diff]))),
     ("B 峰值 %.0f kWh/段（A %.0f）" % (plan_b[i_diff], plan_a[i_diff]),
       (float(xs[i_diff]), float(plan_b[i_diff])))],
])
fig.savefig(os.path.join(OUT, "fig17_q4_regret.png"), bbox_inches="tight")
plt.close(fig)
print("  fig17_q4_regret.png")

names = ["问1\n(单日折算)", "问2\n(两阶段SP)", "问3\n(预报+调整)", "问4-2\n(波动电价)", "问4-3\n(波动+调整)"]
cost_vals = [cost1 * 334, cost2, cost3, cost42, cost43]
cost_cols = [SEM["none"], LIGHT, SEM["buy"], LIGHT, SEM["buy"]]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.9))
a = axes[0]
bc = a.bar(names, np.array(cost_vals) / 1e4, 0.6, color=cost_cols)
for i, v in enumerate(cost_vals):
    a.annotate("%.0f" % (v / 1e4), (i, v / 1e4), ha="center", va="bottom", fontsize=9)
a.set_ylim(0, max(cost_vals) / 1e4 * 1.32)
a.set_ylabel("费用 (万元)")
a.set_title("（a）报告期费用对比（统一折算 334 天口径）", fontsize=11)
a.legend(handles=[Patch(color=SEM["none"], label="单日折算口径"),
                  Patch(color=LIGHT, label="计划口径（含紧急购电）"),
                  Patch(color=SEM["buy"], label="含实时调整")],
         loc="upper left", fontsize=8.5)
despine(a)

b = axes[1]
cmp_names = ["问1\n无储能", "问1\n有储能", "问2", "问3", "问4-2", "问4-3"]
cmp_vals = [base1 * 334, cost1 * 334, cost2, cost3, cost42, cost43]
bc2 = b.bar(cmp_names, np.array(cmp_vals) / 1e4, 0.58,
            color=[SEM["none"], SEM["buy"], SEM["buy"], SEM["buy"], SEM["buy"], SEM["buy"]])
for i, v in enumerate(cmp_vals):
    b.annotate("%.0f" % (v / 1e4), (i, v / 1e4), ha="center", va="bottom", fontsize=8.5)
b.set_ylim(0, max(cmp_vals) / 1e4 * 1.24)
b.set_ylabel("费用 (万元)")
b.set_title("（b）储能价值与信息代价（灰色为无储能基准）", fontsize=11)
despine(b)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.text(0.012, 0.012,
         "注：问1 为典型日最优调度按 334 天折算；问2—问4 为报告期实数结算，口径一致。",
         ha="left", va="bottom", fontsize=8.6, color=SEM["ink"])

ok &= place_all("图18a", fig, a, [("bars", bc), ("texts",)], [
    [("实时电价口径比均值电价口径多花 %.0f 万元（+%.1f%%）"
      % ((cost42 - cost2) / 1e4, (cost42 - cost2) / cost2 * 100), (3.30, cost42 / 1e4)),
     ("实时电价口径多花 %.0f 万元" % ((cost42 - cost2) / 1e4), (3.30, cost42 / 1e4))],
])
ok &= place_all("图18b", fig, b, [("bars", bc2), ("texts",)], [
    [("储能使费用由 %.0f 降至 %.0f 万元（−%.1f%%）"
      % (base1 * 334 / 1e4, cost1 * 334 / 1e4, (base1 - cost1) / base1 * 100),
    (1.30, cost1 * 334 / 1e4)),
     ("储能节省 %.1f%%" % ((base1 - cost1) / base1 * 100), (1.30, cost1 * 334 / 1e4))],
])
fig.savefig(os.path.join(OUT, "fig18_four_questions_summary.png"), bbox_inches="tight")
plt.close(fig)
print("  fig18_four_questions_summary.png")

plan_kwh, adj_kwh = q3["g_plan"], q3["g_adj"]
under_kwh = np.maximum(plan_kwh - adj_kwh, 0.0)
over_kwh = np.maximum(adj_kwh - plan_kwh, 0.0)
pv_q3, fc_q3 = q3["pv"], q3["fc10"]
err = (fc_q3 - pv_q3[:, None, :]).mean(axis=1)
day_only = (pv_q3 > 1.0) | (fc_q3.mean(axis=1) > 1.0)
err_day = err[day_only]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
a = axes[0]
cnt_e, _, _ = a.hist(err_day, bins=70, color=SEM["buy"], alpha=0.9, edgecolor="white", lw=0.3)
a.axvline(0, color=SEM["ref"], ls="--", lw=1.4, label="零误差线")
a.axvline(err_day.mean(), color=SEM["price"], ls="-.", lw=1.5,
          label="均值 %.1f kW" % err_day.mean())
a.set_xlabel("预报误差 (kW)　[仅白天时段]")
a.set_ylabel("频数")
a.set_ylim(0, float(cnt_e.max()) * 1.28)
a.set_title("（a）白天时段预报误差分布\n(均值 %.1f kW, MAE %.1f kW)"
            % (err_day.mean(), np.abs(err_day).mean()), fontsize=11)
a.legend(loc="upper left")
despine(a)

b = axes[1]
edges2 = np.linspace(0.0, float(max(under_kwh.max(), over_kwh.max())), 45)
cnt_u, _, pat_u = b.hist(under_kwh[under_kwh > 1e-6], bins=edges2, color=SEM["emerg"], alpha=0.85,
                         edgecolor="white", lw=0.3, label="少买（违约）")
cnt_v, _, pat_v = b.hist(over_kwh[over_kwh > 1e-6], bins=edges2, color=SEM["adj"], alpha=0.9,
                         edgecolor="white", lw=0.3, label="多买（超额）")
b.set_xlabel("偏差量 (kWh/10min)")
b.set_ylabel("频数")
b.set_ylim(0, float(max(cnt_u.max(), cnt_v.max())) * 1.30)
b.set_title("（b）计划-调整偏差分布", fontsize=11)
b.legend(loc="upper right")
despine(b)

c = axes[2]
i_winter = day_index("2025-12-21")
c.fill_between(hrs, plan_kwh[i_winter], adj_kwh[i_winter],
                where=(adj_kwh[i_winter] > plan_kwh[i_winter]), color=SEM["adj"], alpha=0.25)
c.fill_between(hrs, plan_kwh[i_winter], adj_kwh[i_winter],
                where=(adj_kwh[i_winter] < plan_kwh[i_winter]), color=SEM["emerg"], alpha=0.25)
c.plot(hrs, plan_kwh[i_winter], color=SEM["buy"], lw=1.8, label="计划购电量")
c.plot(hrs, adj_kwh[i_winter], color=SEM["adj"], lw=1.5, ls="--", label="调整购电量")
c.set_xlabel("时刻 (小时)")
c.set_ylabel("购电量 (kWh/10min)")
c.set_ylim(0, float(plan_kwh[i_winter].max()) * 1.25)
c.set_title("（c）2025-12-21 计划 %.1f MWh vs 调整 %.1f MWh"
            % (plan_kwh[i_winter].sum() / 1000, adj_kwh[i_winter].sum() / 1000), fontsize=11)
c.legend(loc="upper left")
despine(c)
fig.tight_layout(rect=[0, 0, 1, 0.93])

ok &= place_all("图19a", fig, a,
                [("line", np.array([err_day.mean(), err_day.mean()]),
                  np.array([0.0, float(cnt_e.max())])), ("texts",)], [
    [("预报整体偏低 %.0f kW（MAE %.0f kW）" % (-err_day.mean(), np.abs(err_day).mean()),
      (float(err_day.mean()), float(cnt_e.max()) * 0.75)),
     ("均值偏低 %.0f kW" % (-err_day.mean()), (float(err_day.mean()), float(cnt_e.max()) * 0.75))],
])
i_under = int(np.argmax(cnt_u))
ok &= place_all("图19b", fig, b, [("bars", pat_u), ("bars", pat_v), ("texts",)], [
    [("少买合计 %.0f MWh（违约费 %.1f 万元）" % (under_kwh.sum() / 1000, float(q3["C_def"]) / 1e4),
      (float(0.5 * (edges2[i_under] + edges2[i_under + 1])), float(cnt_u[i_under]))),
     ("少买 %.0f MWh" % (under_kwh.sum() / 1000),
      (float(0.5 * (edges2[i_under] + edges2[i_under + 1])), float(cnt_u[i_under])))],
])
fig.savefig(os.path.join(OUT, "fig19_q3_forecast_error.png"), bbox_inches="tight")
plt.close(fig)
print("  fig19_q3_forecast_error.png")

voi_by_label = {str(k): float(val) for k, val in zip(q3["voi_labels"], q3["voi_costs"])}


def _pick(pre):
    for k, val in voi_by_label.items():
        if k.startswith(pre):
            return val
    return float("nan")


cost_plan_a, cost_plan_b, cost_plan_c = _pick("A"), _pick("B"), _pick("C 三")
voi_gap = cost_plan_b - cost_plan_c
voi_marginal = list(q3["voi_info"])
fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
a = axes[0]
bc = a.bar(range(3), np.array([cost_plan_a, cost_plan_b, cost_plan_c]) / 1e4, 0.6,
           color=[SEM["none"], SEM["emerg"], SEM["charge"]])
a.set_xticks(range(3))
a.set_xticklabels(["A 仅 0:00 决策\n（不改签，基准）", "B 三时刻改签\n（预报不更新）",
                   "C 三时刻改签\n（当期预报）"], fontsize=9.5)
for i, vv in enumerate([cost_plan_a, cost_plan_b, cost_plan_c]):
    a.annotate("%.1f" % (vv / 1e4), (i, vv / 1e4), ha="center", va="bottom", fontsize=9)
a.set_ylim(0, max(cost_plan_a, cost_plan_b, cost_plan_c) / 1e4 * 1.34)
a.set_ylabel("报告期总费用 (万元)")
a.set_title("（a）改签机会为负价值，信息更新为正价值", fontsize=11)
despine(a)

b = axes[1]
bc2 = b.bar(range(3), np.array(voi_marginal) / 1e4, 0.55, color=SEM["charge"])
b.axhline(0, color=SEM["ref"], lw=1.1)
b.set_xticks(range(3))
b.set_xticklabels(["6:00 发布", "12:00 发布", "18:00 发布"], fontsize=9.5)
for i, vv in enumerate(voi_marginal):
    b.annotate("%.2f" % (vv / 1e4), (i, vv / 1e4), ha="center", va="bottom", fontsize=9)
b.set_ylim(0, max(voi_marginal) / 1e4 * 1.36)
b.set_ylabel("边际信息价值 (万元)")
b.set_title("（b）各发布时刻的边际信息价值（均非负）", fontsize=11)
despine(b)
fig.tight_layout(rect=[0, 0, 1, 0.93])

ok &= place_all("图20a", fig, a, [("bars", bc), ("texts",)], [
    [("B 比 A 多花 %.1f 万元；C 比 B 省 %.1f 万元" % ((cost_plan_b - cost_plan_a) / 1e4, voi_gap / 1e4),
       (1.28, cost_plan_b / 1e4)),
     ("改签 −%.1f / 信息 +%.1f 万元" % ((cost_plan_b - cost_plan_a) / 1e4, voi_gap / 1e4),
       (1.28, cost_plan_b / 1e4))],
])
ok &= place_all("图20b", fig, b, [("bars", bc2), ("texts",)], [
    [("18:00 边际价值仅 %.1f 元 ≈ 0（傍晚光伏为 0）" % voi_marginal[2], (2.28, float(voi_marginal[2] / 1e4))),
     ("18:00 仅 %.1f 元 ≈ 0" % voi_marginal[2], (2.28, float(voi_marginal[2] / 1e4))),
     ("≈ 0（%.1f 元）" % voi_marginal[2], (2.28, float(voi_marginal[2] / 1e4)))],
])
fig.savefig(os.path.join(OUT, "fig20_voi_decomposition.png"), bbox_inches="tight")
plt.close(fig)
print("  fig20_voi_decomposition.png")

print("全部标注避让成功" if ok else "存在未能完全避让的标注")
print("已生成：图12 / 图13 / 图14 / 图15 / 图16 / 图17 / 图18 / 图19 / 图20")
