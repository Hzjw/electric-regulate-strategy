"""核实附件1 与附件2/4 的数量关系，并重做 LP 尺度不变性演示"""
import os, sys, io
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT = os.path.join(ROOT, "data")
N_T, N_DAY = 144, 365

a1 = pd.read_excel(os.path.join(ATT, "附件1.xlsx"))
price1 = pd.to_numeric(a1["电价"]).values
load1 = pd.to_numeric(a1["小区负载"]).values
pv1 = pd.to_numeric(a1["光伏发电预测功率"]).values

load2 = (pd.read_excel(os.path.join(ATT, "附件2.xlsx"), sheet_name="小区负载")
         .iloc[:, 1:1 + N_T].apply(pd.to_numeric).values)
pv2 = (pd.read_excel(os.path.join(ATT, "附件2.xlsx"), sheet_name="光伏发电实际功率")
       .iloc[:, 1:1 + N_T].apply(pd.to_numeric).values)
price4 = (pd.read_excel(os.path.join(ATT, "附件4.xlsx"))
          .iloc[:, 1:1 + N_T].apply(pd.to_numeric).values)

print("附件1 与 附件2 逐时段均值日的对比")
for name, day, year in [("小区负载", load1, load2), ("光伏功率", pv1, pv2)]:
    mean_day = year.mean(axis=0)
    diff = np.abs(day - mean_day)
    rel = diff / np.maximum(np.abs(mean_day), 1e-9)
    print(name, "均值差 %.3e" % abs(day.mean() - mean_day.mean()),
          "最大绝对差 %.6f" % diff.max(),
          "最大相对差 %.6f%%" % (rel.max() * 100),
          "完全相同时段 %d/144" % int((diff < 1e-6).sum()))

avg_price = price4.mean(axis=0)
print("电价", "均值差 %.3e" % abs(price1.mean() - avg_price.mean()),
      "最大绝对差 %.6f" % np.abs(price1 - avg_price).max(),
      "相关系数 %.6f" % np.corrcoef(price1, avg_price)[0, 1])

print("负载", "附件1 %.4f/%.4f" % (load1.mean(), load1.std()),
      "附件2 全年 %.4f/%.4f" % (load2.mean(), load2.std()))
print("光伏", "附件1 %.4f/%.4f" % (pv1.mean(), pv1.std()),
      "附件2 全年 %.4f/%.4f" % (pv2.mean(), pv2.std()))
print("电价", "附件1 %.4f/%.4f" % (price1.mean(), price1.std()),
      "附件4 全年 %.4f/%.4f" % (price4.mean(), price4.std()))

from scipy.optimize import linprog

# min c'x, s.t. x1+x2>=10, 0<=x<=100
cost0 = np.array([0.4, 1.3])
A0 = np.array([[-1.0, -1.0]]); rhs0 = np.array([-10.0])
bnd = [(0, 100)] * 2
res0 = linprog(cost0, A_ub=A0, b_ub=rhs0, bounds=bnd)
print("原始尺度", "x=%s" % np.round(res0.x, 6).tolist(), "目标 %.6f" % res0.fun)

# 尺度变换 x = scale*y 后重解
scale = np.array([1000.0, 0.001])
cost1 = cost0 * scale
A1 = A0 * scale
res1 = linprog(cost1, A_ub=A1, b_ub=rhs0, bounds=[(0, 100/1000), (0, 100/0.001)])
x_back = res1.x * scale
print("缩放1000x", "还原 x=%s" % np.round(x_back, 6).tolist(),
      "目标 %.6f" % res1.fun, "目标值差 %.3e" % abs(res0.fun - res1.fun))

dates = pd.to_datetime(pd.read_excel(os.path.join(ATT, "附件4.xlsx")).iloc[:, 0])
low_idx = np.where(price4 < 0.05)
seg = low_idx[1]
print("电价<0.05 的点", len(low_idx[0]), "个",
      "日期", sorted(set(dates.iloc[low_idx[0]].dt.strftime("%Y-%m-%d"))))
print("时段区间", ["%02d:%02d-%02d:%02d" % (((s * 10) // 60), (s * 10) % 60,
                                             ((s + 1) * 10) // 60, ((s + 1) * 10) % 60)
                  for s in seg])

fc_raw = pd.read_excel(os.path.join(ATT, "附件3.xlsx"), header=0)
fc_raw.columns = ["日期", "预报时刻"] + ["h%d" % k for k in range(1, 25)]
fc_raw["日期"] = pd.to_datetime(fc_raw["日期"].ffill().astype(str), format="mixed")
fc_grid = np.full((N_DAY, 4, 24), np.nan)
day_index = pd.DatetimeIndex(sorted(fc_raw["日期"].unique()))
day_pos = {d: i for i, d in enumerate(day_index)}
fc_times = ["0:00", "6:00", "12:00", "18:00"]
for row in range(len(fc_raw)):
    d = fc_raw["日期"].iloc[row]; t = str(fc_raw["预报时刻"].iloc[row]).strip()
    if t in fc_times:
        fc_grid[day_pos[d], fc_times.index(t), :] = pd.to_numeric(fc_raw.iloc[row, 2:26]).values
pv_hour = pv2.reshape(N_DAY, 24, 6).mean(axis=2)
base_h = [0, 6, 12, 18]
for k in range(4):
    errs = []
    for d in range(N_DAY):
        for h in range(24):
            # 时段为区间终点，j = h+1-base
            j = (h + 1 - base_h[k]) % 24
            if j == 0: j = 24
            errs.append(fc_grid[d, k, j - 1] - pv_hour[d, h])
    err = np.array(errs)
    print(fc_times[k], "发布", "平均误差 %.1f" % err.mean(), "中位 %.1f" % np.median(err),
          "标准差 %.1f" % err.std(), "平均绝对误差 %.1f" % np.abs(err).mean())
print("误差为 预报 - 实际，负值表示预报偏低")
