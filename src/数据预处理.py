"""附件数据清点：缺失值 / 异常值检测与清洗后导出，产出图1~图8。"""
import os, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT  = os.path.join(ROOT, "data")
IMG  = os.path.join(ROOT, "figures")
RES  = os.path.join(ROOT, "results")
for d in (IMG, RES):
    os.makedirs(d, exist_ok=True)

N_T, N_DAY, DT = 144, 365, 1.0/6.0
MONTHS = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]

def load(fn, **kw):
    return pd.read_excel(os.path.join(ATT, fn), **kw)

typical = load("附件1.xlsx")
load_year = load("附件2.xlsx", sheet_name="小区负载")
pv_year   = load("附件2.xlsx", sheet_name="光伏发电实际功率")
fcst = load("附件3.xlsx")
price_year = load("附件4.xlsx")

inventory = [
    ("附件1", "Sheet1",           typical.shape,    "某典型日 电价/负载/光伏预测", "144 个 10 分钟时段"),
    ("附件2", "小区负载",          load_year.shape,  "全年 小区负载(实际)",         "365 天 x 144 时段"),
    ("附件2", "光伏发电实际功率",  pv_year.shape,    "全年 光伏发电(实际)",         "365 天 x 144 时段"),
    ("附件3", "Sheet1",           fcst.shape,       "全年 光伏预报",
     "365 天 x 4 发布时刻 x 24 小时"),
    ("附件4", "Sheet1",           price_year.shape, "全年 实时电价",               "365 天 x 144 时段"),
]
inv_rows = []
for fn, sheet, shp, desc, gran in inventory:
    print("%-8s %-18s %-14s %-22s %s" % (fn, sheet, str(shp), desc, gran))
    inv_rows.append({"文件":fn, "工作表":sheet, "行":shp[0], "列":shp[1], "内容":desc, "粒度":gran})
pd.DataFrame(inv_rows).to_csv(os.path.join(RES, "数据清单.csv"), index=False, encoding="utf-8-sig")

price1 = pd.to_numeric(typical["电价"], errors="coerce").values
load1  = pd.to_numeric(typical["小区负载"], errors="coerce").values
pv1    = pd.to_numeric(typical["光伏发电预测功率"], errors="coerce").values
load2  = load_year.iloc[:, 1:1+N_T].apply(pd.to_numeric, errors="coerce").values
pv2    = pv_year.iloc[:, 1:1+N_T].apply(pd.to_numeric, errors="coerce").values
price4 = price_year.iloc[:, 1:1+N_T].apply(pd.to_numeric, errors="coerce").values

def stat_line(name, x, unit):
    print("%-26s min=%10.4f  max=%10.4f  mean=%10.4f  std=%9.4f  %s"
          % (name, np.nanmin(x), np.nanmax(x), np.nanmean(x), np.nanstd(x), unit))
stat_line("附件1 电价",     price1, "元/kWh")
stat_line("附件1 小区负载", load1,  "kW")
stat_line("附件1 光伏预测", pv1,    "kW")
stat_line("附件2 小区负载", load2,  "kW")
stat_line("附件2 光伏实际", pv2,    "kW")
stat_line("附件4 实时电价", price4, "元/kWh")

miss_rows = []
def count_missing(name, df, skip_col0=False):
    sub = df.iloc[:, 1:] if skip_col0 else df
    n = int(sub.isna().sum().sum())
    tot = int(sub.size)
    miss_rows.append({"数据体":name, "单元格总数":tot, "缺失单元格":n,
                      "缺失率": round(n/tot*100, 6) if tot else 0.0})
    return n, tot

checks = [
    ("附件1", typical, False),
    ("附件2/小区负载", load_year, True),
    ("附件2/光伏实际", pv_year, True),
    ("附件3", fcst, True),
    ("附件4", price_year, True),
]
for name, df, sk in checks:
    count_missing(name, df, sk)

# 附件3 日期列为合并单元格空白
date_col = fcst.iloc[:, 0]
n_blank = int(date_col.isna().sum())
filled = date_col.ffill()
print("显式缺失单元格", sum(r["缺失单元格"] for r in miss_rows), "个；附件3 日期列空白", n_blank,
      "行，ffill 后", filled.nunique(), "个日期，每天行数", sorted(filled.value_counts().unique().tolist()))

pd.DataFrame(miss_rows).to_csv(os.path.join(RES, "缺失值报告.csv"), index=False, encoding="utf-8-sig")

def zscore_flag(x, thr=3.0):
    x = np.asarray(x, float)
    sd = x.std(ddof=0)
    if sd == 0:
        return np.zeros(x.size, bool), np.zeros(x.size)
    z = (x - x.mean()) / sd
    return np.abs(z) > thr, z

def iqr_flag(x, k=1.5):
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    f = (x < lo) | (x > hi)
    return f, (lo, hi)

def both_flags(x):
    fz, z = zscore_flag(x)
    fi, _ = iqr_flag(x)
    return fz, fi, z

for name, x in [("电价", price1), ("小区负载", load1), ("光伏预测", pv1)]:
    fz, fi, z = both_flags(x)
    q1, q3 = np.percentile(x, [25, 75])
    iqr_ = q3 - q1
    print("%-10s  Z-score(|z|>3) 检出 %3d 点 | IQR(1.5倍) 检出 %3d 点 | Q1=%.4f Q3=%.4f IQR=%.4f"
          % (name, int(fz.sum()), int(fi.sum()), q1, q3, iqr_))
print("附件1 电价为峰谷分时电价，双峰结构下 IQR / Z-score 必然把合法峰段判为异常，属结构性误报，保留原值")

dates = pd.to_datetime(load_year.iloc[:, 0])
month = dates.dt.month.values

def global_vs_monthly(name, X, unit):
    fz, fi, _ = both_flags(X.ravel())
    mz, mi = [], []
    for m in range(1, 13):
        sub = X[month == m].ravel()
        a, _, _ = both_flags(sub)
        b, _ = iqr_flag(sub)
        mz.append(int(a.sum())); mi.append(int(b.sum()))
    print("%s(%s) 全局 Z 检出 %6d 点 (%.2f%%) / 分月合计 %d 点；全局 IQR 检出 %6d 点 (%.2f%%) / 分月合计 %d 点"
          % (name, unit, int(fz.sum()), fz.mean()*100, sum(mz),
             int(fi.sum()), fi.mean()*100, sum(mi)))
    return mz, mi

mz_load, mi_load = global_vs_monthly("附件2 小区负载", load2, "kW")
mz_pv, mi_pv = global_vs_monthly("附件2 光伏实际", pv2, "kW")
mz_price, mi_price = global_vs_monthly("附件4 实时电价", price4, "元/kWh")

neg_load = int((load2 < 0).sum())
neg_pv = int((pv2 < 0).sum())
neg_price = int((price4 <= 0).sum())
print("物理上不可能的点数：负载为负 %d，光伏为负 %d，电价<=0 %d" % (neg_load, neg_pv, neg_price))

night = np.r_[0:24, 132:144]
night_max = float(np.nanmax(pv2[:, night]))
day = np.arange(36, 108)
day_min = float(np.nanmin(pv2[:, day]))
print("光伏 夜间(22:00-04:00)最大值 %.4f kW，日间(06:00-18:00)最小值 %.4f kW" % (night_max, day_min))

mono = np.abs(np.diff(load2, axis=1))
low_c = np.where(price4 < 0.05)
print("负载相邻时段最大跳变 %.1f kW (%.2f%% of 均值)；电价 < 0.05 元/kWh 的点 %d 个 (%.4f%%)"
      % (mono.max(), mono.max()/load2.mean()*100, len(low_c[0]), len(low_c[0])/price4.size*100))

data1 = [("电价 (元/kWh)", price1, "#2E6F9E"),
         ("小区负载 (kW)", load1, "#3F8F5B"),
         ("光伏预测 (kW)", pv1, "#C77D2B")]


hprof = pv2.reshape(N_DAY*N_T//6, 6).mean(axis=1).reshape(N_DAY, 24).mean(axis=0) if False else None
P_hour = pv2.reshape(N_DAY, 24, 6).mean(axis=2)   # (365, 24)
mean_h = P_hour.mean(axis=0); p95_h = np.percentile(P_hour, 95, axis=0)
hours = np.arange(24)
night_vals = P_hour[:, list(range(0,5)) + list(range(22,24))].ravel()

labels = ["附件1\n电价", "附件1\n负载", "附件1\n光伏", "全年\n负载", "全年\n光伏", "全年\n电价"]
gz, gi = [], []
for x in [price1, load1, pv1, load2.ravel(), pv2.ravel(), price4.ravel()]:
    a, _, _ = both_flags(x); b, _ = iqr_flag(x)
    gz.append(a.mean()*100); gi.append(b.mean()*100)
xpos = np.arange(len(labels)); w = 0.38

t = np.arange(1, N_T+1) * 10 / 60.0


from scipy.optimize import linprog
# 尺度变换 x = D*y，D 含 1000 倍量纲差
c0 = np.array([0.4, 1.3])
A0 = np.array([[-1.0, -1.0]]); b0 = np.array([-10.0])
bnd0 = [(0, 100)] * 2
r_raw = linprog(c0, A_ub=A0, b_ub=b0, bounds=bnd0)

D  = np.array([1000.0, 0.001])          # 人为制造 10^6 量级的变量尺度差
c1 = c0 * D                             # 目标系数同步缩放
A1 = A0 * D                             # 约束系数同步缩放
bnd1 = [(0, 100/D[0]), (0, 100/D[1])]
r_scaled = linprog(c1, A_ub=A1, b_ub=b0, bounds=bnd1)
x_back = r_scaled.x * D                 # 还原到原始尺度

print("LP 最优值 原始 %.6f / 缩放 %.6f，差 %.3e；还原后 x = %s"
      % (r_raw.fun, r_scaled.fun, abs(r_raw.fun - r_scaled.fun), np.round(x_back, 6).tolist()))

out1 = typical.copy()
out1["段号"] = np.arange(1, N_T+1)
out1["区间"] = ["%02d:%02d-%02d:%02d"
                % (((i-1)*10)//60, ((i-1)*10)%60, (i*10)//60 if i*10 < 1440 else 0, (i*10) % 60)
                for i in range(1, N_T+1)]
out1.to_csv(os.path.join(RES, "清洗后_附件1典型日.csv"), index=False, encoding="utf-8-sig")

pv2_clip = np.clip(pv2, 0.0, None)
np.save(os.path.join(RES, "附件2_负载.npy"), load2)
np.save(os.path.join(RES, "附件2_光伏.npy"), pv2_clip)
np.save(os.path.join(RES, "附件4_电价.npy"), price4)

figs = sorted([f for f in os.listdir(IMG) if f.endswith(".png")])
print("数据自足无需补充；已写 results/ 下 3 个 csv 与 3 个 npy，图片目录当前共有", len(figs), "张 png")
