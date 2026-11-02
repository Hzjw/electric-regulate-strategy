"""循环成本敏感性扫描：c_cyc 对单日/报告期/问2 的影响"""
import os, sys, io, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdout.reconfigure(line_buffering=True)   # 行缓冲输出
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import lp_core as L
import stoch as ST

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

RUN_LAYER_C = True          # 层C 开关（约16分钟）
REP0 = 31                   # 报告期起点 2025-02-01
SL = slice(REP0, L.N_DAY)
E_US = L.SOC_MAX - L.SOC_MIN        # 可用容量 9600 kWh

D = L.load_all(verbose=False)
pr = D["typical"]["price"]
lk_t, pv_t = D["typical"]["load_kw"], D["typical"]["pv_kw"]
ld, pva = D["actual"]["load_kw"], D["actual"]["pv_kw"]
dates = D["actual"]["dates"]
DTYPE = ST.day_type(dates)

NS_DAY = float(pr @ np.maximum(lk_t * L.DT - pv_t * L.DT, 0.0))          # 单日无储能
NS_YEAR = float((pr[None, :] * np.maximum(ld[SL] * L.DT - pva[SL] * L.DT, 0)).sum())

PI_MIN, PI_MAX = float(pr.min()), float(pr.max())
C_ARB = PI_MAX - PI_MIN / (L.ETA_C * L.ETA_D)      # 推论2：纯套利失效的下界
C_STOP = PI_MAX                                    # 推论1：电池完全闲置的上界

print("循环成本敏感性扫描 —— 假设5「忽略电池寿命损耗」的量化")
print("附件1 电价", "%.4f" % PI_MIN, "~", "%.4f" % PI_MAX, "元/kWh",
      "峰谷比", "%.3f" % (PI_MAX / PI_MIN), "可用容量", "%.0f" % E_US, "kWh")
print("解析边界：纯套利失效", "%.4f" % C_ARB, "电池完全闲置", "%.4f" % C_STOP, "元/kWh")
print("无储能购电费：单日", "%.2f" % NS_DAY, "元，报告期", "%.2f" % NS_YEAR, "元")


def efc_of(dsum):
    """由放电总量(kWh)折算等效全循环次数"""
    return float(dsum) / L.ETA_D / E_US


t0 = time.time()
gridA = np.round(np.arange(0.0, C_STOP + 0.21, 0.01), 4)
rowsA = []
for cyc_cost in gridA:
    res = L.solve_day_lp(pr, lk_t, pv_t, soc_start=L.SOC_INIT, soc_end=L.SOC_INIT, c_cyc=cyc_cost)
    assert res["status"] == 0, "层A 求解失败 c_cyc=%.4f" % cyc_cost
    cycle_fee = cyc_cost * res["d"].sum()
    total = float(res["cost"])                  # res.fun = 购电费+循环成本
    rowsA.append(dict(c_cyc=cyc_cost, 购电费=total - cycle_fee, 循环成本=cycle_fee, 总成本=total,
                      EFC=efc_of(res["d"].sum()), 充电量=res["c"].sum(), 放电量=res["d"].sum(),
                      弃光量=res["s"].sum()))
dfA = pd.DataFrame(rowsA)
dfA["节省"] = NS_DAY - dfA["总成本"]
dfA["节省率%"] = dfA["节省"] / NS_DAY * 100
print("层A", len(dfA), "个网格点，用时", "%.1f" % (time.time() - t0), "s")


def cross_first(xs, ys, thr):
    """返回 ys 首次 <= thr 的 x（线性内插）"""
    for i in range(1, len(ys)):
        if ys[i] <= thr:
            x_prev, x_cur, y_prev, y_cur = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if y_prev == y_cur:
                return float(x_cur)
            return float(x_prev + (y_prev - thr) * (x_cur - x_prev) / (y_prev - y_cur))
    return float("nan")


sav0 = dfA["节省"].iloc[0]
c10 = cross_first(dfA["c_cyc"].values, dfA["节省"].values, 0.10 * sav0)
c50 = cross_first(dfA["c_cyc"].values, dfA["节省"].values, 0.50 * sav0)
# EFC 归零点二分定位
lo, hi = 0.0, C_STOP + 0.5
for _ in range(30):
    mid = 0.5 * (lo + hi)
    res = L.solve_day_lp(pr, lk_t, pv_t, soc_start=L.SOC_INIT, soc_end=L.SOC_INIT, c_cyc=mid)
    if efc_of(res["d"].sum()) < 1e-9:
        hi = mid
    else:
        lo = mid
c_stop_num = hi
print("层A c_cyc=0 基准：EFC", "%.4f" % dfA.EFC.iloc[0], "总成本", "%.2f" % dfA.总成本.iloc[0],
      "节省", "%.2f" % sav0, "元")
print("层A 阈值：节省降至 50%/10% 于", "%.4f" % c50, "/", "%.4f" % c10,
      "EFC 归零于", "%.6f" % c_stop_num, "对 pi_max", "%.4f" % PI_MAX,
      "偏差", "%.2e" % (abs(c_stop_num - PI_MAX) / PI_MAX))
dfA.to_csv("results/循环成本_单日.csv", index=False, encoding="utf-8-sig")

SURP = float(np.maximum(pv_t * L.DT - lk_t * L.DT, 0.0).sum())   # 全日光伏盈余


def _solve(cc):
    return L.solve_day_lp(pr, lk_t, pv_t, soc_start=L.SOC_INIT, soc_end=L.SOC_INIT, c_cyc=cc)


def _bisect(pred, lo, hi, it=50):
    """在 [lo,hi] 上二分 pred 的分界点"""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


# 三临界：停止套利/弃光/完全闲置
a_no_grid = _bisect(lambda v: _solve(v)["c"].sum() <= SURP + 1e-9, 0.85, 1.00)
a_zero_curt = _bisect(lambda v: _solve(v)["s"].sum() > 1e-9, 0.90, 1.20)
a_stop = _bisect(lambda v: _solve(v)["d"].sum() < 1e-9, 1.30, 1.50)
print("层A 三个临界：电网套利停止", "%.8f" % a_no_grid, "弃光重新出现", "%.8f" % a_zero_curt,
      "电池完全闲置", "%.8f" % a_stop)
print("层A 临界与解析值偏差", "%.2e" % abs(a_no_grid - (1.3482 - 0.3859 / 0.9)),
      "%.2e" % abs(a_zero_curt - 1.0956), "%.2e" % abs(a_stop - PI_MAX))

pm_2d = np.tile(pr, (L.N_DAY, 1))
t0 = time.time()
gridB = np.round(np.arange(0.0, C_STOP + 0.055, 0.05), 4)
rowsB = []
for i, cyc_cost in enumerate(gridB):
    res = L.solve_multiday_lp(pm_2d, ld, pva, soc_init=L.SOC_INIT, use_emg=True,
                              verbose=False, c_cyc=cyc_cost)
    assert res["status"] == 0, "层B 求解失败 c_cyc=%.4f" % cyc_cost
    g = res["g"][SL]
    cost_buy = float((pr[None, :] * g).sum())
    dsum = float(res["d"][SL].sum())
    cycle_fee = cyc_cost * dsum
    rowsB.append(dict(c_cyc=cyc_cost, 购电费=cost_buy, 循环成本=cycle_fee, 总成本=cost_buy + cycle_fee,
                      EFC=efc_of(dsum), 年放电量=dsum, 年充电量=float(res["c"][SL].sum()),
                      年弃光量=float(res["s"][SL].sum()), 紧急购电=float(res["e"][SL].sum())))
dfB = pd.DataFrame(rowsB)
dfB["节省"] = NS_YEAR - dfB["总成本"]
dfB["节省率%"] = dfB["节省"] / NS_YEAR * 100
dfB["日均EFC"] = dfB["EFC"] / (L.N_DAY - REP0)
dfB.to_csv("results/循环成本_年度.csv", index=False, encoding="utf-8-sig")   # 先落盘
assert abs(dfB.总成本.iloc[0] - 11796484.28) < 1.0, \
    "层B c_cyc=0 未复现完美信息基准 11,796,484.28，实得 %.2f" % dfB.总成本.iloc[0]
print("层B", len(dfB), "个网格点，用时", "%.1f" % (time.time() - t0), "s；c_cyc=0 复现完美信息基准",
      "%.2f" % dfB.总成本.iloc[0], "元")

savB0 = dfB["节省"].iloc[0]
b10 = cross_first(dfB["c_cyc"].values, dfB["节省"].values, 0.10 * savB0)
b50 = cross_first(dfB["c_cyc"].values, dfB["节省"].values, 0.50 * savB0)
SURP_Y = float(np.maximum(pva[SL] * L.DT - ld[SL] * L.DT, 0.0).sum())   # 报告期光伏盈余


def _myear(cc):
    return L.solve_multiday_lp(pm_2d, ld, pva, soc_init=L.SOC_INIT, use_emg=True,
                               verbose=False, c_cyc=cc)


def _bisect_y(pred, lo, hi, it=22):
    """二分定位 pred 由假变真的分界点，越界返回 nan"""
    if pred(lo):
        print("层B 二分下界", "%.3f" % lo, "处判定已为真，该指标在年度口径下无正临界值")
        return float("nan")
    if not pred(hi):
        print("层B 二分上界", "%.3f" % hi, "处判定仍为假，请扩大搜索区间")
        return float("nan")
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


b_no_grid = _bisect_y(lambda v: _myear(v)["c"][SL].sum() <= SURP_Y + 1e-6, 0.0, 1.40)
b_zero_curt = _bisect_y(lambda v: _myear(v)["s"][SL].sum() > 1e-9, 0.0, 1.40)
b_stop = _bisect_y(lambda v: efc_of(_myear(v)["d"][SL].sum()) < 1e-9, 1.30, 1.50)
print("层B 三个临界：电网套利停止", "%.6f" % b_no_grid, "弃光重新出现", "%.6f" % b_zero_curt,
      "年 EFC 归零", "%.6f" % b_stop)
print("层B c_cyc=0 基准：年 EFC", "%.2f" % dfB.EFC.iloc[0], "（日均", "%.4f" % dfB.日均EFC.iloc[0],
      "）节省", "%.2f" % savB0, "元")
print("层B 节省降至 50% / 10% 的临界值", "%.4f" % b50, "/", "%.4f" % b10)

# 层C：问2 全年滚动带 c_cyc
N = L.N_T
IG = {k: i * N for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}


def settle_day(g_plan, load_kw, pv_kw, soc_start, c_cyc=0.0):
    """结算 LP：目标加循环成本 c_cyc*sum(d)"""
    n = N
    idx = np.arange(n)
    c_obj = np.zeros(6 * n)
    c_obj[IG["E2"] + idx] = L.BETA_EMG * pr
    c_obj[IG["C"] + idx] = L.THRU_PEN
    c_obj[IG["D"] + idx] = L.THRU_PEN + c_cyc
    row_ix, col_ix, val, rhs = [], [], [], []
    for name, coef in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        row_ix.append(idx)
        col_ix.append(IG[name] + idx)
        val.append(coef * np.ones(n))
    rhs.append(load_kw * L.DT - pv_kw * L.DT)
    soc_rows = n + idx
    row_ix.append(soc_rows)
    col_ix.append(IG["E"] + idx)
    val.append(np.ones(n))
    row_ix.append(soc_rows)
    col_ix.append(IG["C"] + idx)
    val.append(-L.ETA_C * np.ones(n))
    row_ix.append(soc_rows)
    col_ix.append(IG["D"] + idx)
    val.append((1.0 / L.ETA_D) * np.ones(n))
    row_ix.append(soc_rows[1:])
    col_ix.append(IG["E"] + idx[1:] - 1)
    val.append(-np.ones(n - 1))
    soc0 = np.zeros(n)
    soc0[0] = soc_start
    rhs.append(soc0)
    A_eq = coo_matrix((np.concatenate(val), (np.concatenate(row_ix), np.concatenate(col_ix))),
                      shape=(2 * n, 6 * n)).tocsr()
    lb = np.zeros(6 * n)
    ub = np.empty(6 * n)
    ub[IG["GD"]:IG["GD"] + n] = g_plan
    ub[IG["C"]:IG["C"] + n] = L.E_MAX
    ub[IG["D"]:IG["D"] + n] = L.E_MAX
    ub[IG["S"]:IG["S"] + n] = pv_kw * L.DT
    lb[IG["E"]:IG["E"] + n] = L.SOC_MIN
    ub[IG["E"]:IG["E"] + n] = L.SOC_MAX
    ub[IG["E2"]:IG["E2"] + n] = np.inf
    return linprog(c_obj, A_eq=A_eq, b_eq=np.concatenate(rhs),
                   bounds=list(zip(lb, ub)), method="highs")


dfC = None
if RUN_LAYER_C:
    GRID_C = [0.0, 0.25, 0.50, 0.75, 1.00, 1.20]
    rowsC = []
    for cyc_cost in GRID_C:
        t0 = time.time()
        rng = np.random.default_rng(42)      # 固定种子 42
        plan = np.zeros((L.N_DAY, N))
        charge = np.zeros((L.N_DAY, N))
        discharge = np.zeros((L.N_DAY, N))
        emg = np.zeros((L.N_DAY, N))
        pct = np.full(L.N_DAY, np.nan)
        soc = L.SOC_INIT
        for day in range(L.N_DAY):
            if day < 5:
                plan[day] = np.maximum(ld[day] * L.DT - pva[day] * L.DT, 0.0)
                res_settle = settle_day(plan[day], ld[day], pva[day], soc, cyc_cost)
                x = res_settle.x
                charge[day] = x[N:2 * N]
                discharge[day] = x[2 * N:3 * N]
                emg[day] = x[5 * N:6 * N]
                soc = x[4 * N:5 * N][-1]
                continue
            load_fc = ST.point_forecast(ld, DTYPE, day)
            pv_fc = ST.point_forecast(pva, DTYPE, day)
            res_load = ST.residual_pool(ld, DTYPE, day, eps=0.0)
            res_pv = ST.residual_pool(pva, DTYPE, day, eps=50.0)
            load_sc = ST.sample_scenarios(res_load, 20, rng) * load_fc
            pv_sc = np.clip(ST.sample_scenarios(res_pv, 20, rng) * pv_fc, 0.0, None)
            res = ST.solve_two_stage(pr, load_fc, pv_fc, load_sc, pv_sc, soc, None, c_cyc=cyc_cost)
            if res is None:
                plan[day] = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0.0)
            else:
                plan[day] = res["g_plan"]
                net_sc = (load_sc - pv_sc).sum(axis=1) * L.DT
                net_act = float((ld[day] - pva[day]).sum() * L.DT)
                pct[day] = float((net_sc < net_act).mean() * 100)
            res_settle = settle_day(plan[day], ld[day], pva[day], soc, cyc_cost)
            assert res_settle.success, "层C 第 %d 天结算失败 c_cyc=%.2f" % (day, cyc_cost)
            x = res_settle.x
            charge[day] = x[N:2 * N]
            discharge[day] = x[2 * N:3 * N]
            emg[day] = x[5 * N:6 * N]
            soc = x[4 * N:5 * N][-1]
        plan_rep, emg_rep, disch_rep = plan[SL], emg[SL], discharge[SL]
        cost_buy = float((pr[None, :] * plan_rep).sum())
        cost_emg = float(L.BETA_EMG * (pr[None, :] * emg_rep).sum())
        cost_cyc = float(cyc_cost * disch_rep.sum())
        rowsC.append(dict(c_cyc=cyc_cost, 购电费=cost_buy, 紧急费=cost_emg, 循环成本=cost_cyc,
                          总费用=cost_buy + cost_emg + cost_cyc, EFC=efc_of(disch_rep.sum()),
                          放电量=float(disch_rep.sum()), 紧急购电=float(emg_rep.sum()),
                          计划量=float(plan_rep.sum()),
                          平均分位=float(np.nanmean(pct[REP0:])), 用时s=time.time() - t0))
        print("层C c_cyc", "%.2f" % cyc_cost, "总费用", "%.2f" % rowsC[-1]["总费用"], "元",
              "EFC", "%.3f" % rowsC[-1]["EFC"], "紧急购电", "%.1f" % rowsC[-1]["紧急购电"],
              "kWh 用时", "%.0f" % rowsC[-1]["用时s"], "s")
    dfC = pd.DataFrame(rowsC)
    dfC["相对无储能节省"] = NS_YEAR - dfC["总费用"]
    dfC["节省率%"] = dfC["相对无储能节省"] / NS_YEAR * 100
    dfC.to_csv("results/循环成本_问2.csv", index=False, encoding="utf-8-sig")   # 先落盘
    assert abs(dfC.总费用.iloc[0] - 13234106.71) < 5.0, \
        "层C c_cyc=0 未复现问2 头条结果 13,234,106.71，实得 %.2f" % dfC.总费用.iloc[0]
    print("层C c_cyc=0 复现问2 头条结果", "%.2f" % dfC.总费用.iloc[0], "元")

npz = dict(c_arb=C_ARB, c_stop_analytic=PI_MAX, c_stop_day=c_stop_num, c_stop_year=b_stop,
           c50_day=c50, c10_day=c10, c50_year=b50, c10_year=b10,
           c_no_grid=a_no_grid, c_zero_curt=a_zero_curt, surplus=SURP,
           c_no_grid_year=b_no_grid, c_zero_curt_year=b_zero_curt, surplus_year=SURP_Y,
           sav_day0=sav0, sav_year0=savB0, NS_DAY=NS_DAY, NS_YEAR=NS_YEAR)
np.savez("results/循环成本_结果.npz", **npz)

# 临界点 npz，供图27 使用
np.savez("results/循环成本_临界点.npz",
         c_no_grid=a_no_grid, c_zero_curt=a_zero_curt, c_stop=a_stop,
         surplus=SURP, pi_max=PI_MAX, pi_min=PI_MIN)

print("已写出 results/循环成本_单日.csv、results/循环成本_年度.csv、results/循环成本_问2.csv、"
      "results/循环成本_结果.npz、results/循环成本_临界点.npz、figures/fig27_cycle_cost_sensitivity.png")

print("结论：c_cyc=0.5 元/kWh 时单日 EFC 由", "%.3f" % dfA.EFC.iloc[0], "降至",
      "%.3f" % float(dfA.iloc[(dfA.c_cyc - 0.5).abs().argmin()].EFC), "，总成本仍低于无储能",
      "%.2f" % float(dfA.iloc[(dfA.c_cyc - 0.5).abs().argmin()].节省), "元")
print("结论：c_cyc >= pi_max =", "%.4f" % PI_MAX,
      "元/kWh 时电池完全闲置；c_cyc < 1.0956 元/kWh 时问1『弃光由 6247.96 kWh 降为 0』不变")
if dfC is not None:
    print("结论：问2 c_cyc=0.5 元/kWh 时报告期总费用",
          "%.2f" % float(dfC.iloc[(dfC.c_cyc - 0.5).abs().argmin()].总费用), "元，相对无储能仍节省",
          "%.2f" % float(dfC.iloc[(dfC.c_cyc - 0.5).abs().argmin()]["节省率%"]), "%")
