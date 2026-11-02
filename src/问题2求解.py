"""问题2 求解：两阶段随机规划，写 result2.xlsx。"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl, shutil, datetime as dt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core as L
import stoch as ST

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
rng = np.random.default_rng(42)

N = L.N_T
N_SCEN = 20                     # 情景数 20
REP0 = 31
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]

D = L.load_all(verbose=False)
price = D["typical"]["price"]
load_2d = D["actual"]["load_kw"]
pv_2d = D["actual"]["pv_kw"]
dates = D["actual"]["dates"]
DTYPE = ST.day_type(dates)
print("报告期 %s ~ %s，情景数 %d"
      % (L.REPORT_START.date(), L.REPORT_END.date(), N_SCEN))


def settle_day(g_plan, load_kw, pv_kw, soc_start, soc_term=None):
    """按实际负载与光伏结算已承诺计划量。"""
    n = N
    base = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(n)
    obj = np.zeros(6 * n)
    obj[base["E2"] + idx] = L.BETA_EMG * price
    obj[base["C"] + idx] = L.THRU_PEN        # 吞吐量正则化
    obj[base["D"] + idx] = L.THRU_PEN

    row, col, val, rhs = [], [], [], []
    for name, coef in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        row.append(idx)
        col.append(base[name] + idx)
        val.append(coef * np.ones(n))
    rhs.append(load_kw * L.DT - pv_kw * L.DT)
    bal = n + idx
    row.append(bal); col.append(base["E"] + idx); val.append(np.ones(n))
    row.append(bal); col.append(base["C"] + idx); val.append(-L.ETA_C * np.ones(n))
    row.append(bal); col.append(base["D"] + idx); val.append((1.0 / L.ETA_D) * np.ones(n))
    row.append(bal[1:]); col.append(base["E"] + idx[1:] - 1); val.append(-np.ones(n - 1))
    soc0 = np.zeros(n)
    soc0[0] = soc_start
    rhs.append(soc0)
    A_eq = coo_matrix((np.concatenate(val), (np.concatenate(row), np.concatenate(col))),
                      shape=(2 * n, 6 * n)).tocsr()
    # g_draw <= g_plan 由变量上界施加
    lb = np.zeros(6 * n)
    ub = np.empty(6 * n)
    ub[base["GD"]:base["GD"] + n] = g_plan
    ub[base["C"]:base["C"] + n] = L.E_MAX
    ub[base["D"]:base["D"] + n] = L.E_MAX
    ub[base["S"]:base["S"] + n] = pv_kw * L.DT
    lb[base["E"]:base["E"] + n] = L.SOC_MIN
    ub[base["E"]:base["E"] + n] = L.SOC_MAX
    ub[base["E2"]:base["E2"] + n] = np.inf
    # 每日末端不闭环
    if soc_term is not None:
        lb[base["E"] + n - 1] = soc_term
    return linprog(obj, A_eq=A_eq, b_eq=np.concatenate(rhs),
                   bounds=list(zip(lb, ub)), method="highs")


t_start = time.time()
plan_2d = np.zeros((L.N_DAY, N))
chg_2d = np.zeros((L.N_DAY, N))
dis_2d = np.zeros((L.N_DAY, N))
spill_2d = np.zeros((L.N_DAY, N))
soc_2d = np.zeros((L.N_DAY, N))
emg_2d = np.zeros((L.N_DAY, N))
pct_of_plan = np.full(L.N_DAY, np.nan)      # 计划量在情景净负载分布中的分位
soc = L.SOC_INIT
for d in range(L.N_DAY):
    if d < 5:                                # 热身期用当日实际
        plan_2d[d] = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0.0)
        settled = settle_day(plan_2d[d], load_2d[d], pv_2d[d], soc)
        x = settled.x
        chg_2d[d] = x[N:2*N]; dis_2d[d] = x[2*N:3*N]; spill_2d[d] = x[3*N:4*N]
        soc_2d[d] = x[4*N:5*N]; emg_2d[d] = x[5*N:6*N]; soc = soc_2d[d, -1]
        continue
    load_fc = ST.point_forecast(load_2d, DTYPE, d)
    pv_fc = ST.point_forecast(pv_2d, DTYPE, d)
    load_res = ST.residual_pool(load_2d, DTYPE, d, eps=0.0)
    pv_res = ST.residual_pool(pv_2d, DTYPE, d, eps=50.0)
    load_scen = ST.sample_scenarios(load_res, N_SCEN, rng) * load_fc
    pv_scen = np.clip(ST.sample_scenarios(pv_res, N_SCEN, rng) * pv_fc, 0.0, None)
    two_stage = ST.solve_two_stage(price, load_fc, pv_fc, load_scen, pv_scen, soc, None)
    if two_stage is None:
        plan_2d[d] = np.maximum(load_fc * L.DT - pv_fc * L.DT, 0.0)
    else:
        plan_2d[d] = two_stage["g_plan"]
        net_sc = (load_scen - pv_scen).sum(axis=1) * L.DT
        net_act = float((load_2d[d] - pv_2d[d]).sum() * L.DT)
        pct_of_plan[d] = float((net_sc < net_act).mean() * 100)
    settled = settle_day(plan_2d[d], load_2d[d], pv_2d[d], soc, None)   # 储能跨日自由传递
    assert settled.success, "第 %d 天结算失败" % d
    x = settled.x
    chg_2d[d] = x[N:2*N]
    dis_2d[d] = x[2*N:3*N]
    spill_2d[d] = x[3*N:4*N]
    soc_2d[d] = x[4*N:5*N]
    emg_2d[d] = x[5*N:6*N]
    soc = soc_2d[d, -1]
    if (d + 1) % 40 == 0:
        print("已完成 %3d/%d 天，用时 %.0f s" % (d + 1, L.N_DAY, time.time() - t_start))
print("滚动完成，用时 %.1f s" % (time.time() - t_start))

report = slice(REP0, L.N_DAY)
rep_dates = dates.iloc[report].reset_index(drop=True)
plan_rep = plan_2d[report]
emg_rep = emg_2d[report]
chg_rep = chg_2d[report]
dis_rep = dis_2d[report]
spill_rep = spill_2d[report]
soc_rep = soc_2d[report]
cost_plan = float((price * plan_rep).sum())
cost_emg = float(L.BETA_EMG * (price * emg_rep).sum())
print("报告期计划购电 %.2f 元，紧急购电 %.2f 元（%.2f kWh，涉及 %d 天）"
      % (cost_plan, cost_emg, emg_rep.sum(), int((emg_rep.sum(axis=1) > 1e-6).sum())))
print("报告期合计 %.2f 元" % (cost_plan + cost_emg))

pct_vals = pct_of_plan[REP0:]
pct_vals = pct_vals[~np.isnan(pct_vals)]
above_p80 = pct_vals > 80
print("净负载相对情景分布的平均分位 %.1f%%，超出 P80 的天数占比 %.1f%%"
      % (pct_vals.mean(), above_p80.mean() * 100))

net_base = np.maximum(load_2d[report] * L.DT - pv_2d[report] * L.DT, 0)
cost_nostore = float((price * net_base).sum())
cost_perfect = float((price * np.maximum(load_2d[report] * L.DT - pv_2d[report] * L.DT, 0)).sum())
print("无储能基准 %.2f 元，两阶段随机规划 %.2f 元" % (cost_nostore, cost_plan + cost_emg))
print("相对无储能节省 %.2f 元（%.2f%%）"
      % (cost_nostore - cost_plan - cost_emg,
         (cost_nostore - cost_plan - cost_emg) / cost_nostore * 100))

template = os.path.join(L.ATT, "附件5", "result2.xlsx")
out = os.path.join(L.RES, "result2.xlsx")
shutil.copy(template, out)
wb = openpyxl.load_workbook(out)
ws = wb["计划购电量"]
for i in range(len(rep_dates)):
    for t in range(N):
        ws.cell(row=2 + i, column=2 + t, value=float(round(plan_rep[i, t], 6)))
    ws.cell(row=2 + i, column=146, value=float(round(plan_rep[i].sum(), 6)))
    ws.cell(row=2 + i, column=147, value=float(round((price * plan_rep[i]).sum(), 6)))
cf_pos = wb.sheetnames.index("充放电量")
del wb["充放电量"]
ws2 = wb.create_sheet("充放电量", cf_pos)
ws2.append(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"])
for i in range(len(rep_dates)):
    chg = L.block_sums(chg_rep[i])
    dis = L.block_sums(dis_rep[i])
    first_row = 2 + i * 6
    for k, (block_name, _, _) in enumerate(L.BLOCKS):
        blk_row = first_row + k
        if k == 0:
            ws2.cell(row=blk_row, column=1, value=rep_dates.iloc[i].to_pydatetime())
            ws2.cell(row=blk_row, column=5, value=dt.time(0, 0))
            ws2.cell(row=blk_row, column=6, value=float(round(
                soc_2d[REP0 - 1, -1] if i == 0 else soc_rep[i - 1, -1], 6)))
        if k == 1:
            ws2.cell(row=blk_row, column=5, value="24:00")
            ws2.cell(row=blk_row, column=6, value=float(round(soc_rep[i, -1], 6)))
        ws2.cell(row=blk_row, column=2, value=block_name)
        ws2.cell(row=blk_row, column=3, value=float(round(chg[k], 6)))
        ws2.cell(row=blk_row, column=4, value=float(round(dis[k], 6)))
em_pos = wb.sheetnames.index("紧急购电量")
del wb["紧急购电量"]
ws3 = wb.create_sheet("紧急购电量", em_pos)
ws3.append(["日期", "购电时间段", "购电量"])
excel_row = 2
for i in range(len(rep_dates)):
    active = np.where(emg_rep[i] > 1e-6)[0]
    ws3.cell(row=excel_row, column=1, value=rep_dates.iloc[i].to_pydatetime())
    if len(active) == 0:
        ws3.cell(row=excel_row, column=2, value="无")
        ws3.cell(row=excel_row, column=3, value=0.0)
        excel_row += 1
    else:
        # 连续为正的时段合并成区间
        runs = []
        start = active[0]
        for a, b in zip(active[:-1], active[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, active[-1]))
        for j, (a, b) in enumerate(runs):
            if j > 0:
                ws3.cell(row=excel_row, column=1, value=None)
            ws3.cell(row=excel_row, column=2, value="%02d:%02d-%02d:%02d"
                     % ((a * 10) // 60, (a * 10) % 60, ((b + 1) * 10) // 60 % 24,
                        ((b + 1) * 10) % 60))
            ws3.cell(row=excel_row, column=3, value=float(round(emg_rep[i, a:b + 1].sum(), 6)))
            excel_row += 1
wb.save(out)
print("已写入", out)

print("表3 指定日期的紧急购电量")
for key_date in KEY_DATES:
    pos = int(np.where(rep_dates == pd.Timestamp(key_date))[0][0])
    print(key_date, "紧急 %.2f kWh" % emg_rep[pos].sum(), "计划 %.2f kWh" % plan_rep[pos].sum(),
          "当日 %.2f 元" % (price @ plan_rep[pos] + L.BETA_EMG * price @ emg_rep[pos]))

np.savez(os.path.join(L.RES, "问2_结果.npz"),
         g=plan_rep, e=emg_rep, c=chg_rep, d=dis_rep, s=spill_rep, E=soc_rep, price=price,
         g_all=plan_2d, E_all=soc_2d, e_all=emg_2d, pct=pct_of_plan,
         dates=rep_dates.values.astype("datetime64[D]"),
         cost_plan=cost_plan, cost_emg=cost_emg, cost_nostore=cost_nostore, n_scen=N_SCEN)
print("已生成数据：results/问2_结果.npz")
