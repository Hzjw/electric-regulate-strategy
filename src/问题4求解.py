"""问题4 求解：实时电价下的两阶段随机规划与滚动决策。"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl, shutil, datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core as L
import stoch as ST

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
rng = np.random.default_rng(42)

N = L.N_T
N_SCEN = 20
N_SCEN_STAGE = 12
DECIDE_T = {1: 36, 2: 72, 3: 108}
REP0 = 31
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]

D = L.load_all(verbose=False)
price_mean = D["typical"]["price"]
price_rt = D["price_rt"]["price"]
load_2d = D["actual"]["load_kw"]
pv_2d = D["actual"]["pv_kw"]
FC = D["forecast"]["fc"]
dates = D["actual"]["dates"]
DTYPE = ST.day_type(dates)
fc10 = np.zeros((L.N_DAY, 4, N))
for d in range(L.N_DAY):
    for k in range(4):
        fc10[d, k] = L.forecast_to_10min(FC[d], k)
# 预报偏差订正表：扩张窗口，样本外
bias = L.forecast_bias_table(fc10, pv_2d)
avg_price = price_rt.mean(axis=0)
print("附件1 电价 vs 附件4 逐时段均值：最大绝对差 %.6f，相关系数 %.6f"
      % (np.abs(price_mean - avg_price).max(), np.corrcoef(price_mean, avg_price)[0, 1]))


def settle_day(price, g_plan, load_kw, pv_kw, soc_start):
    """实时结算；储能跨日自由传递。"""
    n = N
    base = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(n)
    obj = np.zeros(6 * n)
    obj[base["E2"] + idx] = L.BETA_EMG * price
    obj[base["C"] + idx] = L.THRU_PEN
    obj[base["D"] + idx] = L.THRU_PEN
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
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
    return linprog(obj, A_eq=A_eq, b_eq=np.concatenate(rhs),
                   bounds=list(zip(lb, ub)), method="highs")


def run_42(price2d):
    """问题4-2：两阶段随机规划（逐日 0:00 定计划，实时结算）"""
    plan_2d = np.zeros((L.N_DAY, N))
    chg_2d = np.zeros((L.N_DAY, N))
    dis_2d = np.zeros((L.N_DAY, N))
    spill_2d = np.zeros((L.N_DAY, N))
    soc_2d = np.zeros((L.N_DAY, N))
    emg_2d = np.zeros((L.N_DAY, N))
    soc = L.SOC_INIT
    for d in range(L.N_DAY):
        price_day = price2d[d]
        load_fc = ST.load_forecast(load_2d, DTYPE, d) if d >= 5 else load_2d[d]
        pv_fc = ST.pv_forecast_persistence(pv_2d, DTYPE, d) if d >= 5 else pv_2d[d]
        if d >= 5:
            load_res = ST.load_residual_pool(load_2d, DTYPE, d)
            pv_res = ST.pv_residual_pool_persist(pv_2d, DTYPE, d)
            load_scen = ST.load_scenarios(load_res, load_fc, N_SCEN, rng)
            pv_scen = np.clip(ST.sample_scenarios(pv_res, N_SCEN, rng) * pv_fc, 0, None)
            two_stage = ST.solve_two_stage(price_day, load_fc, pv_fc, load_scen, pv_scen, soc, None)
            plan = (two_stage["g_plan"] if two_stage is not None
                    else np.maximum(load_fc * L.DT - pv_fc * L.DT, 0))
        else:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0)
        plan_2d[d] = plan
        settled = settle_day(price_day, plan, load_2d[d], pv_2d[d], soc)
        x = settled.x
        chg_2d[d] = x[N:2*N]
        dis_2d[d] = x[2*N:3*N]
        spill_2d[d] = x[3*N:4*N]
        soc_2d[d] = x[4*N:5*N]
        emg_2d[d] = x[5*N:6*N]
        soc = soc_2d[d, -1]
        if (d + 1) % 60 == 0:
            print("4-2 已完成 %3d/%d 天" % (d + 1, L.N_DAY))
    return plan_2d, chg_2d, dis_2d, spill_2d, soc_2d, emg_2d


def run_43(price2d, active=(1, 2, 3)):
    """问题4-3：情景化滚动随机决策"""
    plan_2d = np.zeros((L.N_DAY, N))
    adj_2d = np.zeros((L.N_DAY, N))
    chg_2d = np.zeros((L.N_DAY, N))
    dis_2d = np.zeros((L.N_DAY, N))
    spill_2d = np.zeros((L.N_DAY, N))
    soc_2d = np.zeros((L.N_DAY, N))
    emg_2d = np.zeros((L.N_DAY, N))
    soc = L.SOC_INIT
    for d in range(L.N_DAY):
        price_day = price2d[d]
        load_fc = ST.load_forecast(load_2d, DTYPE, d) if d >= 5 else load_2d[d]
        pv_fc = fc10[d, 0] if d >= 5 else pv_2d[d]
        if d >= 5:
            load_res = ST.load_residual_pool(load_2d, DTYPE, d)
            pv_res = ST.fc_residual_pool(fc10[:, 0], pv_2d, d, 0, eps=50.0)
            load_scen = ST.load_scenarios(load_res, load_fc, N_SCEN, rng)
            pv_scen = np.clip(ST.sample_scenarios(pv_res, N_SCEN, rng) * pv_fc, 0, None)
            two_stage = ST.solve_two_stage(price_day, load_fc, pv_fc, load_scen, pv_scen, soc, None)
            plan = (two_stage["g_plan"] if two_stage is not None
                    else np.maximum(load_fc * L.DT - pv_fc * L.DT, 0))
        else:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0)
        plan_2d[d] = plan
        committed = plan.copy()
        for k in active:
            t_decide = DECIDE_T[k]
            # 阶段初值取自全天结算储能轨迹
            soc_traj = ST.soc_trajectory(price_day, committed, load_2d[d], pv_2d[d], soc)
            soc_at = float(soc_traj[0][t_decide - 1]) if soc_traj is not None else soc
            pv_stage = np.clip(fc10[d, k] - bias[d, k], 0.0, None) if d >= 5 else pv_2d[d]
            stage, stage_map, stage_n = L.solve_stage_lp(price_day, load_fc, pv_stage, plan,
                                                         t_decide, soc_at)
            if stage.success:
                committed[t_decide:] = stage.x[stage_map["GA"]:stage_map["GA"] + stage_n]
        adj_2d[d] = committed
        soc_traj = ST.soc_trajectory(price_day, committed, load_2d[d], pv_2d[d], soc)
        soc_2d[d], chg_2d[d], dis_2d[d], spill_2d[d], emg_2d[d] = soc_traj
        soc = soc_2d[d, -1]
        if (d + 1) % 60 == 0:
            print("4-3 已完成 %3d/%d 天" % (d + 1, L.N_DAY))
    return plan_2d, adj_2d, chg_2d, dis_2d, spill_2d, soc_2d, emg_2d


t42 = time.time()
plan42, chg42, dis42, spill42, soc42, emg42 = run_42(price_rt)
print("4-2 完成，用时 %.1f s" % (time.time() - t42))
report = slice(REP0, L.N_DAY)
cost42 = float((price_rt[report] * plan42[report]).sum()
               + L.BETA_EMG * (price_rt[report] * emg42[report]).sum())
print("4-2 报告期费用 %.2f 元（紧急 %.2f kWh）" % (cost42, emg42[report].sum()))

t43 = time.time()
plan43, adj43, chg43, dis43, spill43, soc43, emg43 = run_43(price_rt)
print("4-3 完成，用时 %.1f s" % (time.time() - t43))
short43 = np.maximum(plan43[report] - adj43[report], 0)
over43 = np.maximum(adj43[report] - plan43[report], 0)
parts = {
    "base": float((price_rt[report] * np.minimum(plan43[report], adj43[report])).sum()),
    "def": float((price_rt[report] * L.BETA_DEF * short43).sum()),
    "exc": float((price_rt[report] * L.BETA_EXC * over43).sum()),
    "emg": float(L.BETA_EMG * (price_rt[report] * emg43[report]).sum())}
cost43 = sum(parts.values())
print("4-3 报告期分解：基准 %.2f，违约 %.2f，超额 %.2f，紧急 %.2f，合计 %.2f"
      % (parts["base"], parts["def"], parts["exc"], parts["emg"], cost43))


def write_result(tpl_name, out_name, sheets):
    shutil.copy(os.path.join(L.ATT, "附件5", tpl_name), os.path.join(L.RES, out_name))
    wb = openpyxl.load_workbook(os.path.join(L.RES, out_name))
    for name, mat in sheets.get("plan", []):
        ws = wb[name]
        for i in range(mat.shape[0]):
            for t in range(N):
                ws.cell(row=2 + i, column=2 + t, value=float(round(mat[i, t], 6)))
            ws.cell(row=2 + i, column=146, value=float(round(mat[i].sum(), 6)))
            ws.cell(row=2 + i, column=147,
                    value=float(round((price_rt[report][i] * mat[i]).sum(), 6)))
    cf_pos = wb.sheetnames.index("充放电量")
    del wb["充放电量"]
    ws2 = wb.create_sheet("充放电量", cf_pos)
    ws2.append(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"])
    rep_dates = dates.iloc[report].reset_index(drop=True)
    for i in range(len(rep_dates)):
        chg = L.block_sums(sheets["c"][i])
        dis = L.block_sums(sheets["d"][i])
        first_row = 2 + i * 6
        for k, (block_name, _, _) in enumerate(L.BLOCKS):
            blk_row = first_row + k
            if k == 0:
                ws2.cell(row=blk_row, column=1, value=rep_dates.iloc[i].to_pydatetime())
                ws2.cell(row=blk_row, column=5, value=dt.time(0, 0))
                ws2.cell(row=blk_row, column=6, value=float(round(
                    L.SOC_INIT if i == 0 and REP0 == 0 else sheets["E"][max(i - 1, 0), -1], 6)))
            if k == 1:
                ws2.cell(row=blk_row, column=5, value="24:00")
                ws2.cell(row=blk_row, column=6, value=float(round(sheets["E"][i, -1], 6)))
            ws2.cell(row=blk_row, column=2, value=block_name)
            ws2.cell(row=blk_row, column=3, value=float(round(chg[k], 6)))
            ws2.cell(row=blk_row, column=4, value=float(round(dis[k], 6)))
    em_pos = wb.sheetnames.index("紧急购电量")
    del wb["紧急购电量"]
    ws3 = wb.create_sheet("紧急购电量", em_pos)
    ws3.append(["日期", "购电时间段", "购电量"])
    excel_row = 2
    for i in range(len(rep_dates)):
        active = np.where(sheets["e"][i] > 1e-6)[0]
        ws3.cell(row=excel_row, column=1, value=rep_dates.iloc[i].to_pydatetime())
        if len(active) == 0:
            ws3.cell(row=excel_row, column=2, value="无")
            ws3.cell(row=excel_row, column=3, value=0.0)
            excel_row += 1
        else:
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
                         % ((a*10)//60, (a*10) % 60, ((b+1)*10)//60 % 24, ((b+1)*10) % 60))
                ws3.cell(row=excel_row, column=3,
                         value=float(round(sheets["e"][i, a:b+1].sum(), 6)))
                excel_row += 1
    wb.save(os.path.join(L.RES, out_name))
    print("已写入", out_name)


write_result("result4-2.xlsx", "result4-2.xlsx",
             {"plan": [("计划购电量", plan42[report])], "c": chg42[report], "d": dis42[report],
              "E": soc42[report], "e": emg42[report]})
write_result("result4-3.xlsx", "result4-3.xlsx",
             {"plan": [("计划购电量", plan43[report]), ("调整购电量", adj43[report])],
              "c": chg43[report], "d": dis43[report], "E": soc43[report], "e": emg43[report]})

# 方案A 按实时电价完整结算
res2 = np.load(os.path.join(L.RES, "问2_结果.npz"), allow_pickle=True)
plan2 = res2["g_all"]
soc_a = L.SOC_INIT
buy_a = 0.0
emg_a = 0.0
for d in range(REP0, L.N_DAY):
    lp_a = settle_day(price_rt[d], plan2[d], load_2d[d], pv_2d[d], soc_a)
    soc_a = lp_a.x[4 * N + N - 1]
    buy_a += float(price_rt[d] @ plan2[d])
    emg_a += float(L.BETA_EMG * (price_rt[d] @ lp_a.x[5 * N:6 * N]))
cost_det_rt = buy_a + emg_a
regret = cost_det_rt - cost42
print("方案A 确定性等价电价规划 + 实时电价完整结算 = 购电 %.2f + 紧急 %.2f = %.2f 元"
      % (buy_a, emg_a, cost_det_rt))
print("方案B 实时电价直接优化（问题4-2）= %.2f 元" % cost42)
print("后悔值 %.2f 元（%.2f%%）" % (regret, regret / cost_det_rt * 100))

np.savez(os.path.join(L.RES, "问4_结果.npz"),
         g42=plan42[report], e42=emg42[report], c42=chg42[report], d42=dis42[report],
         E42=soc42[report],
         g43_plan=plan43[report], g43_adj=adj43[report], e43=emg43[report], c43=chg43[report],
         d43=dis43[report], E43=soc43[report],
         price_rt=price_rt[report], price_mean=price_mean,
         dates=dates.iloc[report].values.astype("datetime64[D]"),
         C42=cost42, C43=cost43, C_det_in_rt=cost_det_rt, regret=regret, cb43=parts)
print("已生成数据：results/问4_结果.npz")
