"""问题3 求解：滚动时域随机决策，写 result3.xlsx。"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl, shutil, datetime as _dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core as L
import stoch as ST

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
rng = np.random.default_rng(42)

N = L.N_T
N_SCEN = 20
DECIDE_T = {1: 36, 2: 72, 3: 108}
REP0 = 31
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]

print("问题3 滚动时域随机决策模型")

D = L.load_all(verbose=False)
price = D["typical"]["price"]
load_2d = D["actual"]["load_kw"]
pv_2d = D["actual"]["pv_kw"]
FC = D["forecast"]["fc"]
dates = D["actual"]["dates"]
DTYPE = ST.day_type(dates)
FC10 = np.zeros((L.N_DAY, 4, N))
for d in range(L.N_DAY):
    for k in range(4):
        FC10[d, k] = L.forecast_to_10min(FC[d], k)
# 偏差订正表：扩张窗口，样本外
BIAS = L.forecast_bias_table(FC10, pv_2d)
print("预报已降尺度为 144 段，午后各版本平均偏低 %.1f~%.1f kW"
      % (BIAS[REP0:, 0, 72:].mean(), BIAS[REP0:, 3, 72:].mean()))


def settle_day(g_plan, load_kw, pv_kw, soc_start):
    """核算充放电量；GD 为承诺购电上限。"""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n = N
    ofs = {name: i * n for i, name in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    seg = np.arange(n)
    cost = np.zeros(6 * n)
    cost[ofs["E2"] + seg] = L.BETA_EMG * price
    cost[ofs["C"] + seg] = L.THRU_PEN
    cost[ofs["D"] + seg] = L.THRU_PEN
    eq_row, eq_col, eq_val, eq_rhs = [], [], [], []
    for name, coef in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        eq_row.append(seg)
        eq_col.append(ofs[name] + seg)
        eq_val.append(coef * np.ones(n))
    eq_rhs.append(load_kw * L.DT - pv_kw * L.DT)
    soc_row = n + seg
    eq_row.append(soc_row)
    eq_col.append(ofs["E"] + seg)
    eq_val.append(np.ones(n))
    eq_row.append(soc_row)
    eq_col.append(ofs["C"] + seg)
    eq_val.append(-L.ETA_C * np.ones(n))
    eq_row.append(soc_row)
    eq_col.append(ofs["D"] + seg)
    eq_val.append((1.0 / L.ETA_D) * np.ones(n))
    eq_row.append(soc_row[1:])
    eq_col.append(ofs["E"] + seg[1:] - 1)
    eq_val.append(-np.ones(n - 1))
    e0 = np.zeros(n)
    e0[0] = soc_start
    eq_rhs.append(e0)
    A_eq = coo_matrix((np.concatenate(eq_val), (np.concatenate(eq_row), np.concatenate(eq_col))),
                      shape=(2 * n, 6 * n)).tocsr()
    lb = np.zeros(6 * n)
    ub = np.empty(6 * n)
    ub[ofs["GD"]:ofs["GD"] + n] = g_plan
    ub[ofs["C"]:ofs["C"] + n] = L.E_MAX
    ub[ofs["D"]:ofs["D"] + n] = L.E_MAX
    ub[ofs["S"]:ofs["S"] + n] = pv_kw * L.DT
    lb[ofs["E"]:ofs["E"] + n] = L.SOC_MIN
    ub[ofs["E"]:ofs["E"] + n] = L.SOC_MAX
    ub[ofs["E2"]:ofs["E2"] + n] = np.inf
    return linprog(cost, A_eq=A_eq, b_eq=np.concatenate(eq_rhs),
                   bounds=list(zip(lb, ub)), method="highs")


def run(active_stages, vint=None, verbose=False):
    """全年滚动：阶段0 定计划，阶段1~3 调整；vint 为各阶段预报版本。"""
    if vint is None:
        vint = {k: k for k in (1, 2, 3)}
    rng = np.random.default_rng(42)
    gp = np.zeros((L.N_DAY, N))
    ga = np.zeros((L.N_DAY, N))
    cc = np.zeros((L.N_DAY, N))
    dd = np.zeros((L.N_DAY, N))
    ss = np.zeros((L.N_DAY, N))
    EE = np.zeros((L.N_DAY, N))
    ee = np.zeros((L.N_DAY, N))
    soc = L.SOC_INIT
    for d in range(L.N_DAY):
        load_fc = ST.load_forecast(load_2d, DTYPE, d) if d >= 5 else load_2d[d]
        pv_fc = FC10[d, 0] if d >= 5 else pv_2d[d]
        if d >= 5:
            load_pool = ST.load_residual_pool(load_2d, DTYPE, d)
            pv_pool = ST.fc_residual_pool(FC10[:, 0], pv_2d, d, 0, eps=50.0)
            load_scen = ST.load_scenarios(load_pool, load_fc, N_SCEN, rng)
            pv_scen = np.clip(ST.sample_scenarios(pv_pool, N_SCEN, rng) * pv_fc, 0, None)
            res0 = ST.solve_two_stage(price, load_fc, pv_fc, load_scen, pv_scen, soc, None)
            plan = (res0["g_plan"] if res0 is not None
                    else np.maximum(load_fc * L.DT - pv_fc * L.DT, 0))
        else:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0)
        gp[d] = plan
        committed = plan.copy()
        for k in active_stages:
            t_dec = DECIDE_T[k]
            # 阶段初值取自全天结算轨迹
            soc_path = ST.soc_trajectory(price, committed, load_2d[d], pv_2d[d], soc)
            soc_at = float(soc_path[0][t_dec - 1]) if soc_path is not None else soc
            ver = vint.get(k, k)
            pv_stage = (np.clip(FC10[d, ver] - BIAS[d, ver], 0.0, None)
                        if d >= 5 else pv_2d[d])
            # 只优化 [t_dec, 144)，负载用预报
            rk, ofs, m = L.solve_stage_lp(price, load_fc, pv_stage, plan, t_dec, soc_at)
            if rk.success:
                committed[t_dec:] = rk.x[ofs["GA"]:ofs["GA"] + m]
        ga[d] = committed
        res = settle_day(committed, load_2d[d], pv_2d[d], soc)
        x = res.x
        cc[d] = x[N:2*N]
        dd[d] = x[2*N:3*N]
        ss[d] = x[3*N:4*N]
        EE[d] = x[4*N:5*N]
        ee[d] = x[5*N:6*N]
        soc = EE[d, -1]
    return gp, ga, cc, dd, ss, EE, ee


def cost_of(gp, ga, ee_):
    rp = slice(REP0, L.N_DAY)
    under = np.maximum(gp[rp] - ga[rp], 0)
    over = np.maximum(ga[rp] - gp[rp], 0)
    return {"base": float((price * np.minimum(gp[rp], ga[rp])).sum()),
            "def": float((price * L.BETA_DEF * under).sum()),
            "exc": float((price * L.BETA_EXC * over).sum()),
            "emg": float(L.BETA_EMG * (price * ee_[rp]).sum()),
            "u": float(under.sum()), "v": float(over.sum()), "e": float(ee_[rp].sum()),
            "tot": (float((price * np.minimum(gp[rp], ga[rp])).sum())
                    + float((price * L.BETA_DEF * under).sum())
                    + float((price * L.BETA_EXC * over).sum())
                    + float(L.BETA_EMG * (price * ee_[rp]).sum()))}


t0 = time.time()
gp, ga, cc, dd, ss, EE, ee = run((1, 2, 3), verbose=True)
C = cost_of(gp, ga, ee)
print("全年滚动完成，用时 %.1f s" % (time.time() - t0))
print("基准购电 %.2f 元，违约 %.2f 元（少买 %.2f kWh）" % (C["base"], C["def"], C["u"]))
print("超额 %.2f 元（多买 %.2f kWh），紧急购电 %.2f 元（紧急 %.2f kWh）"
      % (C["exc"], C["v"], C["emg"], C["e"]))
print("报告期合计 %.2f 元" % C["tot"])

sl = slice(REP0, L.N_DAY)
dates_r = dates.iloc[sl].reset_index(drop=True)
out = os.path.join(L.RES, "result3.xlsx")
shutil.copy(os.path.join(L.ATT, "附件5", "result3.xlsx"), out)
wb = openpyxl.load_workbook(out)
for name, mat in [("计划购电量", gp[sl]), ("调整购电量", ga[sl])]:
    ws = wb[name]
    for i in range(len(dates_r)):
        for t in range(N):
            ws.cell(row=2 + i, column=2 + t, value=float(round(mat[i, t], 6)))
        ws.cell(row=2 + i, column=146, value=float(round(mat[i].sum(), 6)))
        ws.cell(row=2 + i, column=147, value=float(round((price * mat[i]).sum(), 6)))
idx_cf = wb.sheetnames.index("充放电量")
del wb["充放电量"]
ws2 = wb.create_sheet("充放电量", idx_cf)
ws2.append(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"])
for i in range(len(dates_r)):
    chg = L.block_sums(cc[sl][i])
    dis = L.block_sums(dd[sl][i])
    b0 = 2 + i * 6
    for k, (name, _, _) in enumerate(L.BLOCKS):
        r = b0 + k
        if k == 0:
            ws2.cell(row=r, column=1, value=dates_r.iloc[i].to_pydatetime())
            ws2.cell(row=r, column=5, value=_dt.time(0, 0))
            ws2.cell(row=r, column=6, value=float(round(EE[REP0 - 1, -1], 6)))
        if k == 1:
            ws2.cell(row=r, column=5, value="24:00")
            ws2.cell(row=r, column=6, value=float(round(EE[sl][i, -1], 6)))
        ws2.cell(row=r, column=2, value=name)
        ws2.cell(row=r, column=3, value=float(round(chg[k], 6)))
        ws2.cell(row=r, column=4, value=float(round(dis[k], 6)))
idx_em = wb.sheetnames.index("紧急购电量")
del wb["紧急购电量"]
ws3 = wb.create_sheet("紧急购电量", idx_em)
ws3.append(["日期", "购电时间段", "购电量"])
row = 2
for i in range(len(dates_r)):
    nz = np.where(ee[sl][i] > 1e-6)[0]
    ws3.cell(row=row, column=1, value=dates_r.iloc[i].to_pydatetime())
    if len(nz) == 0:
        ws3.cell(row=row, column=2, value="无")
        ws3.cell(row=row, column=3, value=0.0)
        row += 1
    else:
        segs, s0 = [], nz[0]
        for a, b in zip(nz[:-1], nz[1:]):
            if b != a + 1:
                segs.append((s0, a))
                s0 = b
        segs.append((s0, nz[-1]))
        for j, (a, b) in enumerate(segs):
            if j > 0:
                ws3.cell(row=row, column=1, value=None)
            ws3.cell(row=row, column=2, value="%02d:%02d-%02d:%02d"
                     % ((a*10)//60, (a*10) % 60, ((b+1)*10)//60 % 24, ((b+1)*10) % 60))
            ws3.cell(row=row, column=3, value=float(round(ee[sl][i, a:b+1].sum(), 6)))
            row += 1
wb.save(out)
print("结果已写入", out)

for ds in KEY_DATES:
    di = int(np.where(dates_r == pd.Timestamp(ds))[0][0])
    print(ds, "紧急 %.2f kWh" % ee[sl][di].sum(), "计划 %.2f" % gp[sl][di].sum(),
          "调整 %.2f" % ga[sl][di].sum(), "偏差 %+.2f" % (ga[sl][di].sum() - gp[sl][di].sum()))

# VOI 对照：固定决策时刻，只换预报版本
print("价值分解（各配置共用同一情景集）")
CASES = [
    ("A 仅0:00决策（不改签）", (), {}),
    ("B 三时刻改签，预报始终用0:00版", (1, 2, 3), {1: 0, 2: 0, 3: 0}),
    ("C1 仅6:00用0:00版预报", (1, 2, 3), {1: 0, 2: 2, 3: 3}),
    ("C2 仅12:00用6:00版预报", (1, 2, 3), {1: 1, 2: 1, 3: 3}),
    ("C3 仅18:00用12:00版预报", (1, 2, 3), {1: 1, 2: 2, 3: 2}),
]
MAIN = "C 三时刻改签，各用当期预报（本方案）"
vcost = {MAIN: C["tot"]}
for lab, act, vt in CASES:
    plan2, adj2, _, _, _, _, emg2 = run(act, vt)
    vcost[lab] = cost_of(plan2, adj2, emg2)["tot"]
    print(lab, "总费用 %.2f 元" % vcost[lab])

CA = vcost["A 仅0:00决策（不改签）"]
CB = vcost["B 三时刻改签，预报始终用0:00版"]
CC = vcost[MAIN]
print("决策机会价值 %.2f 元" % (CA - CB), "信息更新价值 %.2f 元" % (CB - CC),
      "净效应 %.2f 元" % (CA - CC))
print("分次信息价值（用上一版预报的代价）")
for tag, key in [("6:00 发布", "C1 仅6:00用0:00版预报"),
                 ("12:00 发布", "C2 仅12:00用6:00版预报"),
                 ("18:00 发布", "C3 仅18:00用12:00版预报")]:
    print(tag, "%+.2f 元" % (vcost[key] - CC))

np.savez(os.path.join(L.RES, "问3_结果.npz"),
         g_plan=gp[sl], g_adj=ga[sl], c=cc[sl], d=dd[sl], s=ss[sl], E=EE[sl], e=ee[sl],
         price=price, dates=dates_r.values.astype("datetime64[D]"),
         load=load_2d[sl], pv=pv_2d[sl], fc10=FC10[sl], bias=BIAS[sl],
         C_base=C["base"], C_def=C["def"], C_exc=C["exc"], C_emg=C["emg"], C_tot=C["tot"],
         C_A=CA, C_B=CB, C_C=CC,
         voi_info=np.array([vcost[k] for k in
                            ["C1 仅6:00用0:00版预报", "C2 仅12:00用6:00版预报",
                             "C3 仅18:00用12:00版预报"]]),
         voi_labels=np.array([k for k in vcost]),
         voi_costs=np.array([vcost[k] for k in vcost]))
print("已生成数据：results/问3_结果.npz")
