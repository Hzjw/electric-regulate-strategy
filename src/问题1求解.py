"""问题1 求解：单日确定性 LP，写 result1.xlsx。"""
import os, sys, shutil
import numpy as np
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core as L

D = L.load_all(verbose=False)
price = D["typical"]["price"]
load_kw = D["typical"]["load_kw"]
pv_kw = D["typical"]["pv_kw"]

SOC0 = SOC_END = L.SOC_INIT                 # 日内闭环：末端钉回初值

sol = L.solve_day_lp(price, load_kw, pv_kw, soc_start=SOC0, soc_end=SOC_END)
assert sol["status"] == 0, "LP 未求得最优解: %s" % sol.get("message")
g, c, d, s, E = sol["g"], sol["c"], sol["d"], sol["s"], sol["E"]

# 命题1：纯 LP 不会同时充放电
simul = float(np.minimum(c, d).max())

tpl = os.path.join(L.ATT, "附件5", "result1.xlsx")
out_path = os.path.join(L.RES, "result1.xlsx")
shutil.copy(tpl, out_path)                  # 保留模板表头，只填数值
wb = openpyxl.load_workbook(out_path)

ws = wb["计划购电量"]
for i in range(L.N_T):
    ws.cell(row=2 + i, column=2, value=float(round(g[i], 6)))

# 充放电量按六大时段汇总
ws2 = wb["充放电量"]
chg = L.block_sums(c)
dis = L.block_sums(d)
for i in range(6):
    ws2.cell(row=2 + i, column=2, value=float(round(chg[i], 6)))    # 充电量
    ws2.cell(row=2 + i, column=3, value=float(round(dis[i], 6)))    # 放电量
ws2.cell(row=2, column=5, value=float(SOC0))        # 0:00 储电量
ws2.cell(row=3, column=5, value=float(E[-1]))       # 24:00 储电量
wb.save(out_path)

# 无储能基准：每段直接买净负荷
net = np.maximum(load_kw * L.DT - pv_kw * L.DT, 0.0)
cost_base = float((price * net).sum())
save = cost_base - sol["cost"]

np.savez(os.path.join(L.RES, "问1_结果.npz"), g=g, c=c, d=d, s=s, E=E,
         price=price, load_kw=load_kw, pv_kw=pv_kw, dual=sol["dual_balance"],
         cost=sol["cost"], cost_base=cost_base)

print("问题1 单日 LP")
print("同时充放电最大重叠 %.10f kWh" % simul)
print("购电 %.2f kWh / %.2f 元；弃光 %.2f kWh；SOC %.1f~%.1f"
      % (g.sum(), sol["cost"], s.sum(), E.min(), E.max()))
print("无储能基准 %.2f 元，储能节省 %.2f 元（%.2f%%）"
      % (cost_base, save, save / cost_base * 100))
print("已写 result1.xlsx 与 results/问1_结果.npz")
