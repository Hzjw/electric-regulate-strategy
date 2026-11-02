"""电价不确定性的两阶段SP + CVaR 风险前沿（图28）"""
import os, sys, time, io
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core as L
import stoch as ST

N = L.N_T


def solve_two_stage_pu(price_scen, load_fc, pv_fc, load_scen, pv_scen,
                       soc_start, soc_term, beta_emg=5.0, c_cyc=0.0,
                       lam=0.0, alpha=0.95):
    """价格不确定的两阶段SP：price_scen 为 (S,n) 电价"""
    price = np.asarray(price_scen, dtype=float)
    if price.ndim == 1:
        price = np.tile(price, (load_scen.shape[0], 1))
    S, n = price.shape
    n_var = n + S * 6 * n
    i_eta = None
    i_z = None
    if lam > 0.0:
        i_eta = n_var
        i_z = n_var + 1
        n_var = n_var + 1 + S

    c_obj = np.zeros(n_var)
    # 一阶段计划量按期望电价计价
    c_obj[0:n] = price.mean(axis=0)
    for w in range(S):
        base = n + w * 6 * n
        c_obj[base + 5 * n: base + 6 * n] = (beta_emg / S) * price[w]
        if c_cyc:
            c_obj[base + 2 * n: base + 3 * n] = c_cyc / S
    if lam > 0.0:
        c_obj[i_eta] = lam
        c_obj[i_z:i_z + S] = lam / ((1.0 - alpha) * S)

    rows, cols, vals, b_eq = [], [], [], []
    rows_u, cols_u, vals_u, b_ub = [], [], [], []
    idx = np.arange(n)
    for w in range(S):
        base = n + w * 6 * n
        i_gd, i_c, i_d, i_s, i_e, i_e2 = (base + 0 * n, base + 1 * n, base + 2 * n,
                                          base + 3 * n, base + 4 * n, base + 5 * n)
        r = w * 2 * n + idx
        for blk, co in [(i_gd, 1.0), (i_e2, 1.0), (i_d, 1.0), (i_c, -1.0), (i_s, -1.0)]:
            rows.append(r)
            cols.append(blk + idx)
            vals.append(co * np.ones(n))
        b_eq.append(load_scen[w] * L.DT - pv_scen[w] * L.DT)

        r2 = w * 2 * n + n + idx
        rows.append(r2)
        cols.append(i_e + idx)
        vals.append(np.ones(n))
        rows.append(r2)
        cols.append(i_c + idx)
        vals.append(-L.ETA_C * np.ones(n))
        rows.append(r2)
        cols.append(i_d + idx)
        vals.append((1.0 / L.ETA_D) * np.ones(n))
        rows.append(r2[1:])
        cols.append(i_e + idx[1:] - 1)
        vals.append(-np.ones(n - 1))
        bb = np.zeros(n)
        bb[0] = soc_start
        b_eq.append(bb)

        rows_u.append(w * n + idx)
        cols_u.append(i_gd + idx)
        vals_u.append(np.ones(n))
        rows_u.append(w * n + idx)
        cols_u.append(idx)
        vals_u.append(-np.ones(n))
        b_ub.append(np.zeros(n))

    A_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * S * n, n_var)).tocsr()
    A_ub = coo_matrix((np.concatenate(vals_u), (np.concatenate(rows_u), np.concatenate(cols_u))),
                      shape=(S * n, n_var)).tocsr()

    if lam > 0.0:
        # CVaR 行：E_w[C_w] - eta - z_w <= 0
        cr, cc, cv, cb = [], [], [], []
        for w in range(S):
            base = n + w * 6 * n
            # A_cv 独立成矩阵，稍后 vstack
            row = w
            cr.append(np.full(n, row))
            cc.append(idx)
            cv.append(price[w])
            cr.append(np.full(n, row))
            cc.append(base + 5 * n + idx)
            cv.append(beta_emg * price[w])
            if c_cyc:
                cr.append(np.full(n, row))
                cc.append(base + 2 * n + idx)
                cv.append(c_cyc * np.ones(n))
            cr.append(np.array([row]))
            cc.append(np.array([i_eta]))
            cv.append(np.array([-1.0]))
            cr.append(np.array([row]))
            cc.append(np.array([i_z + w]))
            cv.append(np.array([-1.0]))
            cb.append(0.0)
        A_cv = coo_matrix((np.concatenate(cv), (np.concatenate(cr), np.concatenate(cc))),
                          shape=(S, n_var)).tocsr()
        from scipy.sparse import vstack as spvstack
        A_ub = spvstack([A_ub, A_cv]).tocsr()
        b_cv = np.array(cb)

    lb = np.zeros(n_var)
    ub = np.empty(n_var)
    ub[0:n] = np.inf
    for w in range(S):
        base = n + w * 6 * n
        ub[base + 0 * n: base + 1 * n] = np.inf
        ub[base + 1 * n: base + 2 * n] = L.E_MAX
        ub[base + 2 * n: base + 3 * n] = L.E_MAX
        ub[base + 3 * n: base + 4 * n] = pv_scen[w] * L.DT
        lb[base + 4 * n: base + 5 * n] = L.SOC_MIN
        ub[base + 4 * n: base + 5 * n] = L.SOC_MAX
        ub[base + 5 * n: base + 6 * n] = np.inf
    if lam > 0.0:
        lb[i_eta] = -np.inf
        ub[i_eta] = np.inf
        lb[i_z:i_z + S] = 0.0
        ub[i_z:i_z + S] = np.inf

    bub = np.concatenate(b_ub)
    if lam > 0.0:
        bub = np.concatenate([bub, b_cv])
    res = linprog(c_obj, A_eq=A_eq, b_eq=np.concatenate(b_eq),
                  A_ub=A_ub, b_ub=bub,
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        return None
    y = res.x
    d_mean = np.zeros(n)
    E_sum = np.zeros(n)
    for w in range(S):
        b = n + w * 6 * n
        d_mean += y[b + 2 * n: b + 3 * n]
        E_sum += y[b + 4 * n: b + 5 * n]
    out = {"g_plan": y[0:n].copy(), "obj": float(res.fun), "x": y, "S": S, "n": n,
           "d_mean": d_mean / S, "E_mean": E_sum / S}
    if lam > 0.0:
        out["eta"] = float(y[i_eta])
        out["z"] = y[i_z:i_z + S].copy()
        out["cvar"] = float(y[i_eta] + y[i_z:i_z + S].sum() / ((1.0 - alpha) * S))
    return out


def scenario_costs(price_scen, res, beta_emg=5.0, c_cyc=0.0):
    """逐情景计算费用 C_w"""
    S, n = res["S"], res["n"]
    y = res["x"]
    price = np.asarray(price_scen, dtype=float)
    if price.ndim == 1:
        price = np.tile(price, (S, 1))
    g_plan = y[0:n]
    C = np.zeros(S)
    for w in range(S):
        b = n + w * 6 * n
        e = y[b + 5 * n: b + 6 * n]
        d = y[b + 2 * n: b + 3 * n]
        C[w] = float(price[w] @ g_plan) + beta_emg * float(price[w] @ e) + c_cyc * float(d.sum())
    return C


def build_joint_pools(load_2d, pv_2d, price_2d, dtypes, d, k=60, eps_pv=50.0):
    """三源联合残差池：负载/光伏/电价共用候选日清单"""
    cand = [j for j in range(d) if dtypes[j] == dtypes[d]]
    if len(cand) < 5:
        cand = list(range(d))
    cand = [j for j in cand[-k:] if j >= 3]
    load_res, pv_res, price_res = [], [], []
    for j in cand:
        # 负载分解式残差：日总量比与日内形状比
        fl = ST.load_forecast(load_2d, dtypes, j, k=12)
        tot_f = fl.sum() * L.DT
        tot_a = load_2d[j].sum() * L.DT
        row = np.empty(N + 1)
        row[0] = np.clip(tot_a / max(tot_f, 1e-9), 0.5, 1.8)
        row[1:] = np.clip(load_2d[j] / np.maximum(fl, 1e-9), 0.5, 1.8)
        load_res.append(row)
        # 光伏、电价都用点预测的比值残差
        fp = ST.pv_forecast_persistence(pv_2d, dtypes, j, k=12)
        pv_res.append(np.clip((pv_2d[j] + eps_pv) / (fp + eps_pv), 0.3, 2.2))
        fq = ST.point_forecast(price_2d, dtypes, j, k=8)
        price_res.append(np.clip(price_2d[j] / np.maximum(fq, 1e-9), 0.2, 2.5))
    if not load_res:
        load_res = [np.concatenate([[1.0], np.ones(N)])]
        pv_res = [np.ones(N)]
        price_res = [np.ones(N)]
    return np.array(load_res), np.array(pv_res), np.array(price_res)


def make_scenarios(pool_l, pool_p, pool_q, S, rng, joint=True):
    """整日成块抽取；joint=True 三源同步，False 各自独立"""
    m = len(pool_l)
    blk_l = rng.integers(0, m, size=S)
    blk_p = blk_l if joint else rng.integers(0, len(pool_p), size=S)
    blk_q = blk_l if joint else rng.integers(0, len(pool_q), size=S)
    return blk_l, blk_p, blk_q


def scen_from_idx(pool_l, pool_p, pool_q, i_l, i_p, i_q, lf, pf, qf):
    """由块索引生成三类情景曲线"""
    ratio = pool_l[i_l]
    T_pt = lf.sum() * L.DT
    s_pt = lf / np.maximum(T_pt / 24.0, 1e-9)
    load_scen = (T_pt * ratio[:, 0:1] / 24.0) * (s_pt[None, :] * ratio[:, 1:])
    pv_scen = np.clip(pool_p[i_p] * pf, 0.0, None)
    price_scen = np.clip(pool_q[i_q] * qf, 0.0, None)
    return load_scen, pv_scen, price_scen


def settle_day(price, g_plan, load_kw, pv_kw, soc_start):
    """按已承诺计划量做的当日实时调度"""
    n = N
    IG = {k: i * n for i, k in enumerate(["GD", "C", "D", "S", "E", "E2"])}
    idx = np.arange(n)
    c_obj = np.zeros(6 * n)
    c_obj[IG["E2"] + idx] = L.BETA_EMG * price
    c_obj[IG["C"] + idx] = L.THRU_PEN
    c_obj[IG["D"] + idx] = L.THRU_PEN
    eq_r, eq_c, eq_v, b_eq = [], [], [], []
    for name, co in [("GD", 1.0), ("E2", 1.0), ("D", 1.0), ("C", -1.0), ("S", -1.0)]:
        eq_r.append(idx)
        eq_c.append(IG[name] + idx)
        eq_v.append(co * np.ones(n))
    b_eq.append(load_kw * L.DT - pv_kw * L.DT)
    r_s = n + idx
    eq_r.append(r_s)
    eq_c.append(IG["E"] + idx)
    eq_v.append(np.ones(n))
    eq_r.append(r_s)
    eq_c.append(IG["C"] + idx)
    eq_v.append(-L.ETA_C * np.ones(n))
    eq_r.append(r_s)
    eq_c.append(IG["D"] + idx)
    eq_v.append((1.0 / L.ETA_D) * np.ones(n))
    eq_r.append(r_s[1:])
    eq_c.append(IG["E"] + idx[1:] - 1)
    eq_v.append(-np.ones(n - 1))
    bb = np.zeros(n)
    bb[0] = soc_start
    b_eq.append(bb)
    A_eq = coo_matrix((np.concatenate(eq_v), (np.concatenate(eq_r), np.concatenate(eq_c))),
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
    return linprog(c_obj, A_eq=A_eq, b_eq=np.concatenate(b_eq),
                   bounds=list(zip(lb, ub)), method="highs")


def cvar_of(C, alpha):
    """经验 CVaR：最坏 (1-alpha) 比例情景的平均费用"""
    C_sorted = np.sort(np.asarray(C, dtype=float))
    k = max(1, int(round((1.0 - alpha) * len(C_sorted))))
    return float(C_sorted[-k:].mean())


def run_case(price_rt, load_2d, pv_2d, DTYPE, mode="unc", lam=0.0, alpha=0.95,
             joint=True, pools="aligned", S=20, seed=42, dmax=None, verbose=True):
    """mode/pools/joint 控制电价口径与抽样方式"""
    rng = np.random.default_rng(seed)
    n_day = L.N_DAY if dmax is None else dmax
    g_plan_daily = np.zeros((L.N_DAY, N))
    emg_daily = np.zeros((L.N_DAY, N))
    soc_daily = np.zeros((L.N_DAY, N))
    cvar_list, ecost_list, lp_cvar_list = [], [], []
    t0 = time.time()
    soc = L.SOC_INIT
    for d in range(n_day):
        price_mat = None
        if d >= 5:
            lf = ST.load_forecast(load_2d, DTYPE, d)
            pf = ST.pv_forecast_persistence(pv_2d, DTYPE, d)
            if pools == "lib":
                if mode != "known":
                    raise ValueError("pools='lib' 只用于电价已知的回归复现")
                load_pool_lib = ST.load_residual_pool(load_2d, DTYPE, d)
                pv_pool_lib = ST.pv_residual_pool_persist(pv_2d, DTYPE, d)
                load_scen = ST.load_scenarios(load_pool_lib, lf, S, rng)
                pv_scen = np.clip(ST.sample_scenarios(pv_pool_lib, S, rng) * pf, 0.0, None)
            else:
                qf = ST.point_forecast(price_rt, DTYPE, d, k=8)
                load_pool, pv_pool, price_pool = build_joint_pools(load_2d, pv_2d, price_rt,
                                                                   DTYPE, d)
                blk_l, blk_p, blk_q = make_scenarios(load_pool, pv_pool, price_pool, S, rng,
                                                     joint=joint)
                load_scen, pv_scen, price_scen = scen_from_idx(load_pool, pv_pool, price_pool,
                                                               blk_l, blk_p, blk_q, lf, pf, qf)
                pv_scen = np.clip(pv_scen, 0.0, None)
            price_mat = price_rt[d] if mode == "known" else price_scen
            # 一阶段用当日实际起始储电量
            r = solve_two_stage_pu(price_mat, lf, pf, load_scen, pv_scen, soc, None,
                                   lam=lam, alpha=alpha)
            plan = r["g_plan"] if r is not None else np.maximum(lf * L.DT - pf * L.DT, 0)
            if r is not None:
                costs = scenario_costs(price_mat, r)
                cvar_list.append(cvar_of(costs, alpha))
                ecost_list.append(float(costs.mean()))
                if lam > 0:
                    lp_cvar_list.append(r["cvar"])
        else:
            plan = np.maximum(load_2d[d] * L.DT - pv_2d[d] * L.DT, 0)
        g_plan_daily[d] = plan
        settle = settle_day(price_rt[d], plan, load_2d[d], pv_2d[d], soc)
        if not settle.success:
            raise RuntimeError("day %d 结算失败" % d)
        y = settle.x
        emg_daily[d] = y[5 * N:6 * N]
        soc_daily[d] = y[4 * N:5 * N]
        soc = soc_daily[d, -1]
        if verbose and (d + 1) % 100 == 0:
            print("   %s lam=%.2f ... %3d/%d 天  %.0fs"
                  % (mode, lam, d + 1, n_day, time.time() - t0))
    sl = slice(31, L.N_DAY)
    cost = float((price_rt[sl] * g_plan_daily[sl]).sum()
                 + L.BETA_EMG * (price_rt[sl] * emg_daily[sl]).sum())
    return {"gp": g_plan_daily, "e": emg_daily, "E": soc_daily, "cost": cost,
            "emg_kwh": float(emg_daily[sl].sum()), "sec": time.time() - t0,
            "cvar_mean": float(np.mean(cvar_list)) if cvar_list else None,
            "ec_mean": float(np.mean(ecost_list)) if ecost_list else None,
            "lp_cvar_mean": float(np.mean(lp_cvar_list)) if lp_cvar_list else None,
            "gp_total": float(g_plan_daily[sl].sum())}


CASES = [
    ("known", "lib",     False, "A 电价已知·独立抽样（既有 4-2 回归基准）"),
    ("known", "aligned", True,  "B 电价已知·三源联合抽样"),
    ("unc",   "aligned", False, "C 电价不确定·独立抽样"),
    ("unc",   "aligned", True,  "D 电价不确定·三源联合抽样（本文新增主模型）"),
]
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
ALPHA = 0.95
S_SCEN = 20


def selftest():
    """lam=0 时复现 stoch.solve_two_stage"""
    data = L.load_all(verbose=False)
    load_2d = data["actual"]["load_kw"]
    pv_2d = data["actual"]["pv_kw"]
    price_rt = data["price_rt"]["price"]
    day_types = ST.day_type(data["actual"]["dates"])
    rng = np.random.default_rng(42)
    d_obj = 0.0
    d_plan = 0.0
    for d in [31, 60, 120, 200, 300, 364]:
        lf = ST.load_forecast(load_2d, day_types, d)
        pf = ST.pv_forecast_persistence(pv_2d, day_types, d)
        load_pool_lib = ST.load_residual_pool(load_2d, day_types, d)
        pv_pool_lib = ST.pv_residual_pool_persist(pv_2d, day_types, d)
        load_scen = ST.load_scenarios(load_pool_lib, lf, S_SCEN, rng)
        pv_scen = np.clip(ST.sample_scenarios(pv_pool_lib, S_SCEN, rng) * pf, 0, None)
        a = ST.solve_two_stage(price_rt[d], lf, pf, load_scen, pv_scen, L.SOC_INIT, None)
        b = solve_two_stage_pu(price_rt[d], lf, pf, load_scen, pv_scen, L.SOC_INIT, None, lam=0.0)
        d_obj = max(d_obj, abs(a["obj"] - b["obj"]))
        d_plan = max(d_plan, float(np.max(np.abs(a["g_plan"] - b["g_plan"]))))
    print("自检 单日 LP 级：与 stoch.solve_two_stage 最大目标差 %.2e，最大计划量差 %.2e -> %s"
          % (d_obj, d_plan, "通过" if (d_obj < 1e-6 and d_plan < 1e-6) else "未通过"))
    return d_obj, d_plan


def main():
    selftest()

    data = L.load_all(verbose=False)
    price_rt = data["price_rt"]["price"]
    load_2d = data["actual"]["load_kw"]
    pv_2d = data["actual"]["pv_kw"]
    day_types = ST.day_type(data["actual"]["dates"])

    cases = {}
    for mode, pool, joint_flag, label in CASES:
        r = run_case(price_rt, load_2d, pv_2d, day_types, mode=mode, lam=0.0, joint=joint_flag,
                     pools=pool, S=S_SCEN, verbose=False)
        cases[(mode, pool, joint_flag)] = r
        print("%-40s  %14.2f 元   紧急 %11.1f kWh   日均CVaR95 %10.1f 元"
              % (label, r["cost"], r["emg_kwh"], r["cvar_mean"]))

    A = cases[("known", "lib", False)]
    B = cases[("known", "aligned", True)]
    C = cases[("unc", "aligned", False)]
    D = cases[("unc", "aligned", True)]
    print("回归 A 口径复现既有 4-2：%.2f 元，论文交付值 14 274 280.36 元，差 %+.2f 元"
          % (A["cost"], A["cost"] - 14274280.36))
    print("抽样方式影响 B - A = %+.2f 元 (%+.2f%%)"
          % (B["cost"] - A["cost"], 100 * (B["cost"] / A["cost"] - 1)))
    print("电价不确定性代价 D - B = %+.2f 元 (%+.2f%%)"
          % (D["cost"] - B["cost"], 100 * (D["cost"] / B["cost"] - 1)))
    print("价格口径影响（独立抽样下）C - A = %+.2f 元 (%+.2f%%)"
          % (C["cost"] - A["cost"], 100 * (C["cost"] / A["cost"] - 1)))

    print("CVaR 风险前沿（在 D 口径上扫描，alpha=%.2f，S=%d）" % (ALPHA, S_SCEN))
    scan = []
    for lam in LAMBDAS:
        r = run_case(price_rt, load_2d, pv_2d, day_types, mode="unc", lam=lam, alpha=ALPHA,
                     joint=True, pools="aligned", S=S_SCEN, verbose=False)
        scan.append(r)
        tag = ""
        if lam > 0 and r["lp_cvar_mean"] is not None:
            tag = "   [LP内CVaR %.1f，与经验值差 %.2e]" % (r["lp_cvar_mean"],
                                                          abs(r["lp_cvar_mean"] - r["cvar_mean"]))
        print("  lambda=%-5.2f  报告期 %14.2f 元   日均CVaR95 %10.1f 元   计划量 %11.1f kWh%s"
              % (lam, r["cost"], r["cvar_mean"], r["gp_total"], tag))

    dgp = (scan[3]["gp"] - scan[0]["gp"])[31:].sum(axis=1)      # λ=0.2 与 0 的申报量之差
    n_up = int((dgp > 1.0).sum())
    n_dn = int((dgp < -1.0).sum())
    print("相对风险中性（lambda=0），lambda=0.2 时：申报量上调 %d 天、下调 %d 天（报告期 334 天）"
          % (n_up, n_dn))
    print("  日均申报量 %.1f -> %.1f kWh (%+.2f%%)"
          % (scan[0]["gp_total"] / 334.0, scan[3]["gp_total"] / 334.0,
             100 * (scan[3]["gp_total"] / scan[0]["gp_total"] - 1)))

    np.savez(os.path.join(L.RES, "电价不确定性与CVaR_结果.npz"),
             cost_A=A["cost"], cost_B=B["cost"], cost_C=C["cost"], cost_D=D["cost"],
             emg_A=A["emg_kwh"], emg_D=D["emg_kwh"],
             cvar_A=A["cvar_mean"], cvar_B=B["cvar_mean"],
             cvar_C=C["cvar_mean"], cvar_D=D["cvar_mean"],
             lam=np.array(LAMBDAS),
             cost_scan=np.array([r["cost"] for r in scan]),
             cvar_scan=np.array([r["cvar_mean"] for r in scan]),
             gp_scan=np.array([r["gp_total"] for r in scan]),
             emg_scan=np.array([r["emg_kwh"] for r in scan]),
             gp_lam0=scan[0]["gp"], gp_lam02=scan[3]["gp"],
             dgp_daily=dgp, n_up=n_up, n_dn=n_dn, alpha=ALPHA, S=S_SCEN)
    print("已保存 results/电价不确定性与CVaR_结果.npz")
    return True


# 入口置于文件末尾
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
