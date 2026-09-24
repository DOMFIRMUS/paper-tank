#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - statistical analysis of the window rates and of suppression.

Run after process_evaporation.py. Writes every table used by the figures and
by the formatted tables to ../results/tables/, a log (analysis_log.txt) and a
machine-readable summary (results_summary.json).

Two levels of uncertainty are kept apart throughout:

  measurement   how precisely one window's rate is known from its own 1 min
                record (AR-sieve bootstrap; HAC and block bootstrap as checks)
  between-day   how much a rate varies from one window to another under
                nominally identical conditions. The window is the experimental
                unit, so this is the uncertainty that inference about the
                cover must carry. It is estimated by a window-level model.

Sections
  1  Rate of each window: estimators and measurement uncertainty.
  2  Balance of each comparison (what else differed between the two windows).
  3  Suppression of each comparison with measurement CIs.
  4  Window-level model (primary inference): cover x regime on ln(E/X^gamma),
     pooled efficiency, regime interaction, prediction interval, minimum
     detectable difference, equivalence test; mixed model on block rates;
     profile over the elasticity gamma.
  5  Heterogeneity of the three V1 comparisons (measurement variances only).
  6  Within-window elasticity of the rate to VPD and to dE (log-OLS and PPML).
  7  Paired run of 24 August on 30 min blocks.
  8  Advection and irradiance contrasts.
  9  Energy terms.
  10 Sensitivity: window bounds, settling, reference window, thermistor,
     humidity-sensor accuracy.
  11 Micrometer against the level transducer.
  12 Night-time check of the transducer for a thermal response, and what a
     thermal coefficient of that size would do to suppression.

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
Project: MAISSER-PE, FACEPE APQ-1428-3.01/25
License: MIT
Usage:   python statistical_analysis.py
"""

import itertools
import json

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_ljungbox

import maisser_core as mc

SEED = 20260917          # fixed: every interval below is reproducible
N_BOOT = 2000
ALPHA = 0.05
EQUIV_MARGIN_PP = 10.0   # pre-specified equivalence margin for the regime effect

rng = np.random.default_rng(SEED)
LOG = []
SUMMARY = {}


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def pct_ci(x, alpha=ALPHA):
    lo, hi = np.nanpercentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def save(df, name, **kw):
    df.to_csv(mc.TABLES / name, index=False, float_format="%.6g", **kw)


def fmt(df):
    return df.to_string(index=False, float_format=lambda v: f"{v:.4g}")


con = mc.load_consolidated()
wr = pd.read_csv(mc.PROCESSED / "window_rates.csv", parse_dates=["start", "end"]).set_index("win")
subs = {w: mc.window_slice(con, s, e) for w, s, e, *_ in mc.WINDOWS}
COVER = {w: int(wr.loc[w, "cover"] == "yes") for w in wr.index}


def series(win):
    sub = subs[win]
    return mc.hours_since_start(sub.TIMESTAMP), sub.nivel_tanque_Avg.to_numpy()


def norm_value(win, key, frame=wr):
    col = mc.NORMALISERS[key]
    return 1.0 if col is None else float(frame.loc[win, col])


# ===========================================================================
log("=" * 78)
log("1. RATE OF EACH WINDOW: ESTIMATORS AND MEASUREMENT UNCERTAINTY")
log("=" * 78)
boot, mbb, rows = {}, {}, []
for win in wr.index:
    t, y = series(win)
    # Measurement uncertainty: Newey-West, 120 min bandwidth, fixed-b critical
    # value simulated for a trend slope (best-calibrated of the methods compared
    # in coverage_check.py). It is represented by normal draws whose 95 %
    # interval equals the fixed-b interval, so that suppression and ratios of
    # two windows get their intervals by simulation.
    r_ = wr.loc[win]
    boot[win] = r_.E_mm_h + r_.E_se_hac * r_.cv_fixed_b / 1.96 * rng.standard_normal(N_BOOT)
    sieve, p = mc.sieve_rates(t, y, N_BOOT, rng)
    mbb[win] = mc.mbb_rates(t, y, 60, N_BOOT, rng)
    blk = mc.block_rates(subs[win], 60)
    r = wr.loc[win]
    rows.append({
        "win": win, "E_ols": r.E_mm_h, "se_ols": r.E_se_ols,
        "se_hac120": r.E_se_hac, "cv_fixed_b": r.cv_fixed_b, "ci_lo": r.E_ci_lo, "ci_hi": r.E_ci_hi,
        "se_sieve": float(np.std(sieve, ddof=1)), "sieve_ar_order": p,
        "sieve_lo": float(np.percentile(sieve, 2.5)), "sieve_hi": float(np.percentile(sieve, 97.5)),
        "se_hac_andrews": r.E_se_hac_andrews, "andrews_lags": r.andrews_lags,
        "se_mbb60": float(np.std(mbb[win], ddof=1)),
        "E_gls": r.E_gls, "se_gls": r.E_se_gls,
        "E_block60_mean": blk.E.mean(), "E_block60_se": blk.E.std(ddof=1) / np.sqrt(len(blk)),
        "n_block60": len(blk), "DW": r.DW, "rho1": r.rho1, "acf_30min": r.acf_30min,
        "acf_60min": r.acf_60min, "n_records": r.n_records,
    })
est = pd.DataFrame(rows)
save(est, "DS01_rate_estimators.csv")
np.savez_compressed(mc.TABLES / "bootstrap_rates.npz", **boot)
log(fmt(est[["win", "E_ols", "se_ols", "se_hac120", "cv_fixed_b", "se_sieve", "se_hac_andrews", "se_mbb60",
             "E_gls", "DW", "rho1", "acf_30min", "acf_60min"]]))
log(f"\nDurbin-Watson {est.DW.min():.2f}-{est.DW.max():.2f} (median {est.DW.median():.2f}); "
    f"residual autocorrelation at 30 min {est.acf_30min.min():.2f} to {est.acf_30min.max():.2f}")
log(f"Measurement SE relative to naive OLS: sieve {np.min(est.se_sieve / est.se_ols):.1f}-"
    f"{np.max(est.se_sieve / est.se_ols):.1f}x; HAC-120 {np.min(est.se_hac120 / est.se_ols):.1f}-"
    f"{np.max(est.se_hac120 / est.se_ols):.1f}x; fixed-b critical values "
    f"{est.cv_fixed_b.min():.2f}-{est.cv_fixed_b.max():.2f}")
log(f"Largest |GLS - OLS| rate difference: {(100 * (est.E_gls / est.E_ols - 1)).abs().max():.1f} %")
SUMMARY["rates"] = {
    "DW_range": [float(est.DW.min()), float(est.DW.max())],
    "rho1_range": [float(est.rho1.min()), float(est.rho1.max())],
    "hac120_rel_halfwidth_pct_range": [float((100 * est.se_hac120 * est.cv_fixed_b / est.E_ols).min()),
                                       float((100 * est.se_hac120 * est.cv_fixed_b / est.E_ols).max())],
}

# Standard error as a function of HAC bandwidth and bootstrap block length
bw_rows = []
for win in wr.index:
    t, y = series(win)
    tr = mc.ols_trend(t, y)
    row = {"win": win}
    for L in (10, 30, 60, 120):
        row[f"se_hac_{L}"] = mc.hac_slope_se(t, tr["resid"], L)
    r2 = np.random.default_rng(SEED + 1)
    for L in (15, 30, 60, 120):
        row[f"se_mbb_{L}"] = float(np.std(mc.mbb_rates(t, y, L, 1000, r2), ddof=1))
    bw_rows.append(row)
save(pd.DataFrame(bw_rows), "DS02_se_by_bandwidth.csv")

# Cover effect within 24 August at full resolution: z of the rate difference
for lab, col in (("HAC-120", "se_hac120"), ("AR-sieve", "se_sieve")):
    s6, s7 = est.set_index("win").loc["W06", col], est.set_index("win").loc["W07", col]
    z = (wr.loc["W06", "E_mm_h"] - wr.loc["W07", "E_mm_h"]) / np.hypot(s6, s7)
    log(f"24 Aug uncovered - covered rate difference, {lab} SE: z = {z:.1f}")

# Resolution of the level channel. The logger stores the SD of the three
# 20 s samples of each minute; three samples one quantisation step q apart
# (0, 0, q) give SD = q sqrt(2)/3, which is the most frequent non-zero value.
sd_vals = con.loc[con.TIMESTAMP < mc.RECONFIGURATION, "nivel_tanque_Std"].round(3)
q_step = sd_vals[sd_vals > 0.01].mode().iloc[0] / (np.sqrt(2) / 3)
log(f"Level channel: per-scan step {q_step:.3f} mm (from the logged within-minute SD); "
    f"stored value quantised at {mc.LEVEL_QUANTUM_MM} mm (FP2)")
SUMMARY["level_scan_step_mm"] = float(q_step)


# ===========================================================================
log("\n" + "=" * 78)
log("2. BALANCE OF EACH COMPARISON")
log("=" * 78)
bal_rows = []
for cid, label, regime, cw, uw, clean in mc.COMPARISONS:
    c, u = wr.loc[cw], wr.loc[uw]
    d_vpd = c.VPD_kPa - u.VPD_kPa
    d_es = c.es_air_kPa - u.es_air_kPa
    d_ea = c.ea_kPa - u.ea_kPa
    bal_rows.append({
        "comparison": cid, "label": label,
        "clock_unc": f"{int(u.clock_start_h):02d}:{int(round(60 * (u.clock_start_h % 1))):02d}-"
                     f"{int(u.clock_end_h):02d}:{int(round(60 * (u.clock_end_h % 1))):02d}",
        "clock_cov": f"{int(c.clock_start_h):02d}:{int(round(60 * (c.clock_start_h % 1))):02d}-"
                     f"{int(c.clock_end_h):02d}:{int(round(60 * (c.clock_end_h % 1))):02d}",
        "same_day": u.start.date() == c.start.date(),
        "Rs_unc": u.Rs_W_m2, "Rs_cov": c.Rs_W_m2, "u_unc": u.u_m_s, "u_cov": c.u_m_s,
        "Tw_unc": u.Tw_C, "Tw_cov": c.Tw_C, "dTw1dt_unc": u.dTw1_dt_K_h, "dTw1dt_cov": c.dTw1_dt_K_h,
        "Ta_unc": u.Tair_C, "Ta_cov": c.Tair_C, "Ta_rise_unc": u.Tair_rise_K, "Ta_rise_cov": c.Tair_rise_K,
        "RH_unc": u.RH_pct, "RH_cov": c.RH_pct, "ea_unc": u.ea_kPa, "ea_cov": c.ea_kPa,
        "VPD_unc": u.VPD_kPa, "VPD_cov": c.VPD_kPa, "dE_unc": u.dE_kPa, "dE_cov": c.dE_kPa,
        "VPD_change_pct": 100 * d_vpd / u.VPD_kPa, "ea_change_pct": 100 * d_ea / u.ea_kPa,
        "share_of_dVPD_from_es_Ta_pct": 100 * d_es / d_vpd if abs(d_vpd) > 1e-9 else np.nan,
    })
bal = pd.DataFrame(bal_rows)
save(bal, "DS23_comparison_balance.csv")
log(fmt(bal[["label", "clock_unc", "clock_cov", "Rs_unc", "Rs_cov", "Tw_unc", "Tw_cov",
             "ea_unc", "ea_cov", "VPD_change_pct", "ea_change_pct", "share_of_dVPD_from_es_Ta_pct"]]))


# ===========================================================================
log("\n" + "=" * 78)
log("3. SUPPRESSION OF EACH COMPARISON (MEASUREMENT CIs)")
log("=" * 78)


def s_point(cw, uw, key, gamma=1.0, frame=wr):
    nc, nu = norm_value(cw, key, frame) ** gamma, norm_value(uw, key, frame) ** gamma
    return float(mc.suppression(frame.loc[cw, "E_mm_h"] / nc, frame.loc[uw, "E_mm_h"] / nu))


def s_boot(cw, uw, key, draws):
    return mc.suppression(draws[cw] / norm_value(cw, key), draws[uw] / norm_value(uw, key))


s_rows, sb_store = [], {}
for cid, label, regime, cw, uw, clean in mc.COMPARISONS:
    for key in mc.NORMALISERS:
        sb = s_boot(cw, uw, key, boot)
        sb_store[(cid, key)] = sb
        lo, hi = pct_ci(sb)
        mlo, mhi = pct_ci(s_boot(cw, uw, key, mbb))
        s_rows.append({"comparison": cid, "label": label, "regime": regime, "covered": cw,
                       "uncovered": uw, "clean": clean, "normaliser": key,
                       "S_pct": s_point(cw, uw, key), "ci_lo": lo, "ci_hi": hi,
                       "se_meas": float(np.std(sb, ddof=1)), "mbb_lo": mlo, "mbb_hi": mhi,
                       "X_unc": norm_value(uw, key), "X_cov": norm_value(cw, key)})
sup = pd.DataFrame(s_rows)
save(sup, "T3_suppression_ci.csv")
for cid, label, *_ in mc.COMPARISONS:
    parts = []
    for key in mc.NORMALISERS:
        r = sup[(sup.comparison == cid) & (sup.normaliser == key)].iloc[0]
        parts.append(f"{mc.NORMALISER_LABEL[key]:>15s} {r.S_pct:5.1f} [{r.ci_lo:5.1f}, {r.ci_hi:5.1f}]")
    log(f"{label:20s} " + " | ".join(parts))


# ===========================================================================
log("\n" + "=" * 78)
log("4. WINDOW-LEVEL MODEL (PRIMARY INFERENCE)")
log("=" * 78)
# V1 and uncovered windows W01-W07: cells uncovered-weak {W01, W03},
# covered-weak {W02}, uncovered-strong {W04, W06}, covered-strong {W05, W07}.
# Using every uncovered window avoids hand-picking one reference per pair.
V1WIN = ["W01", "W02", "W03", "W04", "W05", "W06", "W07"]
wl = wr.loc[V1WIN].copy()
wl["cov"] = [COVER[w] for w in V1WIN]
wl["strong"] = (wl.regime == "strong").astype(int)


def mdd_multiplier(df, power=0.80):
    """Exact noncentrality giving `power` for a two-sided t test at ALPHA."""
    tc = stats.t.ppf(1 - ALPHA / 2, df)
    def pw(d):
        lower = stats.nct.cdf(-tc, df, d)
        return stats.nct.sf(tc, df, d) + (0.0 if np.isnan(lower) else lower) - power
    return optimize.brentq(pw, 0.1, 12)


def window_model(key, gamma=1.0, frame=wl):
    """OLS of ln(E / X^gamma) on cover x regime, one row per window."""
    d = frame.copy()
    x = 1.0 if mc.NORMALISERS[key] is None else d[mc.NORMALISERS[key]]
    d["y"] = np.log(d.E_mm_h) - gamma * np.log(x)
    full = smf.ols("y ~ cov * strong", d).fit()
    add = smf.ols("y ~ cov + strong", d).fit()
    bc, bi = full.params["cov"], full.params["cov:strong"]
    V = full.cov_params().loc[["cov", "cov:strong"], ["cov", "cov:strong"]].to_numpy()
    df_f, df_a = full.df_resid, add.df_resid
    t_f, t_a = stats.t.ppf(0.975, df_f), stats.t.ppf(0.975, df_a)
    se_c = full.bse["cov"]
    se_cs = np.sqrt(V[0, 0] + V[1, 1] + 2 * V[0, 1])
    # S_weak - S_strong = exp(bc) (exp(bi) - 1), delta method
    diff = 100 * np.exp(bc) * (np.exp(bi) - 1)
    grad = 100 * np.array([np.exp(bc) * (np.exp(bi) - 1), np.exp(bc + bi)])
    se_diff = float(np.sqrt(grad @ V @ grad))
    t90 = stats.t.ppf(0.95, df_f)
    ba, sea = add.params["cov"], add.bse["cov"]
    sigma = np.sqrt(add.scale)
    pi = t_a * np.sqrt(2 * add.scale + sea ** 2)
    mdd = mdd_multiplier(df_f) * full.bse["cov:strong"]
    mdd_z = (1.96 + 0.8416) * full.bse["cov:strong"]
    return {
        "normaliser": key, "gamma": gamma,
        "S_weak": 100 * (1 - np.exp(bc)), "S_weak_lo": 100 * (1 - np.exp(bc + t_f * se_c)),
        "S_weak_hi": 100 * (1 - np.exp(bc - t_f * se_c)),
        "S_strong": 100 * (1 - np.exp(bc + bi)),
        "S_strong_lo": 100 * (1 - np.exp(bc + bi + t_f * se_cs)),
        "S_strong_hi": 100 * (1 - np.exp(bc + bi - t_f * se_cs)),
        "diff_pp": diff, "diff_lo": diff - t_f * se_diff, "diff_hi": diff + t_f * se_diff,
        "diff90_lo": diff - t90 * se_diff, "diff90_hi": diff + t90 * se_diff,
        "equivalent_within_margin": bool(abs(diff) + t90 * se_diff < EQUIV_MARGIN_PP),
        "interaction_log": bi, "interaction_p": float(full.pvalues["cov:strong"]),
        "df_interaction_model": df_f,
        "S_pooled": 100 * (1 - np.exp(ba)), "S_pooled_lo": 100 * (1 - np.exp(ba + t_a * sea)),
        "S_pooled_hi": 100 * (1 - np.exp(ba - t_a * sea)),
        "PI_lo": 100 * (1 - np.exp(ba + pi)), "PI_hi": 100 * (1 - np.exp(ba - pi)),
        "df_additive_model": df_a, "between_window_sd_log": sigma,
        "MDD_log_t": mdd, "MDD_pp_t": 100 * (np.exp(ba + mdd / 2) - np.exp(ba - mdd / 2)),
        "MDD_pp_z": 100 * (np.exp(ba + mdd_z / 2) - np.exp(ba - mdd_z / 2)),
        "cover_p_additive": float(add.pvalues["cov"]),
    }


wm = pd.DataFrame([window_model(k) for k in mc.NORMALISERS])
save(wm, "T4_window_level_model.csv")
for r in wm.itertuples():
    log(f"{mc.NORMALISER_LABEL[r.normaliser]:>15s}: S_weak {r.S_weak:.1f} [{r.S_weak_lo:.1f}, {r.S_weak_hi:.1f}]"
        f", S_strong {r.S_strong:.1f} [{r.S_strong_lo:.1f}, {r.S_strong_hi:.1f}]; weak-strong "
        f"{r.diff_pp:+.1f} pp [{r.diff_lo:+.1f}, {r.diff_hi:+.1f}], interaction p = {r.interaction_p:.2f}; "
        f"pooled {r.S_pooled:.1f} [{r.S_pooled_lo:.1f}, {r.S_pooled_hi:.1f}], PI [{r.PI_lo:.0f}, {r.PI_hi:.0f}]; "
        f"MDD {r.MDD_pp_t:.0f} pp (t) / {r.MDD_pp_z:.0f} pp (z); equivalence +/-{EQUIV_MARGIN_PP:.0f} pp: "
        f"{'shown' if r.equivalent_within_margin else 'not shown'}; between-window SD {r.between_window_sd_log:.3f}")
SUMMARY["window_level_model"] = wm.to_dict(orient="records")

# Between-day spread under nominally identical conditions
rep_sets = {"uncovered weak (W01, W03)": ["W01", "W03"], "uncovered strong (W04, W06)": ["W04", "W06"],
            "V1 strong (W05, W07)": ["W05", "W07"], "V2-26 strong (W09-W11)": ["W09", "W10", "W11"]}
rep_rows = []
for lab, ws in rep_sets.items():
    row = {"set": lab}
    for key in mc.NORMALISERS:
        v = np.log([wr.loc[w, "E_mm_h"] / norm_value(w, key) for w in ws])
        row[f"sd_log_{key}"] = float(np.std(v, ddof=1))
    row["median_rel_se_meas"] = float(np.median([wr.loc[w, "E_se_hac"] / wr.loc[w, "E_mm_h"] for w in ws]))
    rep_rows.append(row)
reps = pd.DataFrame(rep_rows)
save(reps, "DS03_between_day_spread.csv")
log("\nBetween-day SD of ln rate under identical nominal conditions vs measurement SE:")
log(fmt(reps))

# Mixed model on 60 min block rates (window random intercept), as a check
b60 = []
for w in V1WIN:
    b = mc.block_rates(subs[w], 60)
    b["win"], b["cov"], b["strong"] = w, COVER[w], int(wr.loc[w, "regime"] == "strong")
    b60.append(b)
b60 = pd.concat(b60, ignore_index=True)
mm_rows = []
for key in mc.NORMALISERS:
    d = b60[b60.E > 0].copy()
    x = 1.0 if mc.NORMALISERS[key] is None else d[mc.NORMALISERS[key]]
    d["y"] = np.log(d.E) - np.log(x)
    m = smf.mixedlm("y ~ cov * strong", d, groups=d.win).fit(reml=True)
    bc, bi = m.params["cov"], m.params["cov:strong"]
    mm_rows.append({"normaliser": key, "S_weak": 100 * (1 - np.exp(bc)), "S_strong": 100 * (1 - np.exp(bc + bi)),
                    "interaction_p_z": float(m.pvalues["cov:strong"]), "n_blocks": len(d),
                    "var_window": float(m.cov_re.iloc[0, 0]), "var_residual": float(m.scale)})
mm = pd.DataFrame(mm_rows)
save(mm, "DS04_mixed_model_blocks.csv")
log("\nMixed model on 60 min blocks, window random intercept:")
log(fmt(mm))

# Profile over the elasticity gamma: S = 1 - (Ec/Eu)(Xu/Xc)^gamma
prof_rows = []
for key in ("VPD", "dE"):
    for g in np.round(np.arange(0, 3.0001, 0.05), 2):
        row = {"normaliser": key, "gamma": g}
        for cid, label, regime, cw, uw, clean in mc.COMPARISONS[:3]:
            row[f"S_{cid}"] = s_point(cw, uw, key, g)
        m = window_model(key, g)
        row.update({"S_weak": m["S_weak"], "S_strong": m["S_strong"], "S_pooled": m["S_pooled"],
                    "interaction_p": m["interaction_p"], "diff_pp": m["diff_pp"]})
        vals = [row[f"S_{c}"] for c in ("C1", "C2", "C3")]
        row["range_C1_C3"] = max(vals) - min(vals)
        prof_rows.append(row)
prof = pd.DataFrame(prof_rows)
save(prof, "DS05_gamma_profile.csv")

# Minimum detectable regime difference if each of the four cells had r days,
# with the between-window SD of the interaction model
proj_rows = []
for key in mc.NORMALISERS:
    d = wl.copy()
    x = 1.0 if mc.NORMALISERS[key] is None else d[mc.NORMALISERS[key]]
    d["y"] = np.log(d.E_mm_h) - np.log(x)
    full = smf.ols("y ~ cov * strong", d).fit()
    sig, ba = np.sqrt(full.scale), smf.ols("y ~ cov + strong", d).fit().params["cov"]
    for r_ in (1, 2, 3, 4, 5, 6):
        cells = [2, 1, 2, 2] if r_ == 1 else [r_] * 4
        df_ = sum(cells) - 4
        m_ = mdd_multiplier(df_) * sig * np.sqrt(sum(1 / c for c in cells))
        proj_rows.append({"normaliser": key, "days_per_cell": "current (2,1,2,2)" if r_ == 1 else r_,
                          "df": df_, "MDD_pp": 100 * (np.exp(ba + m_ / 2) - np.exp(ba - m_ / 2))})
proj = pd.DataFrame(proj_rows)
save(proj, "DS24_mdd_projection.csv")
log("\nMinimum detectable regime difference (pp) by days per cell:")
log(proj.pivot(index="days_per_cell", columns="normaliser", values="MDD_pp").round(1).to_string())
for key in ("VPD", "dE"):
    p_ = prof[prof.normaliser == key]
    ok = p_[p_.range_C1_C3 <= 5]
    log(f"{key}: range of C1-C3 <= 5 pp only for gamma in "
        + (f"[{ok.gamma.min():.2f}, {ok.gamma.max():.2f}]" if len(ok) else "no value of gamma"))


# ===========================================================================
log("\n" + "=" * 78)
log("5. HETEROGENEITY OF THE THREE V1 COMPARISONS (MEASUREMENT VARIANCES ONLY)")
log("=" * 78)


def random_effects(theta, var, modified=True):
    """DerSimonian-Laird with the Hartung-Knapp-Sidik-Jonkman interval; with
    modified=True the HKSJ scale is floored at 1 (Rover et al. 2015)."""
    theta, var = np.asarray(theta), np.asarray(var)
    k = len(theta)
    w = 1 / var
    fe = np.sum(w * theta) / np.sum(w)
    q = float(np.sum(w * (theta - fe) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1 / (var + tau2)
    re = np.sum(ws * theta) / np.sum(ws)
    qs = np.sum(ws * (theta - re) ** 2) / (k - 1)
    var_hk = (max(1.0, qs) if modified else qs) / np.sum(ws)
    tq = stats.t.ppf(1 - ALPHA / 2, k - 1)
    return {"Q": q, "df": k - 1, "p_Q": float(stats.chi2.sf(q, k - 1)),
            "I2_pct": 100 * max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0, "tau2": tau2,
            "re": re, "re_lo": re - tq * np.sqrt(var_hk), "re_hi": re + tq * np.sqrt(var_hk)}


def to_s(lr):
    return 100 * (1 - np.exp(lr))


het_rows = []
for var_src in ("measurement draws", "AR-sieve"):
    for key in mc.NORMALISERS:
        theta, var = [], []
        for cid, label, regime, cw, uw, clean in mc.COMPARISONS[:3]:
            theta.append(np.log(1 - s_point(cw, uw, key) / 100))
            if var_src == "measurement draws":
                var.append(np.var(np.log(1 - sb_store[(cid, key)] / 100), ddof=1))
            else:
                es_ = est.set_index("win")
                var.append((es_.loc[cw, "se_sieve"] / wr.loc[cw, "E_mm_h"]) ** 2
                           + (es_.loc[uw, "se_sieve"] / wr.loc[uw, "E_mm_h"]) ** 2)
        m = random_effects(theta, var)
        het_rows.append({"variance": var_src, "normaliser": key, "Q": m["Q"], "p_Q": m["p_Q"],
                         "I2_pct": m["I2_pct"], "tau2": m["tau2"], "S_random": to_s(m["re"]),
                         "S_random_lo": to_s(m["re_hi"]), "S_random_hi": to_s(m["re_lo"])})
het = pd.DataFrame(het_rows)
save(het, "DS06_heterogeneity_measurement.csv")
log(fmt(het))


# ===========================================================================
log("\n" + "=" * 78)
log("6. WITHIN-WINDOW ELASTICITY OF THE RATE TO VPD AND TO dE")
log("=" * 78)
# If E = k X within a window (k fixed by cover, fan and lamps), the
# elasticity across the blocks of that window is 1. Window fixed effects
# remove everything constant within a window. Log-OLS needs E > 0; the
# Poisson pseudo-maximum-likelihood (PPML) estimator keeps zero rates.
blocks = {}
for M in (30, 60):
    fr = []
    for w, *_ in mc.WINDOWS:
        b = mc.block_rates(subs[w], M)
        b["win"], b["cover"] = w, COVER[w]
        fr.append(b)
    blocks[M] = pd.concat(fr, ignore_index=True)
    save(blocks[M], f"blocks_{M}min.csv")

el_rows = []
for M in (60, 30):
    b = blocks[M].copy()
    b["lnVPD"], b["lndE"] = np.log(b.VPD_kPa), np.log(b.dE_kPa)
    for subset, sel in (("all", b.index == b.index), ("uncovered", b.cover == 0), ("covered", b.cover == 1)):
        d = b[sel]
        for x in ("lnVPD", "lndE"):
            pos = d[d.E > 0].assign(lnE=lambda z: np.log(z.E))
            f = smf.ols(f"lnE ~ {x} + C(win)", pos).fit()
            g, se = f.params[x], f.bse[x]
            tq = stats.t.ppf(0.975, f.df_resid)
            # PPML only needs E[y|x] = exp(x b); its score equations hold for
            # any real y, so negative block rates (noise around a small true
            # rate) are kept. Dropping them would select on the outcome.
            pp = smf.glm(f"E ~ {x} + C(win)", d, family=sm.families.Poisson()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d.win)[0]})
            gp, sep = pp.params[x], pp.bse[x]
            nn = d[d.E >= 0]
            pp0 = smf.glm(f"E ~ {x} + C(win)", nn, family=sm.families.Poisson()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(nn.win)[0]})
            ncl = d.win.nunique()
            tqc = stats.t.ppf(0.975, ncl - 1)
            el_rows.append({
                "block_min": M, "subset": subset, "driver": x[2:],
                "gamma_ppml": gp, "ppml_lo": gp - tqc * sep, "ppml_hi": gp + tqc * sep,
                "p_ppml_eq1": float(2 * stats.t.sf(abs(gp - 1) / sep, ncl - 1)),
                "gamma_ppml_nonneg": pp0.params[x],
                "gamma_logols": g, "logols_lo": g - tq * se, "logols_hi": g + tq * se,
                "p_logols_eq1": float(2 * stats.t.sf(abs(g - 1) / se, f.df_resid)),
                "n_blocks": len(d), "n_blocks_log": len(pos), "n_windows": ncl,
                "few_clusters": ncl <= 5,
                "n_zero": int((d.E == 0).sum()), "n_negative": int((d.E < 0).sum()),
            })
el = pd.DataFrame(el_rows)
save(el, "T5_elasticity.csv")
log(fmt(el[["block_min", "subset", "driver", "gamma_ppml", "ppml_lo", "ppml_hi", "p_ppml_eq1",
            "gamma_ppml_nonneg", "gamma_logols", "logols_lo", "logols_hi", "n_blocks", "n_negative"]]))
SUMMARY["elasticity"] = el.to_dict(orient="records")

# The regime interaction at the estimated elasticity is conditional on gamma:
# show it at the point estimate and at both ends of its CI
gcond = []
for key in ("VPD", "dE"):
    r = el[(el.block_min == 60) & (el.subset == "all") & (el.driver == key)].iloc[0]
    for lab, g in (("lower 95 %", r.ppml_lo), ("estimate", r.gamma_ppml), ("upper 95 %", r.ppml_hi)):
        m = window_model(key, g)
        gcond.append({"normaliser": key, "gamma_point": lab, "gamma": g, "S_weak": m["S_weak"],
                      "S_strong": m["S_strong"], "diff_pp": m["diff_pp"], "interaction_p": m["interaction_p"],
                      "S_pooled": m["S_pooled"]})
gcond = pd.DataFrame(gcond)
save(gcond, "DS25_regime_effect_at_estimated_gamma.csv")
log("\nWindow-level regime effect at the estimated elasticity (conditional on gamma):")
log(fmt(gcond))


# ===========================================================================
log("\n" + "=" * 78)
log("7. PAIRED RUN OF 24 AUGUST: INDEPENDENT 30 MIN BLOCKS")
log("=" * 78)


def exact_two_sample(a, b):
    """Exact permutation test (difference of means) over all splits, and the
    Hodges-Lehmann shift with its distribution-free CI (Bauer 1972)."""
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.concatenate([a, b])
    n, m = len(a), len(b)
    obs = a.mean() - b.mean()
    total = pooled.sum()
    ranks = stats.rankdata(pooled)
    count, u_null = 0, []
    for comb in itertools.combinations(range(n + m), n):
        idx = list(comb)
        sa = pooled[idx].sum()
        count += abs(sa / n - (total - sa) / m) >= abs(obs) - 1e-12
        u_null.append(ranks[idx].sum() - n * (n + 1) / 2)
    n_perm = len(u_null)
    diffs = np.sort((a[:, None] - b[None, :]).ravel())
    u_null = np.sort(np.asarray(u_null))
    cdf = np.searchsorted(u_null, np.arange(n * m + 1), side="right") / n_perm
    k = int(np.max(np.where(cdf <= ALPHA / 2)[0])) + 1 if np.any(cdf <= ALPHA / 2) else 1
    return count / n_perm, float(np.median(diffs)), float(diffs[k - 1]), float(diffs[n * m - k]), n_perm


def hedges_g_exact(a, b):
    """Hedges g with the exact CI from the noncentral t (Steiger and Fouladi 1997)."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    k = np.sqrt(1 / n1 + 1 / n2)
    t_obs, df = d / k, n1 + n2 - 2
    lo = optimize.brentq(lambda nc: stats.nct.cdf(t_obs, df, nc) - 0.975, -50, 50)
    hi = optimize.brentq(lambda nc: stats.nct.cdf(t_obs, df, nc) - 0.025, -50, 50)
    return d, J * d, J * lo * k, J * hi * k


b30 = {w: mc.block_rates(subs[w], 30, align="clock") for w in ("W06", "W07")}
a, c = b30["W06"].E.to_numpy(), b30["W07"].E.to_numpy()
mw = stats.mannwhitneyu(a, c, alternative="two-sided", method="exact")
mw_asym = stats.mannwhitneyu(a, c, alternative="two-sided", method="asymptotic", use_continuity=True)
welch = stats.ttest_ind(a, c, equal_var=False)
p_perm, hl, hl_lo, hl_hi, n_perm = exact_two_sample(a, c)
d, g, g_lo, g_hi = hedges_g_exact(a, c)
trend_u = stats.spearmanr(np.arange(len(a)), a)
trend_c = stats.spearmanr(np.arange(len(c)), c)
paired = pd.DataFrame([{
    "n_uncovered": len(a), "n_covered": len(c), "median_uncovered": np.median(a), "median_covered": np.median(c),
    "MW_U": mw.statistic, "MW_p_exact": mw.pvalue, "MW_p_asymptotic_cc": mw_asym.pvalue,
    "perm_p_exact": p_perm, "n_permutations": n_perm,
    "welch_t": welch.statistic, "welch_df": welch.df, "welch_p": welch.pvalue,
    "HL_shift": hl, "HL_lo": hl_lo, "HL_hi": hl_hi, "cohen_d": d, "hedges_g": g, "g_lo": g_lo, "g_hi": g_hi,
    "trend_rho_uncovered": trend_u.statistic, "trend_p_uncovered": trend_u.pvalue,
    "trend_rho_covered": trend_c.statistic, "trend_p_covered": trend_c.pvalue,
    "ljungbox_p_uncovered": acorr_ljungbox(a - a.mean(), lags=[2]).lb_pvalue.iloc[0],
    "ljungbox_p_covered": acorr_ljungbox(c - c.mean(), lags=[2]).lb_pvalue.iloc[0],
}])
save(paired, "T6_paired_24Aug_tests.csv")
save(pd.concat([b30["W06"].assign(win="W06"), b30["W07"].assign(win="W07")]), "blocks_30min_24Aug.csv")
r = paired.iloc[0]
log(f"n = {len(a)}/{len(c)}; U = {r.MW_U:.0f}, exact p = {r.MW_p_exact:.2g} (asymptotic with continuity "
    f"correction {r.MW_p_asymptotic_cc:.2g}); permutation p = {r.perm_p_exact:.2g}; Welch p = {r.welch_p:.2g}; "
    f"HL shift {hl:.3f} [{hl_lo:.3f}, {hl_hi:.3f}]; Hedges g {g:.2f} [{g_lo:.2f}, {g_hi:.2f}] (exact)")
log(f"Trend of block rates with time: uncovered rho = {trend_u.statistic:.2f} (p = {trend_u.pvalue:.1g}), "
    f"covered rho = {trend_c.statistic:.2f} (p = {trend_c.pvalue:.2f}) -> blocks are not exchangeable; "
    f"the test compares the uncovered morning with the covered afternoon")
for col in ("VPD_kPa", "dE_kPa"):
    u_, c_ = subs["W06"][col], subs["W07"][col]
    log(f"24 Aug {col}: uncovered {u_.min():.2f}-{u_.max():.2f}, covered {c_.min():.2f}-{c_.max():.2f}")
SUMMARY["paired_24Aug"] = paired.to_dict(orient="records")


# ===========================================================================
log("\n" + "=" * 78)
log("8. ADVECTION AND IRRADIANCE")
log("=" * 78)
sd_between = {rw.normaliser: rw.between_window_sd_log for rw in wm.itertuples()}
df_between = int(wm.df_additive_model.iloc[0])
adv_rows = []
for key in mc.NORMALISERS:
    # the only advection contrast at matched (silicon-band) irradiance
    num, den = "W09", "W14"
    ratio = (wr.loc[num, "E_mm_h"] / norm_value(num, key)) / (wr.loc[den, "E_mm_h"] / norm_value(den, key))
    rb = (boot[num] / norm_value(num, key)) / (boot[den] / norm_value(den, key))
    half = stats.t.ppf(0.975, df_between) * np.sqrt(2) * sd_between[key]
    adv_rows.append({"contrast": "covered V2-26, 26 Aug (u 3.19) / 2 Sep (u 0), Rs 733/731",
                     "normaliser": key, "ratio": ratio, "meas_lo": pct_ci(rb)[0], "meas_hi": pct_ci(rb)[1],
                     "window_lo": ratio * np.exp(-half), "window_hi": ratio * np.exp(half)})
    # uncovered strong / weak: advection confounded with irradiance
    s_ = np.mean([wr.loc[w, "E_mm_h"] / norm_value(w, key) for w in ("W04", "W06")])
    w_ = np.mean([wr.loc[w, "E_mm_h"] / norm_value(w, key) for w in ("W01", "W03")])
    half2 = stats.t.ppf(0.975, df_between) * sd_between[key]
    adv_rows.append({"contrast": "uncovered strong (W04, W06) / weak (W01, W03); Rs ratio "
                                 f"{np.mean(wr.loc[['W04', 'W06'], 'Rs_W_m2']) / np.mean(wr.loc[['W01', 'W03'], 'Rs_W_m2']):.1f}",
                     "normaliser": key, "ratio": s_ / w_, "meas_lo": np.nan, "meas_hi": np.nan,
                     "window_lo": s_ / w_ * np.exp(-half2), "window_hi": s_ / w_ * np.exp(half2)})
    # plate geometry: V2 (25 Aug) against V1 (24 Aug), different days
    num, den = "W08", "W07"
    ratio = (wr.loc[num, "E_mm_h"] / norm_value(num, key)) / (wr.loc[den, "E_mm_h"] / norm_value(den, key))
    rb = (boot[num] / norm_value(num, key)) / (boot[den] / norm_value(den, key))
    adv_rows.append({"contrast": "cover V2 (25 Aug) / cover V1 (24 Aug), strong advection",
                     "normaliser": key, "ratio": ratio, "meas_lo": pct_ci(rb)[0], "meas_hi": pct_ci(rb)[1],
                     "window_lo": ratio * np.exp(-half), "window_hi": ratio * np.exp(half)})
advc = pd.DataFrame(adv_rows)
save(advc, "DS07_advection_contrasts.csv")
log(fmt(advc))


def spearman_exact(x, y):
    rho = stats.spearmanr(x, y).statistic
    ry, rx = stats.rankdata(y), stats.rankdata(x)
    perms = [stats.spearmanr(rx, np.array(p)).statistic for p in itertools.permutations(ry)]
    return rho, float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


reg_rows = []
series_sets = {"calm (u = 0), V2 26 plates": ["W12", "W13", "W14", "W15"],
               "strong advection, V2 26 plates": ["W09", "W10", "W11"]}
for label, ws in series_sets.items():
    d = wr.loc[ws]
    for xcol in ("Rs_W_m2", "VPD_kPa", "dE_kPa", "Tair_C"):
        lr = stats.linregress(d[xcol], d.E_mm_h)
        n = len(d)
        tq = stats.t.ppf(0.975, n - 2)
        rho_s, p_s = spearman_exact(d[xcol].to_numpy(), d.E_mm_h.to_numpy())
        reg_rows.append({"series": label, "x": xcol, "n": n, "slope": lr.slope,
                         "slope_lo": lr.slope - tq * lr.stderr, "slope_hi": lr.slope + tq * lr.stderr,
                         "intercept": lr.intercept, "R2": lr.rvalue ** 2, "p": lr.pvalue,
                         "spearman": rho_s, "spearman_p_exact": p_s})
reg = pd.DataFrame(reg_rows)
save(reg, "DS08_covered_rate_regressions.csv")
log(fmt(reg))
calm = wr.loc[series_sets["calm (u = 0), V2 26 plates"]]
log(f"Calm series collinearity: corr(Rs, Tair) = {np.corrcoef(calm.Rs_W_m2, calm.Tair_C)[0, 1]:.2f}, "
    f"corr(Rs, VPD) = {np.corrcoef(calm.Rs_W_m2, calm.VPD_kPa)[0, 1]:.2f}")
v2 = wr.loc[["W09", "W10", "W11", "W12", "W13", "W14", "W15"]].assign(adv=lambda z: (z.u_m_s > 1).astype(int))
inter = smf.ols("E_mm_h ~ Rs_W_m2 * adv", v2).fit()
log(f"Rs x advection interaction on the seven V2-26 windows: p = {inter.pvalues['Rs_W_m2:adv']:.2f} "
    f"(df = {inter.df_resid:.0f})")
rank = wr.loc[["W09", "W10", "W11"], ["E_mm_h", "VPD_kPa", "dE_kPa", "Rs_W_m2"]].rank()
save(rank.reset_index(), "DS09_26-28Aug_ranks.csv")
log("26-28 Aug ranks (1 = lowest):\n" + rank.to_string())
rr_rows = []
for key in mc.NORMALISERS:
    v = np.array([wr.loc[w, "E_mm_h"] / norm_value(w, key) for w in ("W09", "W10", "W11")])
    rr_rows.append({"normaliser": key, "relative_range_pct": 100 * (v.max() - v.min()) / v.mean(),
                    "cv_pct": 100 * v.std(ddof=1) / v.mean()})
save(pd.DataFrame(rr_rows), "DS10_26-28Aug_relative_range.csv")
log(fmt(pd.DataFrame(rr_rows)))
SUMMARY["advection"] = advc.to_dict(orient="records")
SUMMARY["interaction_Rs_adv_p"] = float(inter.pvalues["Rs_W_m2:adv"])


# ===========================================================================
log("\n" + "=" * 78)
log("9. ENERGY TERMS")
log("=" * 78)
# Rs is a silicon-cell reading calibrated to sunlight; most of the lamps'
# output lies beyond its 1100 nm cut-off, so Rs is an index of lamp output,
# not the radiant load. The storage term G shows that LE/Rs is not a closure.
RHO_C = 4.18e6            # volumetric heat capacity of water, J m-3 K-1
en_rows = []
for w in wr.index:
    r = wr.loc[w]
    depth_m = subs[w].nivel_tanque_Avg.mean() / 1000
    g1 = RHO_C * depth_m * r.dTw1_dt_K_h / 3600
    gm = RHO_C * depth_m * r.dTw_dt_K_h / 3600
    le_b = boot[w] / 3600 * r.lambda_MJ_kg * 1e6
    en_rows.append({"win": w, "Rs_SP_LITE": r.Rs_W_m2, "Rs_LI200X": r.Rs_licor_W_m2,
                    "LI200X_over_SPLITE": r.Rs_licor_W_m2 / r.Rs_W_m2,
                    "LE_W_m2": r.LE_W_m2, "LE_lo": pct_ci(le_b)[0], "LE_hi": pct_ci(le_b)[1],
                    "G_probe1_W_m2": g1, "G_meanprobes_W_m2": gm,
                    "LE_over_Rs_pct": r.LE_over_Rs_pct,
                    "LEplusG1_over_Rs_pct": 100 * (r.LE_W_m2 + g1) / r.Rs_W_m2})
energy = pd.DataFrame(en_rows)
save(energy, "DS11_energy_terms.csv")
log(fmt(energy))


# ===========================================================================
log("\n" + "=" * 78)
log("10. SENSITIVITY ANALYSES")
log("=" * 78)
OFFS = (0, 15, 30, 45)


def sub_rate(win, d_start, d_end):
    s0, e0 = wr.loc[win, "start"], wr.loc[win, "end"]
    sub = mc.window_slice(con, s0 + pd.Timedelta(minutes=d_start), e0 - pd.Timedelta(minutes=d_end))
    t = mc.hours_since_start(sub.TIMESTAMP)
    return mc.ols_trend(t, sub.nivel_tanque_Avg.to_numpy())["E"], sub.VPD_kPa.mean(), sub.dE_kPa.mean()


variants = {w: {(a_, b_): sub_rate(w, a_, b_) for a_ in OFFS for b_ in OFFS}
            for w in {c[3] for c in mc.COMPARISONS} | {c[4] for c in mc.COMPARISONS}}
bound_rows = []
for cid, label, regime, cw, uw, clean in mc.COMPARISONS:
    vals = {k: [] for k in mc.NORMALISERS}
    for kc, kv in itertools.product(variants[cw], variants[uw]):
        ec, vc, dc = variants[cw][kc]
        eu, vu, du = variants[uw][kv]
        vals["none"].append(mc.suppression(ec, eu))
        vals["VPD"].append(mc.suppression(ec / vc, eu / vu))
        vals["dE"].append(mc.suppression(ec / dc, eu / du))
    for key, v in vals.items():
        v = np.asarray(v, dtype=float)
        bound_rows.append({"comparison": cid, "label": label, "normaliser": key,
                           "S_baseline": s_point(cw, uw, key), "S_min": v.min(), "S_p05": np.percentile(v, 5),
                           "S_median": np.median(v), "S_p95": np.percentile(v, 95), "S_max": v.max()})
bounds = pd.DataFrame(bound_rows)
save(bounds, "DS12_window_bounds.csv")
log(fmt(bounds[["label", "normaliser", "S_baseline", "S_min", "S_max"]]))

settle_rows = []
for w, dep in mc.DEPLOYMENTS.items():
    dep = pd.Timestamp(dep)
    for k in range(2, 91, 2):
        sub = mc.window_slice(con, dep + pd.Timedelta(minutes=k), wr.loc[w, "end"])
        t = mc.hours_since_start(sub.TIMESTAMP)
        tr = mc.ols_trend(t, sub.nivel_tanque_Avg.to_numpy())
        settle_rows.append({"win": w, "minutes_after_step": k, "E_mm_h": tr["E"],
                            "se_hac120": mc.hac_slope_se(t, tr["resid"], 120),
                            "cv_fixed_b": mc.fixed_b_cv_trend(len(t), 120)})
settle = pd.DataFrame(settle_rows)
save(settle, "DS13_settling.csv")
for w, g_ in settle.groupby("win"):
    dev = 100 * (g_.E_mm_h / wr.loc[w, "E_mm_h"] - 1)
    log(f"Settling {w}: start 2-90 min after the step changes the rate by {dev.min():+.1f} to {dev.max():+.1f} %")

ref_rows = []
for regime, cw, uw in [("weak", "W02", "W01"), ("weak", "W02", "W03"), ("strong", "W05", "W04"),
                       ("strong", "W05", "W06"), ("strong", "W07", "W04"), ("strong", "W07", "W06")]:
    for key in mc.NORMALISERS:
        lo, hi = pct_ci(s_boot(cw, uw, key, boot))
        ref_rows.append({"regime": regime, "covered": cw, "uncovered": uw, "normaliser": key,
                         "S_pct": s_point(cw, uw, key), "ci_lo": lo, "ci_hi": hi})
refs = pd.DataFrame(ref_rows)
save(refs, "DS14_reference_choice.csv")

probe_rows = []
for cid, label, regime, cw, uw, clean in mc.COMPARISONS:
    for col in ("dE_kPa", "dE_p2_kPa", "dE_p3_kPa", "dE_pmean_kPa"):
        probe_rows.append({"comparison": cid, "label": label, "probe": col,
                           "S_pct": float(mc.suppression(wr.loc[cw, "E_mm_h"] / wr.loc[cw, col],
                                                         wr.loc[uw, "E_mm_h"] / wr.loc[uw, col]))})
probes = pd.DataFrame(probe_rows)
save(probes, "DS15_probe_choice.csv")
log(probes.pivot(index="label", columns="probe", values="S_pct").round(1).to_string())

# HMP45C accuracy: common offsets, and an offset applied to the strong-fan
# windows only (a regime-dependent bias such as RH_Min under fan turbulence)
acc_rows = []
cases = [(d_rh, d_ta, "common") for d_rh, d_ta in itertools.product((-3, 0, 3), (-0.3, 0, 0.3))]
cases += [(d_rh, 0.0, "strong only") for d_rh in (-3, -1, 1, 3)]
for d_rh, d_ta, mode in cases:
    vals = {}
    for w in wr.index:
        sub = subs[w]
        apply = mode == "common" or wr.loc[w, "regime"] == "strong"
        ta = sub.AirTC_Avg + (d_ta if apply else 0)
        rh = np.clip(sub.RH_Min + (d_rh if apply else 0), 0, 100)
        es = mc.saturation_vapour_pressure(ta)
        ea = es * rh / 100
        vals[w] = ((es - ea).mean(), (mc.saturation_vapour_pressure(sub.T108_C_Avg) - ea).mean())
    for cid, label, regime, cw, uw, clean in mc.COMPARISONS:
        ec, eu = wr.loc[cw, "E_mm_h"], wr.loc[uw, "E_mm_h"]
        acc_rows.append({"mode": mode, "dRH": d_rh, "dTa": d_ta, "comparison": cid,
                         "S_VPD": float(mc.suppression(ec / vals[cw][0], eu / vals[uw][0])),
                         "S_dE": float(mc.suppression(ec / vals[cw][1], eu / vals[uw][1]))})
acc = pd.DataFrame(acc_rows)
save(acc, "DS16_sensor_accuracy.csv")
log(acc.groupby(["mode", "comparison"])[["S_VPD", "S_dE"]].agg(["min", "max"]).round(1).to_string())


# ===========================================================================
log("\n" + "=" * 78)
log("11. MICROMETER AGAINST LEVEL TRANSDUCER")
log("=" * 78)
mic = pd.read_csv(mc.MICROMETER_FILE)
mic["ts"] = pd.to_datetime(mic.date + " " + mic.time)
# Intervals free of refills, deployments and removals (no step > 0.8 mm in
# the 1 min level record). The 12:20 reading of 17 August is excluded (README).
MIC_INTERVALS = [
    ("2026-08-17", "09:21", "17:21", "uncovered"), ("2026-08-18", "11:04", "17:08", "V1"),
    ("2026-08-19", "10:12", "16:11", "uncovered"), ("2026-08-20", "17:51", "18:18", "uncovered"),
    ("2026-08-21", "11:12", "15:40", "V1"), ("2026-08-26", "10:17", "18:49", "V2"),
    ("2026-08-27", "09:50", "17:04", "V2"), ("2026-08-28", "08:00", "16:53", "V2"),
    ("2026-08-31", "09:20", "17:40", "V2"), ("2026-09-01", "08:14", "13:38", "V2"),
    ("2026-09-02", "07:40", "17:20", "V2"),
]
EXCLUDED_READINGS = {("2026-08-17", "12:20")}


def mean_at(ts, col, half_width=5):
    s = con[(con.TIMESTAMP >= ts - pd.Timedelta(minutes=half_width)) &
            (con.TIMESTAMP <= ts + pd.Timedelta(minutes=half_width))]
    return s[col].mean()


# The micrometer reads surface height, which rises as the water warms and
# expands; the transducer reads pressure, i.e. mass. The correction assumes
# the water in the stilling well expands like the pan; the uncorrected rate
# is the other bound (no expansion seen in the well).
mic_rows = []
for date, t0, t1, st in MIC_INTERVALS:
    g_ = mic[(mic.date == date) & (mic.time >= t0) & (mic.time <= t1)]
    g_ = g_[[(date, tt) not in EXCLUDED_READINGS for tt in g_.time]]
    h = ((g_.ts - g_.ts.iloc[0]).dt.total_seconds() / 3600).to_numpy()
    lev = np.array([mean_at(ts, "nivel_tanque_Avg") for ts in g_.ts])
    twm = np.array([mean_at(ts, "Tw_mean_C") for ts in g_.ts])
    tw1 = np.array([mean_at(ts, "T108_C_Avg") for ts in g_.ts])
    expansion = lev[0] * (mc.water_density(twm[0]) / mc.water_density(twm) - 1)
    mic_rows.append({"date": date, "start": t0, "end": t1, "state": st, "n_readings": len(g_),
                     "duration_h": h[-1], "Tw_start": twm[0], "Tw_end": twm[-1],
                     "dTw_dt_K_h": np.polyfit(h, twm, 1)[0], "dTw1_dt_K_h": np.polyfit(h, tw1, 1)[0],
                     "expansion_mm": expansion[-1],
                     "E_micrometer": -np.polyfit(h, g_.reading_mm.to_numpy(), 1)[0],
                     "E_micrometer_mass": -np.polyfit(h, g_.reading_mm.to_numpy() - expansion, 1)[0],
                     "E_transducer": -np.polyfit(h, lev, 1)[0]})
micdf = pd.DataFrame(mic_rows)
micdf["ratio_mic_tra"] = micdf.E_micrometer / micdf.E_transducer
micdf["ratio_micmass_tra"] = micdf.E_micrometer_mass / micdf.E_transducer
save(micdf, "DS17_micrometer_intervals.csv")


def lin_ccc(x, y):
    """Lin (1989) concordance with 1/n moments."""
    x, y = np.asarray(x), np.asarray(y)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return 2 * sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2)


def agreement(d, label, col):
    x, y = d.E_transducer.to_numpy(), d[col].to_numpy()
    n = len(d)
    r2 = np.random.default_rng(SEED + 11)
    bs = []
    for _ in range(N_BOOT):
        i = r2.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(lin_ccc(x[i], y[i]))
    diff = x - y                    # transducer - micrometer throughout
    lr = np.log(y / x)
    tq = stats.t.ppf(0.975, n - 1)
    return {"set": label, "micrometer": col, "n": n, "pearson_r": stats.pearsonr(x, y).statistic,
            "ccc": lin_ccc(x, y), "ccc_lo": pct_ci(bs)[0], "ccc_hi": pct_ci(bs)[1],
            "bias_mm_h": diff.mean(), "loa_lo": diff.mean() - 1.96 * diff.std(ddof=1),
            "loa_hi": diff.mean() + 1.96 * diff.std(ddof=1),
            "geo_mean_ratio": np.exp(lr.mean()), "gmr_lo": np.exp(lr.mean() - tq * lr.std(ddof=1) / np.sqrt(n)),
            "gmr_hi": np.exp(lr.mean() + tq * lr.std(ddof=1) / np.sqrt(n)),
            "ratio_min": float((y / x).min()), "ratio_max": float((y / x).max())}


long_iv = micdf[micdf.duration_h >= 2]
agr = pd.DataFrame([
    agreement(long_iv, "intervals >= 2 h", "E_micrometer"),
    agreement(long_iv, "intervals >= 2 h", "E_micrometer_mass"),
    agreement(micdf, "all intervals incl. 27 min on 20 Aug", "E_micrometer"),
    agreement(micdf[micdf.date <= "2026-08-21"], "17-21 Aug (does not reproduce the manuscript)", "E_micrometer"),
])
save(agr, "DS18_micrometer_agreement.csv")
log(fmt(micdf[["date", "state", "duration_h", "n_readings", "dTw_dt_K_h", "expansion_mm", "E_micrometer",
               "E_micrometer_mass", "E_transducer", "ratio_micmass_tra"]]))
log(fmt(agr[["set", "micrometer", "n", "pearson_r", "ccc", "ccc_lo", "ccc_hi", "bias_mm_h",
             "loa_lo", "loa_hi", "geo_mean_ratio", "gmr_lo", "gmr_hi"]]))

# Does the disagreement track the warming of the water? Several variants,
# since the right temperature and the expansion seen in the well are unknown.
drift_rows = []
for lab, ycol, xcol in (("mass-equivalent, mean of probes", "E_micrometer_mass", "dTw_dt_K_h"),
                        ("as read (no expansion in well), probe 1", "E_micrometer", "dTw1_dt_K_h"),
                        ("as read (no expansion in well), mean of probes", "E_micrometer", "dTw_dt_K_h")):
    dd = long_iv.E_transducer - long_iv[ycol]
    lr = stats.linregress(long_iv[xcol], dd)
    tq = stats.t.ppf(0.975, len(dd) - 2)
    loo = [stats.linregress(long_iv[xcol].drop(i), dd.drop(i)).slope for i in long_iv.index]
    drift_rows.append({"variant": lab, "slope_mm_per_K": lr.slope, "lo": lr.slope - tq * lr.stderr,
                       "hi": lr.slope + tq * lr.stderr, "p": lr.pvalue, "n": len(dd),
                       "loo_min": min(loo), "loo_max": max(loo)})


# ===========================================================================
log("\n" + "=" * 78)
log("12. NIGHT-TIME CHECK OF THE TRANSDUCER FOR A THERMAL RESPONSE")
log("=" * 78)
# With lamps and fan off the pan can only lose water, yet on some nights the
# level record rises while the water cools. A pressure transducer with a
# temperature coefficient c (mm/K) reads E_app = E_true + c dT/dt with
# dT/dt < 0 at night. Model: E_app = k dE + c dT1/dt over covered nights.
cc = con[(con.TIMESTAMP >= "2026-08-17") & (con.TIMESTAMP < mc.RECONFIGURATION)].reset_index(drop=True)
dark = (cc.irradiancia_Avg < 5) & cc.nivel_tanque_Avg.notna()
brk = cc.TIMESTAMP.diff() != pd.Timedelta(minutes=1)
grp = ((dark != dark.shift()) | brk).cumsum()
night_rows = []
for _, d in cc[dark].groupby(grp[dark]):
    if len(d) < 360:
        continue
    d = d.iloc[60:-10]                                   # lamp switch-off transient
    jump = np.where(np.abs(np.diff(d.nivel_tanque_Avg.to_numpy())) > 0.8)[0]
    if len(jump):
        d = d.iloc[:max(jump[0] - 5, 0)]                 # stop before a refill/deployment
    if len(d) < 360:
        continue
    key = str(d.TIMESTAMP.iloc[0].date())
    if key not in mc.NIGHT_COVER:
        continue
    h = mc.hours_since_start(d.TIMESTAMP)
    night_rows.append({"start": d.TIMESTAMP.iloc[0], "end": d.TIMESTAMP.iloc[-1], "hours": h[-1],
                       "cover": mc.NIGHT_COVER[key],
                       "E_app_mm_h": -np.polyfit(h, d.nivel_tanque_Avg, 1)[0],
                       "dT1_dt_K_h": np.polyfit(h, d.T108_C_Avg, 1)[0],
                       "dTm_dt_K_h": np.polyfit(h, d.Tw_mean_C, 1)[0],
                       "dE_kPa": d.dE_kPa.mean(), "u_m_s": d.WS_ms_Avg.mean(),
                       "level_rise_max_mm": float(d.nivel_tanque_Avg.rolling(60).mean().diff(180).max())})
nights = pd.DataFrame(night_rows)
save(nights, "DS19_night_periods.csv")
log(fmt(nights))
night_fit = []
for subset, sel in (("V2, 26 plates (26 Aug - 2 Sep)", (nights.cover == "V2") & (nights.start >= "2026-08-26")),
                    ("V2, all", nights.cover == "V2"), ("all covered", nights.cover != "uncovered")):
    q = nights[sel]
    for tcol in ("dT1_dt_K_h", "dTm_dt_K_h"):
        f = sm.OLS(q.E_app_mm_h.to_numpy(), q[["dE_kPa", tcol]].to_numpy()).fit()
        tq = stats.t.ppf(0.975, f.df_resid)
        loo = [sm.OLS(q.E_app_mm_h.drop(i).to_numpy(), q[["dE_kPa", tcol]].drop(i).to_numpy()).fit().params[1]
               for i in q.index]
        night_fit.append({"subset": subset, "temperature": tcol, "n_nights": len(q),
                          "k_mm_h_per_kPa": f.params[0], "c_mm_per_K": f.params[1],
                          "c_lo": f.params[1] - tq * f.bse[1], "c_hi": f.params[1] + tq * f.bse[1],
                          "p_c": float(f.pvalues[1]), "c_loo_min": min(loo), "c_loo_max": max(loo)})
night_fit = pd.DataFrame(night_fit)
save(night_fit, "DS20_night_thermal_fit.csv")
log(fmt(night_fit))
drift = pd.DataFrame(drift_rows)
save(drift, "DS21_micrometer_drift_fit.csv")
log("Micrometer-based apparent thermal coefficient:")
log(fmt(drift))

# What a coefficient of that size would do to suppression (illustration, not
# a correction: c is not identified by these data)
dsens = []
for c_ in (0.0, 0.1, 0.2, 0.3, 0.4):
    for tcol in ("dTw1_dt_K_h", "dTw_dt_K_h"):
        ecorr = wr.E_mm_h - c_ * wr[tcol]
        f2 = wr.assign(E_mm_h=ecorr)
        row = {"c_mm_per_K": c_, "warming_from": "probe 1" if tcol == "dTw1_dt_K_h" else "mean of probes"}
        for cid, label, regime, cw, uw, clean in mc.COMPARISONS:
            for key in mc.NORMALISERS:
                row[f"{cid}_{key}"] = s_point(cw, uw, key, 1.0, f2)
        for key in mc.NORMALISERS:
            m = window_model(key, 1.0, wl.assign(E_mm_h=ecorr.loc[V1WIN]))
            row[f"pooled_{key}"] = m["S_pooled"]
            row[f"interaction_p_{key}"] = m["interaction_p"]
        dsens.append(row)
dsens = pd.DataFrame(dsens)
save(dsens, "DS22_thermal_drift_sensitivity.csv")
log(fmt(dsens[["c_mm_per_K", "warming_from", "C1_none", "C2_none", "C3_none", "C1_VPD", "C2_VPD",
               "C3_VPD", "pooled_VPD", "interaction_p_none", "interaction_p_VPD"]]))
SUMMARY["night_thermal_fit"] = night_fit.to_dict(orient="records")
SUMMARY["micrometer_drift_fit"] = drift.to_dict(orient="records")

pre = mean_at(pd.Timestamp("2026-08-18 10:14"), "nivel_tanque_Avg")
post = mean_at(pd.Timestamp("2026-08-18 10:50"), "nivel_tanque_Avg", 3)
log(f"\n18 Aug buoyancy step: micrometer +3.907 mm; transducer {post - pre:+.2f} mm "
    f"(10:14 +/-5 min to 10:50 +/-3 min)")

with open(mc.TABLES / "results_summary.json", "w", encoding="utf-8") as fh:
    json.dump(SUMMARY, fh, indent=1, default=float)
with open(mc.TABLES / "analysis_log.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
print(f"\nTables written to {mc.TABLES}")
