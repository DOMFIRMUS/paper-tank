#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - shared definitions for the processing, statistics and figures.

Everything that more than one script needs lives here: paths, pan geometry,
experimental windows, the comparisons that define suppression, the derived
meteorological variables and the rate estimators.

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
Project: MAISSER-PE, FACEPE APQ-1428-3.01/25
License: MIT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
for _d in (PROCESSED, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

RAW_FILES = sorted(RAW.glob("*.dat"))
MICROMETER_FILE = RAW / "micrometer_readings.csv"

# Records from this date onwards describe a different sensor geometry (see README)
RECONFIGURATION = "2026-09-10"

# Pan geometry -------------------------------------------------------------
PAN_DIAMETER_M = 1.185
PAN_AREA_M2 = np.pi * (PAN_DIAMETER_M / 2) ** 2      # 1.1029 m2
HEX_SIDE_M = 0.110
HEX_AREA_M2 = 1.5 * np.sqrt(3) * HEX_SIDE_M ** 2     # 0.031437 m2

# Level record: stored as FP2 by the logger, i.e. four significant digits.
# At 200-240 mm this is a 0.1 mm quantum.
LEVEL_QUANTUM_MM = 0.1

# Experimental windows -----------------------------------------------------
# id, start, end, cover (0/1), plate version, number of plates, regime.
# Bounds exclude the post-deployment buoyancy transient (see README).
WINDOWS = [
    ("W01", "2026-08-17 09:20", "2026-08-17 17:26", 0, "-",  0,  "weak"),
    ("W02", "2026-08-18 11:20", "2026-08-18 18:20", 1, "V1", 27, "weak"),
    ("W03", "2026-08-19 10:20", "2026-08-19 15:41", 0, "-",  0,  "weak"),
    ("W04", "2026-08-20 14:35", "2026-08-20 18:24", 0, "-",  0,  "strong"),
    ("W05", "2026-08-21 11:12", "2026-08-21 15:44", 1, "V1", 27, "strong"),
    ("W06", "2026-08-24 07:57", "2026-08-24 13:00", 0, "-",  0,  "strong"),
    ("W07", "2026-08-24 13:25", "2026-08-24 18:01", 1, "V1", 27, "strong"),
    ("W08", "2026-08-25 11:25", "2026-08-25 17:18", 1, "V2", 27, "strong"),
    ("W09", "2026-08-26 09:47", "2026-08-26 18:49", 1, "V2", 26, "strong"),
    ("W10", "2026-08-27 09:22", "2026-08-27 17:04", 1, "V2", 26, "strong"),
    ("W11", "2026-08-28 07:57", "2026-08-28 16:55", 1, "V2", 26, "strong"),
    ("W12", "2026-08-31 08:45", "2026-08-31 17:37", 1, "V2", 26, "calm"),
    ("W13", "2026-09-01 08:35", "2026-09-01 13:35", 1, "V2", 26, "calm"),
    ("W14", "2026-09-02 07:56", "2026-09-02 16:30", 1, "V2", 26, "calm"),
    ("W15", "2026-09-03 08:08", "2026-09-03 16:39", 1, "V2", 26, "calm"),
]
WINDOW_COLS = ["win", "start", "end", "cover", "version", "n_plates", "regime"]

# Cover deployments that immediately precede a covered window (level step).
DEPLOYMENTS = {
    "W02": "2026-08-18 10:45",   # micrometer: 10:14 -> 10:45, 27 V1 plates
    "W07": "2026-08-24 13:04",   # transducer step 13:04-13:05
}

SETTLING_MIN = 20   # minutes of buoyancy transient excluded after deployment

# Cover state during each dark period (lamps and fan off), keyed by the date
# on which the period starts; used for the night-time check of the level
# transducer. From the window schedule and the micrometer notes.
NIGHT_COVER = {
    "2026-08-17": "uncovered", "2026-08-18": "V1", "2026-08-19": "uncovered",
    "2026-08-20": "uncovered", "2026-08-21": "V1", "2026-08-24": "V1",
    "2026-08-25": "V2", "2026-08-26": "V2", "2026-08-27": "V2", "2026-08-28": "V2",
    "2026-08-31": "V2", "2026-09-01": "V2", "2026-09-02": "V2",
}

# Suppression comparisons: id, label, regime, covered window, uncovered window,
# and whether the pair is a clean estimate of cover performance.
COMPARISONS = [
    ("C1", "18 vs 17 Aug, V1", "weak",   "W02", "W01", True),
    ("C2", "21 vs 20 Aug, V1", "strong", "W05", "W04", True),
    ("C3", "24 Aug paired, V1", "strong", "W07", "W06", True),
    ("C4", "25 vs 24 Aug, V2", "strong", "W08", "W06", False),  # thermal confounding
]

# Driving variables used to normalise a rate. "none" is the raw rate.
#   VPD      air vapour pressure deficit, es(Ta) - ea        (manuscript)
#   dE       surface-air vapour pressure difference, es(Tw) - ea  (Dalton)
NORMALISERS = {
    "none": None,
    "VPD": "VPD_kPa",
    "dE": "dE_kPa",
}
NORMALISER_LABEL = {
    "none": "raw",
    "VPD": "VPD-normalised",
    "dE": "Δe-normalised",
}

WATER_PROBES = ["T108_C_Avg", "T108_C_2_Avg", "T108_C_3_Avg"]


def windows_frame():
    return pd.DataFrame(WINDOWS, columns=WINDOW_COLS)


# Physics --------------------------------------------------------------------

def saturation_vapour_pressure(t_celsius):
    """FAO-56 Eq. 11. Returns kPa."""
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def latent_heat_vaporisation(t_celsius):
    """FAO-56. Returns MJ per kg."""
    return 2.501 - 0.002361 * t_celsius


def water_density(t_celsius):
    """Density of air-free water (kg m-3), Thiesen-type empirical formula.

    Used only to convert the micrometer (surface height) record into the
    mass-equivalent level that the hydrostatic transducer measures.
    """
    t = np.asarray(t_celsius, dtype=float)
    return 1000.0 * (1 - (t + 288.9414) / (508929.2 * (t + 68.12963)) * (t - 3.9863) ** 2)


# Loading --------------------------------------------------------------------

def load_raw(paths=RAW_FILES):
    """Read and concatenate TOA5 files.

    Row 0 is file metadata, row 1 the header, rows 2-3 units and aggregation.
    The exports overlap at their boundaries; duplicate timestamps are resolved
    keeping the first occurrence, which preserves a single continuous series.
    """
    frames = []
    for path in paths:
        d = pd.read_csv(path, skiprows=[0, 2, 3], low_memory=False,
                        na_values=["NAN", "-99.999", "-99999", "-100", "NaN"])
        d = d[d.TIMESTAMP.astype(str).str.match(r"\d{4}-\d{2}-\d{2}")].copy()
        d["TIMESTAMP"] = pd.to_datetime(d.TIMESTAMP, format="%Y-%m-%d %H:%M:%S")
        d["source_file"] = path.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).sort_values("TIMESTAMP")
    df = df.drop_duplicates(subset="TIMESTAMP", keep="first")
    return df.reset_index(drop=True)


def build_consolidated(df):
    """Add the derived fields used everywhere downstream.

    VPD is the deficit of the air, es(Ta)(1 - RH/100). dE is the difference
    that drives evaporation in Dalton's law, es(Tw) - ea, with Tw from the
    reference thermistor; the two other probes are kept for sensitivity.
    """
    df = df.copy()
    df["es_kPa"] = saturation_vapour_pressure(df["AirTC_Avg"])
    df["ea_kPa"] = df["es_kPa"] * df["RH_Min"] / 100.0
    df["VPD_kPa"] = df["es_kPa"] - df["ea_kPa"]
    df["Tw_mean_C"] = df[WATER_PROBES].mean(axis=1)
    df["lambda_MJ_kg"] = latent_heat_vaporisation(df["T108_C_Avg"])
    df["esw_kPa"] = saturation_vapour_pressure(df["T108_C_Avg"])
    df["dE_kPa"] = df["esw_kPa"] - df["ea_kPa"]
    df["dE_p2_kPa"] = saturation_vapour_pressure(df["T108_C_2_Avg"]) - df["ea_kPa"]
    df["dE_p3_kPa"] = saturation_vapour_pressure(df["T108_C_3_Avg"]) - df["ea_kPa"]
    df["dE_pmean_kPa"] = saturation_vapour_pressure(df["Tw_mean_C"]) - df["ea_kPa"]
    return df


def load_consolidated():
    """Consolidated series from disk if present, otherwise rebuilt from raw."""
    path = PROCESSED / "consolidated_series.csv"
    if path.exists():
        con = pd.read_csv(path, parse_dates=["TIMESTAMP"], low_memory=False)
        if "dE_kPa" in con.columns:
            return con
    return build_consolidated(load_raw())


def window_slice(con, start, end):
    s = con[(con.TIMESTAMP >= start) & (con.TIMESTAMP <= end)]
    return s.dropna(subset=["nivel_tanque_Avg"]).reset_index(drop=True)


# Rate estimators -----------------------------------------------------------

def hours_since_start(ts):
    return ((ts - ts.iloc[0]).dt.total_seconds() / 3600.0).to_numpy()


def ols_trend(t, y):
    """OLS of level on time. Returns a dict with the fit and its residuals."""
    X = sm.add_constant(t)
    fit = sm.OLS(y, X).fit()
    return {"fit": fit, "E": -fit.params[1], "se_ols": fit.bse[1],
            "resid": fit.resid, "r2": fit.rsquared}


def lag1_autocorrelation(x):
    x = np.asarray(x) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def andrews_bandwidth(resid):
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel.

    Returns the number of lags for the Newey-West estimator. For strongly
    persistent residuals this is far larger than the textbook n^(1/4) rules.
    """
    n = len(resid)
    rho = np.clip(lag1_autocorrelation(resid), -0.97, 0.97)
    alpha = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(np.ceil(1.1447 * (alpha * n) ** (1 / 3)))


def hac_se(fit, lags):
    """Newey-West standard error of the slope with a given number of lags."""
    rob = fit.get_robustcov_results(cov_type="HAC", maxlags=int(lags),
                                    use_correction=True)
    return float(rob.bse[1])


HAC_LAGS = 120     # Bartlett bandwidth (minutes) for the measurement uncertainty of a rate


def hac_slope_se(t, resid, lags):
    """Newey-West (Bartlett) standard error of an OLS slope on time.

    Same estimator as statsmodels get_robustcov_results(cov_type='HAC',
    use_correction=True), written out so it can be used in simulations.
    """
    tc = t - t.mean()
    s = tc * np.asarray(resid)
    n = len(s)
    lrv = np.dot(s, s)
    for j in range(1, int(lags) + 1):
        lrv += 2 * (1 - j / (lags + 1)) * np.dot(s[j:], s[:-j])
    return float(np.sqrt(lrv * n / (n - 2)) / np.sum(tc ** 2))


_FIXED_B_CACHE = {}


def fixed_b_cv_trend(n, lags, reps=5000, seed=7):
    """Fixed-b critical value (two-sided 5 %) for the HAC t statistic of the
    slope on a linear time trend, by simulation under iid errors.

    The published polynomial of Kiefer and Vogelsang (2005) applies to a mean
    or stationary regressor; for a deterministic trend the limit distribution
    differs (Bunzel and Vogelsang 2005) and its critical values are larger,
    so they are simulated here for the actual n and bandwidth.
    """
    key = (int(n), int(lags))
    if key in _FIXED_B_CACHE:
        return _FIXED_B_CACHE[key]
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    stt = np.sum(tc ** 2)
    e = rng.normal(size=(reps, n))
    e = e - e.mean(axis=1, keepdims=True)
    b = e @ tc / stt
    resid = e - b[:, None] * tc[None, :]
    s = resid * tc[None, :]
    lrv = np.sum(s * s, axis=1)
    for j in range(1, int(lags) + 1):
        lrv += 2 * (1 - j / (lags + 1)) * np.sum(s[:, j:] * s[:, :-j], axis=1)
    se = np.sqrt(lrv * n / (n - 2)) / stt
    cv = float(np.quantile(np.abs(b / se), 0.95))
    _FIXED_B_CACHE[key] = cv
    return cv


def glsar_rate(t, y):
    """Feasible GLS with AR(1) errors (iterated Cochrane-Orcutt)."""
    X = sm.add_constant(t)
    g = sm.GLSAR(y, X, rho=1).iterative_fit(maxiter=50)
    return -float(g.params[1]), float(g.bse[1]), float(g.model.rho[0])


def circular_block_indices(n, block_len, n_boot, rng):
    """Index matrix (n_boot x n) for the circular moving-block bootstrap."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def mbb_rates(t, y, block_len=60, n_boot=5000, rng=None):
    """Residual moving-block bootstrap of the OLS rate.

    The design (time) is kept fixed and residual blocks of `block_len`
    consecutive minutes are resampled, which preserves the autocorrelation
    within each block (Kunsch 1989; Lahiri 2003). Returns n_boot rates.
    """
    rng = np.random.default_rng() if rng is None else rng
    tr = ols_trend(t, y)
    tc = t - t.mean()
    stt = np.sum(tc ** 2)
    e = np.asarray(tr["resid"])
    idx = circular_block_indices(len(e), block_len, n_boot, rng)
    slope_star = -tr["E"] + (e[idx] - e[idx].mean(axis=1, keepdims=True)) @ tc / stt
    return -slope_star


def sieve_rates(t, y, n_boot=2000, rng=None, max_p=40, quantum=LEVEL_QUANTUM_MM):
    """AR-sieve bootstrap of the OLS rate (Buhlmann 1997).

    An AR(p) model (order by AIC, p <= max_p) is fitted to the residuals of
    the level regression; synthetic records are built as the fitted line plus
    simulated AR(p) noise, rounded to the storage quantum of the logger, and
    the rate is re-estimated on each. Unlike the moving-block bootstrap this
    keeps dependence longer than any block, which these records have.
    Returns (rates, p).
    """
    from scipy.signal import lfilter
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
    rng = np.random.default_rng() if rng is None else rng
    tr = ols_trend(t, y)
    e = np.asarray(tr["resid"])
    sel = ar_select_order(e, maxlag=max_p, ic="aic", trend="n")
    p = max(sel.ar_lags) if sel.ar_lags else 0
    if p:
        ar = AutoReg(e, lags=p, trend="n").fit()
        phi, sig = np.asarray(ar.params), float(np.sqrt(ar.sigma2))
    else:
        phi, sig = np.array([]), float(np.std(e))
    n, burn = len(y), 600
    eps = rng.normal(0.0, sig, size=(n_boot, n + burn))
    z = lfilter([1.0], np.r_[1.0, -phi], eps, axis=1)[:, burn:]
    ys = np.round((np.asarray(tr["fit"].fittedvalues)[None, :] + z) / quantum) * quantum
    tc = t - t.mean()
    return -(ys @ tc) / np.sum(tc ** 2), p


def block_rates(sub, minutes=60, min_fraction=0.8, align="start"):
    """Split a window into non-overlapping blocks; one OLS rate per block.

    Blocks are aligned to the window start (align="start") or to the clock
    (align="clock", as in the original script). Blocks with less than
    `min_fraction` of their records are dropped.
    """
    if align == "clock":
        key = sub.TIMESTAMP.dt.floor(f"{minutes}min")
    else:
        key = ((sub.TIMESTAMP - sub.TIMESTAMP.iloc[0]).dt.total_seconds()
               // (60 * minutes)).astype(int)
    rows = []
    for k, blk in sub.groupby(key):
        if len(blk) < minutes * min_fraction:
            continue
        h = hours_since_start(blk.TIMESTAMP)
        tr = ols_trend(h, blk.nivel_tanque_Avg.to_numpy())
        # a block whose quantised level never changes returns a slope of
        # order 1e-14 from rounding; it is a rate of exactly zero
        e_blk = 0.0 if abs(tr["E"]) < 1e-9 else tr["E"]
        rows.append({
            "block": k, "t_mid": blk.TIMESTAMP.iloc[len(blk) // 2],
            "n": len(blk), "E": e_blk, "se": tr["se_ols"],
            "VPD_kPa": blk.VPD_kPa.mean(), "dE_kPa": blk.dE_kPa.mean(),
            "Rs": blk.irradiancia_Avg.mean(), "u": blk.WS_ms_Avg.mean(),
            "Tw": blk.T108_C_Avg.mean(), "Ta": blk.AirTC_Avg.mean(),
        })
    return pd.DataFrame(rows)


def window_rate(sub):
    """All rate estimators and diagnostics for one window."""
    t = hours_since_start(sub.TIMESTAMP)
    y = sub.nivel_tanque_Avg.to_numpy()
    tr = ols_trend(t, y)
    rho = lag1_autocorrelation(tr["resid"])
    lags = andrews_bandwidth(tr["resid"])
    e_gls, se_gls, rho_gls = glsar_rate(t, y)
    n = len(y)
    L = HAC_LAGS
    se_hac = hac_slope_se(t, tr["resid"], L)
    cv = fixed_b_cv_trend(n, L)
    return {
        "n_records": n,
        "duration_h": t[-1],
        "E_mm_h": tr["E"],
        "E_se_ols": tr["se_ols"],
        "E_se_hac": se_hac,
        "hac_lags": L,
        "cv_fixed_b": cv,
        "E_ci_lo": tr["E"] - cv * se_hac,
        "E_ci_hi": tr["E"] + cv * se_hac,
        "E_se_hac_andrews": hac_se(tr["fit"], lags),
        "andrews_lags": lags,
        "E_gls": e_gls, "E_se_gls": se_gls,
        "R2": tr["r2"],
        "resid_sd_mm": float(np.std(tr["resid"], ddof=2)),
        "DW": float(durbin_watson(tr["resid"])),
        "rho1": rho,
        "acf_30min": acf_at(tr["resid"], 30),
        "acf_60min": acf_at(tr["resid"], 60),
        "n_eff": n * (1 - rho) / (1 + rho),
    }


def acf_at(x, lag):
    """Sample autocorrelation of a series at one lag."""
    x = np.asarray(x) - np.mean(x)
    return float(np.sum(x[lag:] * x[:-lag]) / np.sum(x * x))


def window_covariates(sub):
    """Mean forcing of one window."""
    rs = sub.irradiancia_Avg.mean()
    return {
        "Rs_W_m2": rs,
        "u_m_s": sub.WS_ms_Avg.mean(),
        "Tw_C": sub.T108_C_Avg.mean(),
        "Tw_min_C": sub.T108_C_Avg.min(),
        "Tw_max_C": sub.T108_C_Avg.max(),
        "Tw2_C": sub.T108_C_2_Avg.mean(),
        "Tw3_C": sub.T108_C_3_Avg.mean(),
        "dTw_dt_K_h": float(np.polyfit(hours_since_start(sub.TIMESTAMP),
                                       sub.Tw_mean_C.to_numpy(), 1)[0]),
        "dTw1_dt_K_h": float(np.polyfit(hours_since_start(sub.TIMESTAMP),
                                        sub.T108_C_Avg.to_numpy(), 1)[0]),
        "Tair_C": sub.AirTC_Avg.mean(),
        "Tair_rise_K": float(np.polyfit(hours_since_start(sub.TIMESTAMP),
                                        sub.AirTC_Avg.to_numpy(), 1)[0]
                             * hours_since_start(sub.TIMESTAMP)[-1]),
        "es_air_kPa": sub.es_kPa.mean(),
        "ea_kPa": sub.ea_kPa.mean(),
        "Rs_licor_W_m2": sub.irradiancia_licor_Avg.mean(),
        "clock_start_h": sub.TIMESTAMP.iloc[0].hour + sub.TIMESTAMP.iloc[0].minute / 60,
        "clock_end_h": sub.TIMESTAMP.iloc[-1].hour + sub.TIMESTAMP.iloc[-1].minute / 60,
        "RH_pct": sub.RH_Min.mean(),
        "VPD_kPa": sub.VPD_kPa.mean(),
        "dE_kPa": sub.dE_kPa.mean(),
        "dE_p2_kPa": sub.dE_p2_kPa.mean(),
        "dE_p3_kPa": sub.dE_p3_kPa.mean(),
        "dE_pmean_kPa": sub.dE_pmean_kPa.mean(),
        "lambda_MJ_kg": latent_heat_vaporisation(sub.T108_C_Avg.mean()),
    }


def suppression(e_cov, e_unc):
    """S (%) = 100 (1 - Ecov/Eunc). Works element-wise on arrays."""
    return 100.0 * (1.0 - np.asarray(e_cov) / np.asarray(e_unc))
