#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - Evaporation suppression by modular hexagonal floating covers
Processing and rate-estimation script (version 2).

Reads the raw TOA5 record produced by a Campbell Scientific CR1000, builds the
consolidated series and estimates the evaporation rate of every experimental
window by linear regression of water level against time.

Changes from version 1:
  * the surface-air vapour pressure difference dE = es(Tw) - ea is added
    alongside the air deficit VPD;
  * every rate is reported with an autocorrelation-consistent standard error
    (Newey-West, Bartlett kernel, 120 min bandwidth) and a 95 % confidence
    interval with fixed-b critical values simulated for a trend slope,
    together with the Durbin-Watson statistic and the residual
    autocorrelation at 1, 30 and 60 min;
  * a feasible GLS estimate with AR(1) errors is reported as a check on the
    OLS rate.

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
Project: MAISSER-PE, FACEPE APQ-1428-3.01/25
License: MIT
Usage:   python process_evaporation.py
Requires: python >= 3.9, pandas, numpy, scipy, statsmodels

Outputs (written to ../data/processed/):
  consolidated_series.csv   1 min series with VPD, dE and derived fields
  window_rates.csv          one row per experimental window
"""

import numpy as np
import pandas as pd

import maisser_core as mc


def main():
    print(f"Pan area        : {mc.PAN_AREA_M2:.4f} m2")
    print(f"Hexagon area    : {mc.HEX_AREA_M2:.6f} m2")
    print(f"27 plates       : {100 * 27 * mc.HEX_AREA_M2 / mc.PAN_AREA_M2:.1f} % coverage")
    print(f"26 plates       : {100 * 26 * mc.HEX_AREA_M2 / mc.PAN_AREA_M2:.1f} % coverage")
    print(f"Settling window : {mc.SETTLING_MIN} min excluded after deployment\n")

    raw = mc.load_raw(mc.RAW_FILES)
    con = mc.build_consolidated(raw)
    con.to_csv(mc.PROCESSED / "consolidated_series.csv", index=False)
    print(f"Consolidated series: {len(con)} records from {len(mc.RAW_FILES)} files, "
          f"{con.TIMESTAMP.min()} to {con.TIMESTAMP.max()}")
    print(f"Apparatus reconfigured on {mc.RECONFIGURATION}; later records are "
          f"retained in the series but used in no window.\n")

    rows = []
    for win, start, end, cov, ver, nplates, regime in mc.WINDOWS:
        sub = mc.window_slice(con, start, end)
        r = {"win": win, "start": sub.TIMESTAMP.iloc[0], "end": sub.TIMESTAMP.iloc[-1],
             "cover": "yes" if cov else "no", "version": ver, "n_plates": nplates,
             "coverage_pct": 100 * nplates * mc.HEX_AREA_M2 / mc.PAN_AREA_M2,
             "regime": regime}
        r.update(mc.window_rate(sub))
        r.update(mc.window_covariates(sub))
        rows.append(r)
    res = pd.DataFrame(rows)

    res["E_over_VPD"] = res.E_mm_h / res.VPD_kPa
    res["E_over_dE"] = res.E_mm_h / res.dE_kPa
    res["LE_W_m2"] = res.E_mm_h / 3600.0 * res.lambda_MJ_kg * 1e6
    res["LE_se_hac120_W_m2"] = res.E_se_hac / 3600.0 * res.lambda_MJ_kg * 1e6
    res["LE_over_Rs_pct"] = np.where(res.Rs_W_m2 > 1, 100 * res.LE_W_m2 / res.Rs_W_m2, np.nan)

    cols = ["win", "start", "end", "clock_start_h", "clock_end_h", "duration_h", "n_records",
            "cover", "version", "n_plates", "coverage_pct", "regime",
            "E_mm_h", "E_se_ols", "E_se_hac", "hac_lags", "cv_fixed_b", "E_ci_lo", "E_ci_hi",
            "E_se_hac_andrews", "andrews_lags",
            "E_gls", "E_se_gls", "R2", "resid_sd_mm", "DW", "rho1", "acf_30min", "acf_60min",
            "n_eff",
            "Rs_W_m2", "Rs_licor_W_m2", "u_m_s", "Tw_C", "Tw_min_C", "Tw_max_C", "Tw2_C", "Tw3_C",
            "dTw_dt_K_h", "dTw1_dt_K_h",
            "Tair_C", "Tair_rise_K", "RH_pct", "es_air_kPa", "ea_kPa", "VPD_kPa",
            "dE_kPa", "dE_p2_kPa", "dE_p3_kPa", "dE_pmean_kPa", "E_over_VPD", "E_over_dE",
            "lambda_MJ_kg", "LE_W_m2", "LE_se_hac120_W_m2", "LE_over_Rs_pct"]
    res = res[cols]
    res.to_csv(mc.PROCESSED / "window_rates.csv", index=False, float_format="%.6g")

    show = res[["win", "start", "cover", "version", "E_mm_h", "E_se_ols", "E_se_hac",
                "E_ci_lo", "E_ci_hi", "DW", "rho1", "acf_60min", "VPD_kPa", "dE_kPa"]]
    print(f"Window rates: {len(res)} windows\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
