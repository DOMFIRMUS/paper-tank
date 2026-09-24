#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - Evaporation suppression by modular hexagonal floating covers
Processing and rate-estimation script.

LEGACY (version 1, as archived in the first Zenodo deposit). Kept for
provenance only: it is the script behind manuscript version 2. The current
analysis is ../process_evaporation.py and ../statistical_analysis.py. When
run from this repository it reads ../../data/raw and writes to
../../results/legacy_v1/, so it never overwrites the current outputs.

Reads the raw TOA5 record produced by a Campbell Scientific CR1000, builds the
consolidated series (adding vapour pressure deficit and latent heat flux), and
estimates the evaporation rate of every experimental window by linear
regression of water level against time.

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
Project: MAISSER-PE, FACEPE APQ-1428-3.01/25
License: MIT
Usage:   python process_evaporation.py
Requires: python >= 3.9, pandas, numpy, scipy

Outputs (written to ../../results/legacy_v1/):
  consolidated_series.csv   1 min series with VPD and derived fields
  window_rates.csv          one row per experimental window, raw and
                            VPD-normalised rates
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "data" / "raw"
OUT = HERE.parents[1] / "results" / "legacy_v1"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILES = sorted(RAW.glob("*.dat"))

# Apparatus reconfiguration: on 10 September 2026 the sensors were repositioned
# (HMP45C lowered close to the water surface, 108 probes set at 4, 12 and 20 cm,
# level transducer moved to the pan bottom) and pH and conductivity probes were
# added. Records from that date onwards are NOT comparable with the campaign and
# are excluded from the window analysis.
RECONFIGURATION = "2026-09-10"

# Pan geometry -------------------------------------------------------------
PAN_DIAMETER_M = 1.185
PAN_AREA_M2 = np.pi * (PAN_DIAMETER_M / 2) ** 2      # 1.1029 m2
HEX_SIDE_M = 0.110
HEX_AREA_M2 = 1.5 * np.sqrt(3) * HEX_SIDE_M ** 2     # 0.031437 m2

# Experimental windows -----------------------------------------------------
# start, end, cover (0/1), plate version, number of plates.
# Bounds exclude the post-deployment buoyancy transient (see README).
WINDOWS = [
    ("2026-08-17 09:20", "2026-08-17 17:26", 0, "-",  0),
    ("2026-08-18 11:20", "2026-08-18 18:20", 1, "V1", 27),
    ("2026-08-19 10:20", "2026-08-19 15:41", 0, "-",  0),
    ("2026-08-20 14:35", "2026-08-20 18:24", 0, "-",  0),
    ("2026-08-21 11:12", "2026-08-21 15:44", 1, "V1", 27),
    ("2026-08-24 07:57", "2026-08-24 13:00", 0, "-",  0),
    ("2026-08-24 13:25", "2026-08-24 18:01", 1, "V1", 27),
    ("2026-08-25 11:25", "2026-08-25 17:18", 1, "V2", 27),
    ("2026-08-26 09:47", "2026-08-26 18:49", 1, "V2", 26),
    ("2026-08-27 09:22", "2026-08-27 17:04", 1, "V2", 26),
    ("2026-08-28 07:57", "2026-08-28 16:55", 1, "V2", 26),
    ("2026-08-31 08:45", "2026-08-31 17:37", 1, "V2", 26),
    ("2026-09-01 08:35", "2026-09-01 13:35", 1, "V2", 26),
    ("2026-09-02 07:56", "2026-09-02 16:30", 1, "V2", 26),
    ("2026-09-03 08:08", "2026-09-03 16:39", 1, "V2", 26),
]

SETTLING_MIN = 20   # minutes of buoyancy transient excluded after deployment


def saturation_vapour_pressure(t_celsius):
    """FAO-56 Eq. 11. Returns kPa."""
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def latent_heat_vaporisation(t_celsius):
    """FAO-56. Returns MJ per kg."""
    return 2.501 - 0.002361 * t_celsius


def load_raw(paths):
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
    df = df.copy()
    df["es_kPa"] = saturation_vapour_pressure(df["AirTC_Avg"])
    df["VPD_kPa"] = df["es_kPa"] * (1 - df["RH_Min"] / 100.0)
    df["Tw_mean_C"] = df[["T108_C_Avg", "T108_C_2_Avg", "T108_C_3_Avg"]].mean(axis=1)
    df["lambda_MJ_kg"] = latent_heat_vaporisation(df["T108_C_Avg"])
    return df


def rate_of_window(df, start, end):
    """Evaporation rate (mm/h) by linear regression of level against time."""
    s = df[(df.TIMESTAMP >= start) & (df.TIMESTAMP <= end)]
    s = s.dropna(subset=["nivel_tanque_Avg"])
    if len(s) < 30:
        return None
    hours = (s.TIMESTAMP - s.TIMESTAMP.iloc[0]).dt.total_seconds() / 3600.0
    reg = stats.linregress(hours, s.nivel_tanque_Avg)
    rate = -reg.slope                                   # mm/h, positive = loss
    lam = latent_heat_vaporisation(s.T108_C_Avg.mean())
    le = rate / 3600.0 * lam * 1e6                      # W/m2
    rs = s.irradiancia_Avg.mean()
    vpd = s.VPD_kPa.mean()
    return {
        "start": s.TIMESTAMP.iloc[0], "end": s.TIMESTAMP.iloc[-1],
        "duration_h": round(hours.iloc[-1], 2),
        "n_records": len(s),
        "E_mm_h": round(rate, 4),
        "E_se_mm_h": round(reg.stderr, 5),
        "R2": round(reg.rvalue ** 2, 4),
        "Rs_W_m2": round(rs, 1),
        "u_m_s": round(s.WS_ms_Avg.mean(), 3),
        "Tw_C": round(s.T108_C_Avg.mean(), 2),
        "Tw_min_C": round(s.T108_C_Avg.min(), 2),
        "Tw_max_C": round(s.T108_C_Avg.max(), 2),
        "Tair_C": round(s.AirTC_Avg.mean(), 2),
        "RH_pct": round(s.RH_Min.mean(), 1),
        "VPD_kPa": round(vpd, 3),
        "E_over_VPD": round(rate / vpd, 4),
        "LE_W_m2": round(le, 1),
        "LE_over_Rs_pct": round(100 * le / rs, 1) if rs > 1 else np.nan,
    }


def independent_windows(df, start, end, minutes=30):
    """Split a window into non-overlapping blocks and return one rate per block."""
    s = df[(df.TIMESTAMP >= start) & (df.TIMESTAMP <= end)].dropna(
        subset=["nivel_tanque_Avg"])
    out = []
    for _, blk in s.groupby(pd.Grouper(key="TIMESTAMP", freq=f"{minutes}min")):
        if len(blk) < minutes * 0.8:
            continue
        h = (blk.TIMESTAMP - blk.TIMESTAMP.iloc[0]).dt.total_seconds() / 3600.0
        out.append(-stats.linregress(h, blk.nivel_tanque_Avg).slope)
    return out


def main():
    print(f"Pan area        : {PAN_AREA_M2:.4f} m2")
    print(f"Hexagon area    : {HEX_AREA_M2:.6f} m2")
    print(f"27 plates       : {100 * 27 * HEX_AREA_M2 / PAN_AREA_M2:.1f} % coverage")
    print(f"26 plates       : {100 * 26 * HEX_AREA_M2 / PAN_AREA_M2:.1f} % coverage")
    print(f"Settling window : {SETTLING_MIN} min excluded after deployment\n")

    raw = load_raw(RAW_FILES)
    con = build_consolidated(raw)
    con.to_csv(OUT / "consolidated_series.csv", index=False)
    print(f"Consolidated series: {len(con)} records from {len(RAW_FILES)} files, "
          f"{con.TIMESTAMP.min()} to {con.TIMESTAMP.max()}")
    print(f"Apparatus reconfigured on {RECONFIGURATION}; later records are "
          f"retained in the series but used in no window.\n")

    rows = []
    for start, end, cov, ver, nplates in WINDOWS:
        r = rate_of_window(con, start, end)
        if r is None:
            continue
        r["cover"] = "yes" if cov else "no"
        r["version"] = ver
        r["n_plates"] = nplates
        r["coverage_pct"] = round(100 * nplates * HEX_AREA_M2 / PAN_AREA_M2, 1)
        rows.append(r)

    res = pd.DataFrame(rows)
    cols = ["start", "end", "duration_h", "n_records", "cover", "version",
            "n_plates", "coverage_pct", "E_mm_h", "E_se_mm_h", "R2", "Rs_W_m2",
            "u_m_s", "Tw_C", "Tw_min_C", "Tw_max_C", "Tair_C", "RH_pct",
            "VPD_kPa", "E_over_VPD", "LE_W_m2", "LE_over_Rs_pct"]
    res = res[cols]
    res.to_csv(OUT / "window_rates.csv", index=False)
    print(f"Window rates      : {len(res)} windows\n")
    print(res.to_string(index=False))

    def row(day, cover=None):
        s = res[res.start.dt.strftime("%Y-%m-%d") == day]
        if cover is not None:
            s = s[s.cover == cover]
        return s.iloc[0]

    # Suppression is a ratio between two windows and is only meaningful if both
    # experienced the same atmospheric demand. It is therefore reported twice:
    # raw, and after dividing each rate by the mean VPD of its own window.
    PAIRS = [("18 vs 17 Aug, V1, weak advection", row("2026-08-18"), row("2026-08-17")),
             ("21 vs 20 Aug, V1, strong advection", row("2026-08-21"), row("2026-08-20")),
             ("24 Aug paired, V1, strong advection",
              row("2026-08-24", "yes"), row("2026-08-24", "no")),
             ("25 vs 24 Aug, V2 (thermally confounded)",
              row("2026-08-25"), row("2026-08-24", "no"))]
    print("\n=== SUPPRESSION: RAW AND VPD-NORMALISED ===")
    print(f"{'comparison':40s} {'VPD unc':>8} {'VPD cov':>8} {'S raw':>8} {'S norm':>8}")
    for lab, cov, unc in PAIRS:
        s_raw = 100 * (1 - cov.E_mm_h / unc.E_mm_h)
        s_nor = 100 * (1 - cov.E_over_VPD / unc.E_over_VPD)
        print(f"{lab:40s} {unc.VPD_kPa:8.3f} {cov.VPD_kPa:8.3f} "
              f"{s_raw:7.1f}% {s_nor:7.1f}%")
    u24, c24 = row("2026-08-24", "no").E_mm_h, row("2026-08-24", "yes").E_mm_h
    print("\n=== ADVECTION AT MATCHED IRRADIANCE, COVER V2, 26 PLATES ===")
    a, b = row("2026-08-26").E_mm_h, row("2026-09-02").E_mm_h
    print(f"26 Aug, 733 W/m2, u = 3.19 m/s : {a:.4f} mm/h")
    print(f" 2 Sep, 731 W/m2, u = 0.00 m/s : {b:.4f} mm/h")
    print(f"Wind increases the covered rate by {100 * (a / b - 1):.0f} %")

    a = independent_windows(con, "2026-08-24 07:57", "2026-08-24 13:00")
    b = independent_windows(con, "2026-08-24 13:25", "2026-08-24 18:01")
    u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = (np.mean(a) - np.mean(b)) / np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2))
    print(f"Independent 30 min windows: n={len(a)} uncovered, n={len(b)} covered")
    print(f"Mann-Whitney U={u_stat:.1f}, p={p:.3g}, Cohen d={d:.2f}")


if __name__ == "__main__":
    main()
