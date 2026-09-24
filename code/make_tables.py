#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - formatted tables.

Run after statistical_analysis.py and coverage_check.py. Writes
  ../results/tables/MAISSER-PE_tables.docx   Tables 2-6 (main text) and S1-S12
                                     (supplement): caption above, three rules,
                                     notes below; each caption names its data file
  ../results/tables/MAISSER-PE_tables.xlsx   every CSV table, one sheet each, in
                                     natural order, plus a README sheet

Numbering: T3-T6 files correspond to Tables 3-6; DS## files are data tables
(Data S##) that feed the formatted supplementary tables or the figures.
Every number is read from the CSV files; nothing is typed by hand.

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
Project: MAISSER-PE, FACEPE APQ-1428-3.01/25
License: MIT
Usage:   python make_tables.py
"""

import re

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Mm

import maisser_core as mc

T = mc.TABLES


def rd(name, **kw):
    return pd.read_csv(T / name, **kw)


wr = pd.read_csv(mc.PROCESSED / "window_rates.csv", parse_dates=["start", "end"]).set_index("win")
est = rd("DS01_rate_estimators.csv").set_index("win")
bal = rd("DS23_comparison_balance.csv")
sup = rd("T3_suppression_ci.csv")
wm = rd("T4_window_level_model.csv").set_index("normaliser")
el = rd("T5_elasticity.csv")
paired = rd("T6_paired_24Aug_tests.csv").iloc[0]
mm = rd("DS04_mixed_model_blocks.csv").set_index("normaliser")
het = rd("DS06_heterogeneity_measurement.csv")
advc = rd("DS07_advection_contrasts.csv")
reg = rd("DS08_covered_rate_regressions.csv")
energy = rd("DS11_energy_terms.csv").set_index("win")
bounds = rd("DS12_window_bounds.csv")
refs = rd("DS14_reference_choice.csv")
probes = rd("DS15_probe_choice.csv")
acc = rd("DS16_sensor_accuracy.csv")
micdf = rd("DS17_micrometer_intervals.csv")
agr = rd("DS18_micrometer_agreement.csv")
nights = rd("DS19_night_periods.csv", parse_dates=["start", "end"])
night_fit = rd("DS20_night_thermal_fit.csv")
drift = rd("DS21_micrometer_drift_fit.csv")
dsens = rd("DS22_thermal_drift_sensitivity.csv")
proj = rd("DS24_mdd_projection.csv")
gcond = rd("DS25_regime_effect_at_estimated_gamma.csv")
cover = rd("DS00_ci_coverage.csv") if (T / "DS00_ci_coverage.csv").exists() else None

NLAB = {"none": "Raw", "VPD": "VPD-normalised", "dE": "Δe-normalised"}
CLAB = {"C1": "18 vs 17 Aug", "C2": "21 vs 20 Aug", "C3": "24 Aug, paired", "C4": "25 vs 24 Aug"}
CREG = {"C1": "weak", "C2": "strong", "C3": "strong", "C4": "strong"}
CIDS = ("C1", "C2", "C3", "C4")


def dlab(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.day} {ts.strftime('%b')}"


def wlab(win):
    return dlab(wr.loc[win, "start"])


def cover_lab(win):
    v = wr.loc[win, "version"]
    return "none" if v == "-" else v


def f(x, d=1):
    if pd.isna(x):
        return "–"
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def fci(x, lo, hi, d=1):
    return f"{f(x, d)} [{f(lo, d)}, {f(hi, d)}]"


def fp(p):
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"{m} × 10{sup_digits(int(e))}"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def sup_digits(n):
    table = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(n).translate(table)


def txt(s):
    """Typography for labels coming from the CSVs."""
    return (s.replace(">=", "≥").replace(" - ", "–").replace("17-21", "17–21")
             .replace("W04, W06", "20, 24 Aug").replace("W01, W03", "17, 19 Aug"))


# --- docx helpers ------------------------------------------------------------

def set_cell_border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, val in edges.items():
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(val))
        e.set(qn("w:color"), "000000")
        borders.append(e)


def add_table(doc, caption, source, header, rows, notes, font=8, bold_rows=()):
    p = doc.add_paragraph()
    head, _, tail = caption.partition(". ")
    r = p.add_run(head + ". ")
    r.bold = True
    r.font.size = Pt(9)
    p.add_run(tail).font.size = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(font)
        set_cell_border(c, top=8, bottom=4)
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = ""
            run = c.paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
            run.bold = (i - 1) in bold_rows
            if j > 0:
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if i == len(rows):
                set_cell_border(c, bottom=8)
    for n in list(notes) + [f"Data file: {source}."]:
        q = doc.add_paragraph()
        q.paragraph_format.space_before = Pt(2)
        q.paragraph_format.space_after = Pt(0)
        q.add_run(n).font.size = Pt(7.5)
    doc.add_paragraph()


def section(doc, landscape):
    s = doc.add_section()
    s.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    s.page_width, s.page_height = (Mm(297), Mm(210)) if landscape else (Mm(210), Mm(297))
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(s, m, Mm(15 if landscape else 20))


def coverage_note():
    if cover is None:
        return ""
    med = cover.filter(like="cov_").median()
    lo = cover.cov_hac120_fixedb.min()
    return (f"In simulated records with the dependence of the real residuals (coverage_check.py) the nominal 95 % "
            f"interval covered the true rate in a median {med.cov_hac120_fixedb:.2f} of cases (minimum {lo:.2f}); "
            f"the intervals are therefore somewhat narrow.")


# --- main tables -------------------------------------------------------------

def table2(doc):
    header = ["Window", "Date", "Cover", "Cov. (%)", "Time", "E (mm h⁻¹) [95 % CI]", "Rs (W m⁻²)", "u (m s⁻¹)",
              "Tw (°C)", "Ta (°C)", "ea (kPa)", "VPD (kPa)", "Δe (kPa)", "E/VPD", "E/Δe"]
    rows = []
    for w in wr.index:
        r = wr.loc[w]
        rows.append([w, wlab(w), cover_lab(w), f(r.coverage_pct), f"{r.start.strftime('%H:%M')}–{r.end.strftime('%H:%M')}",
                     fci(r.E_mm_h, est.loc[w, "ci_lo"], est.loc[w, "ci_hi"], 3),
                     f(r.Rs_W_m2, 0), f(r.u_m_s, 2), f(r.Tw_C, 1), f(r.Tair_C, 1), f(r.ea_kPa, 2),
                     f(r.VPD_kPa, 2), f(r.dE_kPa, 2), f(r.E_over_VPD, 3), f(r.E_over_dE, 3)])
    add_table(doc, "Table 2. Evaporation rates and forcing for the fifteen experimental windows.",
              "data/processed/window_rates.csv; DS01_rate_estimators.csv", header, rows,
              ["E: negative OLS slope of the 1 min level record. 95 % CI: Newey–West standard error (Bartlett kernel, "
               "120 min bandwidth) with a fixed-b critical value simulated for the slope of a time trend. It describes the "
               "measurement precision of one window, not the day-to-day variability of rates (Table 4). "
               + coverage_note(),
               "Rs: SP-LITE reading (silicon cell, 400–1100 nm) in the water-surface plane, an index of lamp output rather "
               "than the radiant load. Tw: reference thermistor. VPD = es(Ta) − ea; Δe = es(Tw) − ea. E/VPD and E/Δe in "
               "mm h⁻¹ kPa⁻¹."])


def table3(doc):
    header = ["Comparison", "Regime", "S raw (%)", "S VPD-norm. (%)", "S Δe-norm. (%)"]
    rows = []
    for cid in CIDS:
        d = sup[sup.comparison == cid].set_index("normaliser")
        rows.append([CLAB[cid] + (", V1" if cid != "C4" else ", V2 †"), CREG[cid]]
                    + [fci(d.loc[k, "S_pct"], d.loc[k, "ci_lo"], d.loc[k, "ci_hi"]) for k in ("none", "VPD", "dE")])
    add_table(doc, "Table 3. Suppression of each covered–uncovered comparison with measurement 95 % CIs.",
              "T3_suppression_ci.csv", header, rows,
              ["S = 100 (1 − E_cov/E_unc), each rate optionally divided by the mean VPD or Δe of its own window. CIs "
               "propagate the measurement interval of both rates (Table 2) by simulation; they ignore day-to-day "
               "variability and are not the uncertainty of the cover's efficiency (Table 4).",
               "† Confounded by water temperature: the V1 and V2 water-temperature ranges do not overlap."])


def table4(doc):
    header = ["", "Raw", "VPD-normalised", "Δe-normalised"]
    ks = ("none", "VPD", "dE")
    cur = proj[proj.days_per_cell.astype(str).str.startswith("current")].set_index("normaliser")
    four = proj[proj.days_per_cell.astype(str) == "4"].set_index("normaliser")
    rows = [
        ["S, weak regime [95 % CI]"] + [fci(wm.loc[k, "S_weak"], wm.loc[k, "S_weak_lo"], wm.loc[k, "S_weak_hi"]) for k in ks],
        ["S, strong regime [95 % CI]"] + [fci(wm.loc[k, "S_strong"], wm.loc[k, "S_strong_lo"], wm.loc[k, "S_strong_hi"]) for k in ks],
        ["S weak − S strong, pp [95 % CI]"] + [fci(wm.loc[k, "diff_pp"], wm.loc[k, "diff_lo"], wm.loc[k, "diff_hi"]) for k in ks],
        ["Interaction p (3 d.f.)"] + [fp(wm.loc[k, "interaction_p"]) for k in ks],
        ["Equivalence within ±10 pp (90 % CI)"] + ["demonstrated" if wm.loc[k, "equivalent_within_margin"]
                                                   else "not demonstrated" for k in ks],
        ["Minimum detectable difference, pp"] + [f(cur.loc[k, "MDD_pp"], 0) for k in ks],
        ["  with 4 days per cell, pp"] + [f(four.loc[k, "MDD_pp"], 0) for k in ks],
        ["Pooled S, no interaction [95 % CI]"] + [fci(wm.loc[k, "S_pooled"], wm.loc[k, "S_pooled_lo"], wm.loc[k, "S_pooled_hi"]) for k in ks],
        ["95 % prediction interval, new pair"] + [f"{f(wm.loc[k, 'PI_lo'], 0)}–{f(wm.loc[k, 'PI_hi'], 0)}" for k in ks],
        ["Between-window SD of ln rate"] + [f(wm.loc[k, "between_window_sd_log"], 3) for k in ks],
        ["Mixed model, 60 min blocks: S weak / strong"] + [f"{f(mm.loc[k, 'S_weak'])} / {f(mm.loc[k, 'S_strong'])}" for k in ks],
        ["Mixed model: interaction p"] + [fp(mm.loc[k, "interaction_p_z"]) for k in ks],
    ]
    add_table(doc, "Table 4. Window-level model of suppression (primary inference).",
              "T4_window_level_model.csv; DS04_mixed_model_blocks.csv; DS24_mdd_projection.csv", header, rows,
              ["OLS of ln(E/X) on cover, regime and their interaction, one observation per window, using the seven V1 and "
               "uncovered windows (uncovered weak: 17 and 19 Aug; covered weak: 18 Aug; uncovered strong: 20 and 24 Aug; "
               "covered strong: 21 and 24 Aug), with X = 1, VPD or Δe. The residual variance is the day-to-day variability "
               "of rates under nominally identical conditions, so these intervals carry the uncertainty that matters for "
               "the cover's efficiency. 'Regime' combines fan setting and lamp setting: the strong-regime windows also had "
               "a silicon-band irradiance index about five times higher.",
               "Pooled S from the model without interaction (4 d.f.). Minimum detectable difference at α = 0.05 and power "
               "0.8 from the noncentral t (3 d.f.). The ±10 pp equivalence margin was fixed before looking at the result. "
               "Mixed model: same fixed effects on 60 min block rates with a random intercept per window."], bold_rows=(7,))


def table5(doc):
    header = ["Blocks", "Subset", "Driver", "γ PPML [95 % CI]", "p (γ = 1)", "γ PPML, E ≥ 0", "γ log-OLS [95 % CI]",
              "Blocks (windows)"]
    rows = []
    for r in el.itertuples():
        flag = " ‡" if r.few_clusters else ""
        rows.append([f"{r.block_min} min", r.subset + flag, "VPD" if r.driver == "VPD" else "Δe",
                     fci(r.gamma_ppml, r.ppml_lo, r.ppml_hi, 2), fp(r.p_ppml_eq1), f(r.gamma_ppml_nonneg, 2),
                     fci(r.gamma_logols, r.logols_lo, r.logols_hi, 2), f"{r.n_blocks} ({r.n_windows})"])
    add_table(doc, "Table 5. Within-window elasticity γ of the evaporation rate to the driving variable.",
              "T5_elasticity.csv", header, rows,
              ["E_b = exp(α_w + γ ln X_b) fitted to block rates with a fixed effect per window, so γ is identified only "
               "from variation within windows, where cover, plate count, fan and lamp setting are constant. γ = 1 is the "
               "proportionality that justifies dividing a rate by X. PPML (Poisson pseudo-maximum likelihood) uses all "
               "blocks, including the few negative block rates, which are measurement noise around a small true rate; the "
               "'E ≥ 0' column shows the estimate when they are dropped. CI with cluster-robust SEs by window and "
               "t(G − 1). Log-OLS drops blocks with E ≤ 0.",
               "‡ Only four windows: cluster-robust intervals and p-values are unreliable and are descriptive only."])


def table6(doc):
    p = paired
    header = ["Statistic", "Value"]
    rows = [["Blocks, uncovered (08:00–13:00) / covered (13:30–18:00)", f"{int(p.n_uncovered)} / {int(p.n_covered)}"],
            ["Median block rate (mm h⁻¹), uncovered / covered", f"{f(p.median_uncovered, 3)} / {f(p.median_covered, 3)}"],
            ["Mann–Whitney U; exact p", f"{p.MW_U:.0f}; {fp(p.MW_p_exact)}"],
            ["   normal approximation with continuity correction", fp(p.MW_p_asymptotic_cc)],
            ["Exact permutation test on the mean difference, p", fp(p.perm_p_exact)],
            ["Welch t (d.f.); p", f"{f(p.welch_t, 2)} ({f(p.welch_df, 1)}); {fp(p.welch_p)}"],
            ["Hodges–Lehmann shift, mm h⁻¹ [95 % CI]", fci(p.HL_shift, p.HL_lo, p.HL_hi, 3)],
            ["Hedges g [exact 95 % CI]", fci(p.hedges_g, p.g_lo, p.g_hi, 2)],
            ["Trend of uncovered block rates in time, Spearman ρ (p)", f"{f(p.trend_rho_uncovered, 2)} ({fp(p.trend_p_uncovered)})"],
            ["Trend of covered block rates in time, Spearman ρ (p)", f"{f(p.trend_rho_covered, 2)} ({fp(p.trend_p_covered)})"]]
    add_table(doc, "Table 6. Paired run of 24 August: tests on 30 min block rates.", "T6_paired_24Aug_tests.csv",
              header, rows,
              ["All uncovered blocks precede all covered blocks and the uncovered rate rises through the morning, so the "
               "blocks are not exchangeable: the tests compare the uncovered morning with the covered afternoon, and their "
               "p-values assume no time-of-day effect. The VPD and Δe ranges of the two windows do not overlap."])


# --- supplementary -----------------------------------------------------------

def table_s1(doc):
    header = ["Window", "Date", "Cover", "n", "E OLS", "SE OLS", "SE HAC-120", "Fixed-b c.v.", "SE AR-sieve",
              "SE HAC Andrews", "SE MBB-60", "E GLS-AR(1)", "DW", "ρ₁", "ACF 30 min"]
    rows = []
    for w in wr.index:
        e = est.loc[w]
        rows.append([w, wlab(w), cover_lab(w), int(e.n_records), f(e.E_ols, 4), f(e.se_ols, 4), f(e.se_hac120, 4),
                     f(e.cv_fixed_b, 2), f(e.se_sieve, 4), f(e.se_hac_andrews, 4), f(e.se_mbb60, 4), f(e.E_gls, 4),
                     f(e.DW, 2), f(e.rho1, 2), f(e.acf_30min, 2)])
    notes = ["SE in mm h⁻¹. OLS assumes independent residuals. HAC-120: Newey–West, Bartlett kernel, 120 min bandwidth "
             "(adopted), with its fixed-b critical value simulated for a trend slope. AR-sieve: bootstrap from an AR(p) "
             "fitted to the residuals. HAC Andrews: AR(1) plug-in bandwidth, too short for these residuals. MBB-60: "
             "moving-block bootstrap, 60 min blocks. GLS-AR(1): feasible GLS with AR(1) errors."]
    if cover is not None:
        med = cover.filter(like="cov_").median()
        notes.append("Median coverage of nominal 95 % intervals in simulated records (DS00_ci_coverage.csv): "
                     f"OLS {med.cov_ols:.2f}, HAC Andrews {med.cov_hac_andrews_t:.2f}, HAC-60 normal "
                     f"{med.cov_hac60_normal:.2f}, MBB-60 {med.cov_mbb60:.2f}, AR-sieve {med.cov_sieve:.2f}, "
                     f"HAC-60 fixed-b {med.cov_hac60_fixedb:.2f}, HAC-120 fixed-b {med.cov_hac120_fixedb:.2f}. The "
                     "simulated noise comes from AR models fitted to the residuals, which understate persistence, so all "
                     "of these coverages are, if anything, optimistic.")
    add_table(doc, "Table S1. Rate estimators, measurement uncertainty and residual diagnostics.",
              "DS01_rate_estimators.csv; DS00_ci_coverage.csv", header, rows, notes)


def table_s2(doc):
    header = ["Comparison", "Time unc. / cov.", "Rs unc. / cov. (W m⁻²)", "Tw unc. / cov. (°C)", "Ta unc. / cov. (°C)",
              "Ta rise unc. / cov. (K)", "ea unc. / cov. (kPa)", "VPD unc. / cov. (kPa)", "Δe unc. / cov. (kPa)",
              "Share of ΔVPD from es(Ta) (%)"]
    rows = []
    for r in bal.itertuples():
        rows.append([CLAB[r.comparison], f"{r.clock_unc} / {r.clock_cov}".replace("-", "–") + ("" if r.same_day else " ‡"),
                     f"{f(r.Rs_unc, 0)} / {f(r.Rs_cov, 0)}", f"{f(r.Tw_unc)} / {f(r.Tw_cov)}",
                     f"{f(r.Ta_unc)} / {f(r.Ta_cov)}", f"{f(r.Ta_rise_unc)} / {f(r.Ta_rise_cov)}",
                     f"{f(r.ea_unc, 2)} / {f(r.ea_cov, 2)}", f"{f(r.VPD_unc, 2)} / {f(r.VPD_cov, 2)}",
                     f"{f(r.dE_unc, 2)} / {f(r.dE_cov, 2)}", f(r.share_of_dVPD_from_es_Ta_pct, 0)])
    add_table(doc, "Table S2. Balance of each comparison: what else differed between the covered and uncovered windows.",
              "DS23_comparison_balance.csv", header, rows,
              ["‡ Windows on different days. Ta rise: increase of the HMP45C air temperature over the window (linear "
               "trend). Until 10 Sep the HMP45C sat at the top of the rig, near the lamps. In C2–C4 most of the VPD "
               "increase comes from es(Ta), i.e. from the warmer air measured at the sensor, while ea fell by only 8–14 %."])


def table_s3(doc):
    header = ["Sensitivity analysis", "Normaliser", "C1 18/17 Aug", "C2 21/20 Aug", "C3 24 Aug", "C4 25/24 Aug"]
    rows = []
    for k in ("none", "VPD", "dE"):
        b = bounds[bounds.normaliser == k].set_index("comparison")
        rows.append(["Window bounds, 256 variants: min–max", NLAB[k]] +
                    [f"{f(b.loc[c, 'S_min'])}–{f(b.loc[c, 'S_max'])}" for c in CIDS])
    for probe, lab in (("dE_p2_kPa", "Δe from thermistor 2"), ("dE_p3_kPa", "Δe from thermistor 3"),
                       ("dE_pmean_kPa", "Δe from mean of three")):
        pp = probes[probes.probe == probe].set_index("comparison")
        rows.append([lab, "Δe"] + [f(pp.loc[c, "S_pct"]) for c in CIDS])
    for mode, lab in (("common", "HMP45C RH ± 3 %, Ta ± 0.3 °C, all windows"),
                      ("strong only", "RH offset ± 3 % in strong-fan windows only")):
        g = acc[acc["mode"] == mode].groupby("comparison")[["S_VPD", "S_dE"]].agg(["min", "max"])
        rows.append([lab, "VPD"] + [f"{f(g.loc[c, ('S_VPD', 'min')])}–{f(g.loc[c, ('S_VPD', 'max')])}" for c in CIDS])
        rows.append(["", "Δe"] + [f"{f(g.loc[c, ('S_dE', 'min')])}–{f(g.loc[c, ('S_dE', 'max')])}" for c in CIDS])
    base = dsens[(dsens.c_mm_per_K == 0) & (dsens.warming_from == "probe 1")].iloc[0]
    for c_ in (0.1, 0.2, 0.3):
        d = dsens[(dsens.c_mm_per_K == c_) & (dsens.warming_from == "probe 1")].iloc[0]
        for k in ("none", "VPD"):
            rows.append([f"Transducer thermal coefficient c = {c_} mm K⁻¹" if k == "none" else "", NLAB[k]] +
                        [f(d[f"{c}_{k}"]) for c in CIDS])
    notes = ["Window bounds: start delayed and end advanced by 0, 15, 30 or 45 min independently in both windows. "
             "Thermistor positions before 10 Sep are not documented; thermistor 2 behaves as a near-surface probe, "
             "0.5–11 K warmer than the others.",
             "Thermal coefficient: rates corrected by c·dTw/dt (probe 1). c is not identified by the data (Table S7), so "
             "these rows show what a coefficient of that size would do, not a correction. Effect on the window-level "
             "model (DS22): pooled VPD-normalised S "
             + ", ".join(f"{f(dsens[(dsens.c_mm_per_K == c_) & (dsens.warming_from == 'probe 1')].iloc[0].pooled_VPD)} %"
                         for c_ in (0.0, 0.1, 0.2, 0.3))
             + " for c = 0, 0.1, 0.2, 0.3 mm K⁻¹; interaction p (VPD) "
             + ", ".join(fp(dsens[(dsens.c_mm_per_K == c_) & (dsens.warming_from == 'probe 1')].iloc[0].interaction_p_VPD)
                         for c_ in (0.0, 0.1, 0.2, 0.3))
             + "; interaction p (raw) "
             + ", ".join(fp(dsens[(dsens.c_mm_per_K == c_) & (dsens.warming_from == 'probe 1')].iloc[0].interaction_p_none)
                         for c_ in (0.0, 0.1, 0.2, 0.3)) + "."]
    add_table(doc, "Table S3. Sensitivity of suppression (%) to analysis choices and instrument uncertainty.",
              "DS12_window_bounds.csv; DS15_probe_choice.csv; DS16_sensor_accuracy.csv; DS22_thermal_drift_sensitivity.csv",
              header, rows, notes)


def table_s4(doc):
    header = ["Covered / uncovered", "Regime", "S raw", "S VPD-norm.", "S Δe-norm."]
    rows = []
    for (cw, uw), g in refs.groupby(["covered", "uncovered"], sort=False):
        g = g.set_index("normaliser")
        rows.append([f"{wlab(cw)} {cover_lab(cw)} / {wlab(uw)}", g.regime.iloc[0]] +
                    [fci(g.loc[k, "S_pct"], g.loc[k, "ci_lo"], g.loc[k, "ci_hi"]) for k in ("none", "VPD", "dE")])
    add_table(doc, "Table S4. Suppression (%) for every covered–uncovered V1 pair within the same advective regime.",
              "DS14_reference_choice.csv", header, rows,
              ["Measurement 95 % CIs. The window-level model of Table 4 uses all of these references at once instead of "
               "one hand-picked pair per regime."])


def table_s5(doc):
    header = ["Date", "Interval", "State", "Readings", "Duration (h)", "dTw/dt (K h⁻¹)", "Expansion (mm)",
              "E micrometer", "E micrometer, mass-eq.", "E transducer"]
    rows = [[dlab(r.date), f"{r.start}–{r.end}", r.state, r.n_readings, f(r.duration_h, 1), f(r.dTw_dt_K_h, 2),
             f(r.expansion_mm, 2), f(r.E_micrometer, 3), f(r.E_micrometer_mass, 3), f(r.E_transducer, 3)]
            for r in micdf.itertuples()]
    add_table(doc, "Table S5. Micrometer and level-transducer rates over common intervals (mm h⁻¹).",
              "DS17_micrometer_intervals.csv", header, rows,
              ["Rates by regression over the micrometer reading times; the transducer value at each time is the mean of "
               "±5 min of 1 min records. The micrometer reads surface height, which rises as the water column warms; the "
               "transducer reads hydrostatic pressure, i.e. mass. Mass-equivalent: height corrected for thermal expansion "
               "with the mean water temperature, assuming the water in the stilling well expands like the pan; the as-read "
               "value is the other bound. The 20 Aug interval (27 min, 2 readings) is excluded from the agreement statistics."])
    header = ["Set", "Micrometer", "n", "Pearson r", "Lin CCC [95 % CI]", "Bias (mm h⁻¹)", "Limits of agreement",
              "Geometric mean ratio [95 % CI]"]
    rows = [[txt(r.set), "mass-eq." if r.micrometer == "E_micrometer_mass" else "as read", r.n, f(r.pearson_r, 2),
             fci(r.ccc, r.ccc_lo, r.ccc_hi, 2), f(r.bias_mm_h, 3), f"{f(r.loa_lo, 3)} to {f(r.loa_hi, 3)}",
             fci(r.geo_mean_ratio, r.gmr_lo, r.gmr_hi, 2)] for r in agr.itertuples()]
    add_table(doc, "Table S6. Agreement between micrometer and transducer rates.", "DS18_micrometer_agreement.csv",
              header, rows,
              ["Lin's concordance (1/n moments) with a case-resampling bootstrap CI; it measures agreement with the identity "
               "line, which Pearson's r does not. Bias and limits of agreement (Bland–Altman): transducer minus "
               "micrometer. Ratio: micrometer / transducer. The last two rows are dominated by the 27 min interval of 20 Aug."])


def table_s7(doc):
    header = ["Source", "Variant", "n", "c (mm K⁻¹) [95 % CI]", "p", "Leave-one-out range"]
    rows = []
    for r in drift.itertuples():
        rows.append(["Micrometer, day intervals ≥ 2 h", r.variant, r.n,
                     fci(r.slope_mm_per_K, r.lo, r.hi, 2), fp(r.p), f"{f(r.loo_min, 2)} to {f(r.loo_max, 2)}"])
    for r in night_fit.itertuples():
        sub_ = r.subset.replace(" - ", "–")
        rows.append(["Night-time level, E_app = k Δe + c dT/dt",
                     f"{sub_}; {'probe 1' if r.temperature.startswith('dT1') else 'mean of probes'}",
                     r.n_nights, fci(r.c_mm_per_K, r.c_lo, r.c_hi, 2), fp(r.p_c),
                     f"{f(r.c_loo_min, 2)} to {f(r.c_loo_max, 2)}"])
    pts = pd.concat([drift.slope_mm_per_K, night_fit.c_mm_per_K])
    loo = pd.concat([drift.loo_min, drift.loo_max, night_fit.c_loo_min, night_fit.c_loo_max])
    add_table(doc, "Table S7. Apparent thermal coefficient of the level transducer.",
              "DS20_night_thermal_fit.csv; DS21_micrometer_drift_fit.csv", header, rows,
              ["c > 0 means the reading falls when the water warms (apparent extra evaporation) and rises when it cools. "
               "On the night of 1–2 Sep the level rose by about 0.2 mm while the covered pan could only lose water. Point "
               f"estimates range from {f(pts.min(), 2)} to {f(pts.max(), 2)} mm K⁻¹ (leave-one-out {f(loo.min(), 2)} to "
               f"{f(loo.max(), 2)}) depending on the nights, temperature and expansion assumption used: c is not "
               "identified by these data. A bench test at constant water mass is needed."])
    header = ["Start", "Hours", "Cover", "E_app (mm h⁻¹)", "dT/dt probe 1 (K h⁻¹)", "Δe (kPa)", "Largest 3 h change (mm)"]
    rows = [[f"{r.start.day} {r.start.strftime('%b %H:%M')}", f(r.hours, 1), r.cover, f(r.E_app_mm_h, 3),
             f(r.dT1_dt_K_h, 2), f(r.dE_kPa, 2), f(r.level_rise_max_mm, 2)] for r in nights.itertuples()]
    add_table(doc, "Table S8. Dark periods (lamps and fan off) used for the night-time check.",
              "DS19_night_periods.csv", header, rows,
              ["First 60 min after lamp switch-off and the last 10 min removed; periods cut before any level step (refill "
               "or plate operation). Largest 3 h change: maximum increase of the 60 min running mean over 3 h (negative "
               "when the level never rose)."])


def table_s9(doc):
    header = ["Contrast", "Normaliser", "Ratio", "Measurement 95 % CI", "Between-day 95 % CI"]
    rows = [[txt(r.contrast), NLAB[r.normaliser], f(r.ratio, 2),
             "–" if pd.isna(r.meas_lo) else f"{f(r.meas_lo, 2)}–{f(r.meas_hi, 2)}",
             f"{f(r.window_lo, 2)}–{f(r.window_hi, 2)}"] for r in advc.itertuples()]
    add_table(doc, "Table S9. Advection and plate-geometry contrasts (ratio of rates).", "DS07_advection_contrasts.csv",
              header, rows,
              ["Between-day CI: the between-window SD of ln rate from Table 4 (t, 4 d.f.). The uncovered strong/weak ratio "
               "compares windows whose silicon-band irradiance differed about fivefold, so it cannot be attributed to "
               "advection. The covered pair of 26 Aug and 2 Sep is the only contrast at a matched irradiance index. V2/V1 "
               "compares two different days and cannot be resolved once day-to-day variability is included."])
    header = ["Series", "x", "n", "Slope [95 % CI]", "R²", "p", "Spearman ρ (exact p)"]
    xl = {"Rs_W_m2": "Rs (per 100 W m⁻²)", "VPD_kPa": "VPD (per kPa)", "dE_kPa": "Δe (per kPa)", "Tair_C": "Ta (per K)"}
    rows = []
    for r in reg.itertuples():
        k = 100 if r.x == "Rs_W_m2" else 1
        rows.append([r.series, xl[r.x], r.n, fci(k * r.slope, k * r.slope_lo, k * r.slope_hi, 3), f(r.R2, 2),
                     f(r.p, 2), f"{f(r.spearman, 2)} ({f(r.spearman_p_exact, 2)})"])
    add_table(doc, "Table S10. Covered rate (mm h⁻¹) of cover V2 with 26 plates against window-mean forcing.",
              "DS08_covered_rate_regressions.csv", header, rows,
              ["With three or four windows no slope is significant at the 5 % level except Ta in the calm series, and the "
               "smallest attainable exact two-sided Spearman p is 0.33 (n = 3) or 0.083 (n = 4). In the calm series Rs, Ta "
               "and VPD are collinear (r ≥ 0.94)."])


def table_s11(doc):
    header = ["Window", "Date", "Cover", "Rs SP-LITE", "Rs LI200X", "LE [95 % CI]", "G probe 1", "LE/Rs (%)",
              "(LE + G)/Rs (%)"]
    rows = []
    for w in wr.index:
        e = energy.loc[w]
        rows.append([w, wlab(w), cover_lab(w), f(e.Rs_SP_LITE, 0), f(e.Rs_LI200X, 0), fci(e.LE_W_m2, e.LE_lo, e.LE_hi, 0),
                     f(e.G_probe1_W_m2, 0), f(e.LE_over_Rs_pct, 0), f(e.LEplusG1_over_Rs_pct, 0)])
    add_table(doc, "Table S11. Latent heat flux, heat storage and the silicon-pyranometer readings (W m⁻²).",
              "DS11_energy_terms.csv", header, rows,
              ["Both pyranometers are silicon cells (400–1100 nm) calibrated to sunlight; most of the output of the infrared "
               "lamps lies beyond 1100 nm, so Rs underestimates the radiant load by a factor that grows as the lamps are "
               "dimmed. LE/Rs is therefore not an energy partition. G = ρ c h dTw/dt from probe 1."])


def table_s12(doc):
    header = ["Variance", "Normaliser", "Q", "p", "I² (%)", "Pooled S [95 % CI]"]
    lab = {"measurement draws": "HAC-120 (adopted)", "AR-sieve": "AR-sieve"}
    rows = [[lab.get(r.variance, r.variance), NLAB[r.normaliser], f(r.Q, 1), fp(r.p_Q), f(r.I2_pct, 0),
             fci(r.S_random, r.S_random_lo, r.S_random_hi)] for r in het.itertuples()]
    add_table(doc, "Table S12. Heterogeneity of the three V1 comparisons using measurement variances only.",
              "DS06_heterogeneity_measurement.csv", header, rows,
              ["DerSimonian–Laird random effects on ln(E_cov/E_unc) with the modified Hartung–Knapp interval. With k = 3 "
               "and variances that omit day-to-day variability, Q depends on the variance estimator and is shown for "
               "completeness only; Table 4 gives the inference."])


README_ROWS = [
    ("window_rates", "data/processed/window_rates.csv", "One row per window: rates, uncertainty, diagnostics and forcing (Table 2)."),
    ("T3", "T3_suppression_ci.csv", "Suppression of each comparison with measurement CIs (Table 3)."),
    ("T4", "T4_window_level_model.csv", "Window-level model, primary inference (Table 4)."),
    ("T5", "T5_elasticity.csv", "Within-window elasticity (Table 5)."),
    ("T6", "T6_paired_24Aug_tests.csv", "Tests on 30 min blocks, 24 Aug (Table 6)."),
    ("DS00", "DS00_ci_coverage.csv", "Coverage of the rate intervals in simulated records (Table S1, Fig. S4)."),
    ("DS01", "DS01_rate_estimators.csv", "Rate estimators and diagnostics (Table S1)."),
    ("DS02", "DS02_se_by_bandwidth.csv", "Standard error against HAC bandwidth and block length."),
    ("DS03", "DS03_between_day_spread.csv", "Between-day SD of ln rate under nominally identical conditions."),
    ("DS04", "DS04_mixed_model_blocks.csv", "Mixed model on 60 min block rates (Table 4)."),
    ("DS05", "DS05_gamma_profile.csv", "Suppression against the assumed elasticity γ (Fig. C)."),
    ("DS06", "DS06_heterogeneity_measurement.csv", "Q and I² with measurement variances (Table S12)."),
    ("DS07", "DS07_advection_contrasts.csv", "Advection and V2/V1 contrasts (Table S9, Figs. 5 and 8)."),
    ("DS08", "DS08_covered_rate_regressions.csv", "Covered rate against forcing (Table S10, Fig. 5)."),
    ("DS09", "DS09_26-28Aug_ranks.csv", "Ranks of rate, VPD, Δe and Rs, 26–28 Aug."),
    ("DS10", "DS10_26-28Aug_relative_range.csv", "Relative range of raw and normalised rates, 26–28 Aug (Fig. 7)."),
    ("DS11", "DS11_energy_terms.csv", "LE, heat storage and pyranometer readings (Table S11, Fig. 6)."),
    ("DS12", "DS12_window_bounds.csv", "Sensitivity to window bounds (Table S3, Fig. S2)."),
    ("DS13", "DS13_settling.csv", "Sensitivity to settling time (Fig. S2)."),
    ("DS14", "DS14_reference_choice.csv", "All within-regime V1 pairs (Table S4, Fig. S2)."),
    ("DS15", "DS15_probe_choice.csv", "Δe-normalised S by thermistor (Table S3)."),
    ("DS16", "DS16_sensor_accuracy.csv", "HMP45C accuracy (Table S3)."),
    ("DS17", "DS17_micrometer_intervals.csv", "Micrometer and transducer rates (Table S5, Fig. S1)."),
    ("DS18", "DS18_micrometer_agreement.csv", "Agreement statistics (Table S6, Fig. S1)."),
    ("DS19", "DS19_night_periods.csv", "Dark periods (Table S8, Fig. S3)."),
    ("DS20", "DS20_night_thermal_fit.csv", "Night-time thermal coefficient fits (Table S7, Fig. S3)."),
    ("DS21", "DS21_micrometer_drift_fit.csv", "Micrometer-based thermal coefficient (Table S7)."),
    ("DS22", "DS22_thermal_drift_sensitivity.csv", "Suppression under an assumed thermal coefficient (Table S3)."),
    ("DS23", "DS23_comparison_balance.csv", "Balance of each comparison (Table S2)."),
    ("DS24", "DS24_mdd_projection.csv", "Minimum detectable difference by days per cell (Table 4)."),
    ("DS25", "DS25_regime_effect_at_estimated_gamma.csv", "Regime effect at the estimated elasticity, conditional on γ."),
]


def natural_key(path):
    m = re.match(r"(T|DS)(\d+)", path.stem)
    return (0 if m.group(1) == "T" else 1, int(m.group(2))) if m else (2, 0)


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(9)
    s0 = doc.sections[0]
    s0.orientation = WD_ORIENT.LANDSCAPE
    s0.page_width, s0.page_height = Mm(297), Mm(210)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(s0, m, Mm(15))
    doc.add_paragraph().add_run("MAISSER-PE — revised and new tables (generated by make_tables.py)").bold = True
    table2(doc)
    section(doc, landscape=False)
    table3(doc)
    table4(doc)
    table5(doc)
    table6(doc)
    section(doc, landscape=True)
    doc.add_paragraph().add_run("Supplementary tables").bold = True
    table_s1(doc)
    table_s2(doc)
    table_s3(doc)
    table_s4(doc)
    table_s5(doc)
    table_s7(doc)
    table_s9(doc)
    table_s11(doc)
    table_s12(doc)
    doc.save(T / "MAISSER-PE_tables.docx")

    with pd.ExcelWriter(T / "MAISSER-PE_tables.xlsx") as xl:
        pd.DataFrame(README_ROWS, columns=["sheet", "file", "content"]).to_excel(xl, sheet_name="README", index=False)
        pd.read_csv(mc.PROCESSED / "window_rates.csv").to_excel(xl, sheet_name="window_rates", index=False)
        for path in sorted(list(T.glob("T[0-9]*.csv")) + list(T.glob("DS*.csv")), key=natural_key):
            pd.read_csv(path).to_excel(xl, sheet_name=path.stem[:31], index=False)
    print("Tables written: MAISSER-PE_tables.docx, MAISSER-PE_tables.xlsx")


if __name__ == "__main__":
    main()
