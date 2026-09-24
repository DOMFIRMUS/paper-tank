#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - figures.

Run after statistical_analysis.py (and coverage_check.py for figure S4).
Every figure is written to ../results/figures/ as a vector PDF with embedded fonts
(preferred by Elsevier for line art), a 1000 dpi PNG (the minimum Elsevier
accepts for bitmapped line art) and a 200 dpi preview. Widths follow the
Elsevier grid: 90 mm (single column), 140 mm (1.5 column), 190 mm (full).

Colour encodes one thing per figure and is always paired with a second
channel (marker shape, fill or line style), using the Okabe-Ito palette,
which remains distinguishable under the common colour-vision deficiencies.

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
Project: MAISSER-PE, FACEPE APQ-1428-3.01/25
License: MIT
Usage:   python make_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

import maisser_core as mc

MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.1, "lines.markersize": 4.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "mathtext.default": "regular",
})

# Okabe-Ito
C_UNC, C_V1, C_V2 = "#D55E00", "#0072B2", "#009E73"
C_RAW, C_VPD, C_DE = "#4D4D4D", "#E69F00", "#CC79A7"
C_SKY, C_GREY = "#56B4E9", "#8C8C8C"
INK, INK2 = "#1A1A1A", "#595959"

STATE_STYLE = {
    "uncovered": dict(color=C_UNC, marker="o", label="Uncovered"),
    "V1": dict(color=C_V1, marker="s", label="Cover V1"),
    "V2": dict(color=C_V2, marker="D", label="Cover V2"),
}
NORM_STYLE = {
    "none": dict(color=C_RAW, marker="o", mfc="white", label="Raw"),
    "VPD": dict(color=C_VPD, marker="s", mfc=C_VPD, label="VPD-normalised"),
    "dE": dict(color=C_DE, marker="D", mfc=C_DE, label="Δe-normalised"),
}

T = mc.TABLES
wr = pd.read_csv(mc.PROCESSED / "window_rates.csv", parse_dates=["start", "end"]).set_index("win")
est = pd.read_csv(T / "DS01_rate_estimators.csv").set_index("win")
sup = pd.read_csv(T / "T3_suppression_ci.csv")
wm = pd.read_csv(T / "T4_window_level_model.csv").set_index("normaliser")
el = pd.read_csv(T / "T5_elasticity.csv")
paired = pd.read_csv(T / "T6_paired_24Aug_tests.csv").iloc[0]
prof = pd.read_csv(T / "DS05_gamma_profile.csv")
advc = pd.read_csv(T / "DS07_advection_contrasts.csv")
reg = pd.read_csv(T / "DS08_covered_rate_regressions.csv")
rrange = pd.read_csv(T / "DS10_26-28Aug_relative_range.csv").set_index("normaliser")
energy = pd.read_csv(T / "DS11_energy_terms.csv").set_index("win")
bounds = pd.read_csv(T / "DS12_window_bounds.csv")
settle = pd.read_csv(T / "DS13_settling.csv")
refs = pd.read_csv(T / "DS14_reference_choice.csv")
micdf = pd.read_csv(T / "DS17_micrometer_intervals.csv")
agr = pd.read_csv(T / "DS18_micrometer_agreement.csv")
nights = pd.read_csv(T / "DS19_night_periods.csv", parse_dates=["start", "end"])
night_fit = pd.read_csv(T / "DS20_night_thermal_fit.csv")
drift = pd.read_csv(T / "DS21_micrometer_drift_fit.csv")
blocks60 = pd.read_csv(T / "blocks_60min.csv")
blocks24 = pd.read_csv(T / "blocks_30min_24Aug.csv")
boot = dict(np.load(T / "bootstrap_rates.npz"))
con = mc.load_consolidated()


def state(win):
    v = wr.loc[win, "version"]
    return "uncovered" if v == "-" else v


def dlab(win):
    ts = wr.loc[win, "start"]
    return f"{ts.day} {ts.strftime('%b')}"


def ci(win):
    return est.loc[win, "ci_lo"], est.loc[win, "ci_hi"]


def pct_ci(x, a=0.05):
    return np.percentile(x, [100 * a / 2, 100 * (1 - a / 2)])


def save(fig, name):
    """Save at exactly the width the figure was created with (90, 140 or
    190 mm). The tight bounding box adds or trims margins, so the canvas is
    rescaled until the cropped figure has the target width; text sizes in
    points are unaffected. Rasterised layers are written at 600 dpi in the PDF.
    """
    target = fig.get_size_inches()[0]
    pad = mpl.rcParams["savefig.pad_inches"]
    for _ in range(6):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        width = bb.width + 2 * pad
        if abs(width - target) < 0.2 * MM:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * target / width, h)
    fig.savefig(mc.FIGURES / f"{name}.pdf", dpi=600)
    fig.savefig(mc.FIGURES / f"{name}.png", dpi=1000)
    fig.savefig(mc.FIGURES / f"{name}_preview.png", dpi=200)
    plt.close(fig)
    print(f"  {name}: {width / MM:.1f} mm wide")


def mn(text):
    """Typographic minus sign in annotation text."""
    return text.replace("-", "−")


def fp(p):
    """p value as 6.5 × 10⁻⁴ (mathtext)."""
    if p >= 0.001:
        return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10$^{{{int(e)}}}$"


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=8, fontweight="bold", color=INK, pad=4)


def errbar(ax, x, y, lo, hi, style, ms=4.5, horizontal=False, **kw):
    err = [[abs(y - lo if not horizontal else x - lo)], [abs(hi - y if not horizontal else hi - x)]]
    if horizontal:
        ax.errorbar(x, y, xerr=err, fmt="none", ecolor=style["color"], elinewidth=0.9, capsize=2, zorder=2)
    else:
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=style["color"], elinewidth=0.9, capsize=2, zorder=2)
    ax.plot(x, y, linestyle="none", marker=style["marker"], ms=ms,
            mfc=style.get("mfc", style["color"]), mec=style["color"], mew=0.9, zorder=3, **kw)


# ---------------------------------------------------------------------------
def fig_method():
    """Rate estimation on 24 August, residual autocorrelation, block test."""
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    day = con[(con.TIMESTAMP >= "2026-08-24 07:30") & (con.TIMESTAMP <= "2026-08-24 18:30")]
    ax.plot(day.TIMESTAMP, day.nivel_tanque_Avg, ".", ms=1.4, color=C_GREY, alpha=0.8,
            label="Level, 1 min mean (stored at 0.1 mm)", rasterized=True)
    for w in ("W06", "W07"):
        sub = mc.window_slice(con, wr.loc[w, "start"], wr.loc[w, "end"])
        fit = mc.ols_trend(mc.hours_since_start(sub.TIMESTAMP), sub.nivel_tanque_Avg.to_numpy())["fit"]
        st = STATE_STYLE[state(w)]
        lo, hi = ci(w)
        ax.plot(sub.TIMESTAMP, fit.fittedvalues, color=st["color"], lw=1.6,
                label=f"{st['label']}: E = {wr.loc[w, 'E_mm_h']:.3f} mm h$^{{-1}}$ [{lo:.3f}, {hi:.3f}]")
    ax.axvspan(pd.Timestamp("2026-08-24 13:00"), pd.Timestamp("2026-08-24 13:25"),
               color="#BDBDBD", alpha=0.35, lw=0, label="Excluded: deployment and settling")
    pre = day[(day.TIMESTAMP >= "2026-08-24 12:55") & (day.TIMESTAMP <= "2026-08-24 13:03")].nivel_tanque_Avg.mean()
    post = day[(day.TIMESTAMP >= "2026-08-24 13:06") & (day.TIMESTAMP <= "2026-08-24 13:14")].nivel_tanque_Avg.mean()
    ax.annotate(f"buoyancy step +{post - pre:.1f} mm (13:04–13:05)",
                xy=(pd.Timestamp("2026-08-24 13:05"), post), xytext=(pd.Timestamp("2026-08-24 13:50"), 221.6),
                fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("Water level (mm)")
    ax.set_xlabel("Local time, 24 August 2026")
    ax.legend(loc="lower left", ncol=1, fontsize=6.5, markerscale=4)
    ax.set_ylim(214.2, 223.0)
    panel_label(ax, "(a) Rate by regression, buoyancy transient excluded (95 % CI: HAC-120, fixed-b)")

    ax = fig.add_subplot(gs[1, 0])
    lags = np.arange(0, 91)
    for w, *_ in mc.WINDOWS:
        sub = mc.window_slice(con, wr.loc[w, "start"], wr.loc[w, "end"])
        e = np.asarray(mc.ols_trend(mc.hours_since_start(sub.TIMESTAMP), sub.nivel_tanque_Avg.to_numpy())["resid"])
        e = e - e.mean()
        acf = np.array([1.0] + [np.sum(e[k:] * e[:-k]) / np.sum(e * e) for k in lags[1:]])
        if w in ("W06", "W07"):
            ax.plot(lags, acf, color=STATE_STYLE[state(w)]["color"], lw=1.3, zorder=3,
                    label=f"24 Aug, {'uncovered' if w == 'W06' else 'cover V1'}")
        else:
            ax.plot(lags, acf, color="#C8C8C8", lw=0.6, zorder=1)
    n = int(np.median(wr.n_records))
    ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color="#9E9E9E", alpha=0.25, lw=0)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.plot([], [], color="#C8C8C8", lw=0.6, label="other 13 windows")
    ax.plot([], [], color="#9E9E9E", lw=5, alpha=0.4, label=f"±1.96/√n, n = {n} (median)")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.45, 1.02)
    ax.set_xlabel("Lag (min)")
    ax.set_ylabel("Autocorrelation of residuals")
    ax.legend(loc="upper right", fontsize=6.3)
    panel_label(ax, "(b) Residual dependence lasts tens of minutes")

    ax = fig.add_subplot(gs[1, 1])
    r = paired
    for i, w in enumerate(("W06", "W07")):
        st = STATE_STYLE[state(w)]
        v = blocks24[blocks24.win == w].E.to_numpy()
        ax.plot(i + np.linspace(-0.13, 0.13, len(v)), v, linestyle="none", marker=st["marker"], ms=4,
                mfc=st["color"], mec="white", mew=0.5)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Uncovered, 08–13 h\n(n = {int(r.n_uncovered)})", f"Cover V1, 13:30–18 h\n(n = {int(r.n_covered)})"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 2.25)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("30 min block rate (mm h$^{-1}$)")
    ax.text(1.0, 2.18, f"Mann–Whitney U = {r.MW_U:.0f}\nexact p = {fp(r.MW_p_exact)}\n"
                       f"HL shift {r.HL_shift:.2f} mm h$^{{-1}}$\n[{r.HL_lo:.2f}, {r.HL_hi:.2f}]",
            fontsize=6.3, color=INK, ha="center", va="top")
    ax.text(0.02, 0.02, f"uncovered blocks rise in time: Spearman ρ = {r.trend_rho_uncovered:.2f}",
            transform=ax.transAxes, fontsize=6.1, color=C_UNC, ha="left", va="bottom")
    panel_label(ax, "(c) Morning uncovered vs afternoon covered")
    save(fig, "figA_method_24Aug")


# ---------------------------------------------------------------------------
def fig_rates():
    """Fig. 3 (revised): rate of every window with its 95 % CI, by regime."""
    groups = [("Weak advection\n≈1.9–2.2 m s$^{-1}$", ["W01", "W02", "W03"]),
              ("Strong advection\n3.2–3.6 m s$^{-1}$", ["W04", "W05", "W06", "W07", "W08", "W09", "W10", "W11"]),
              ("No forced advection\nu = 0", ["W12", "W13", "W14", "W15"])]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 72 * MM), sharey=True,
                             gridspec_kw={"width_ratios": [3.4, 8, 4.4], "wspace": 0.08})
    for i, (ax, (title, ws)) in enumerate(zip(axes, groups)):
        for j, w in enumerate(ws):
            st = STATE_STYLE[state(w)]
            lo, hi = ci(w)
            e = wr.loc[w, "E_mm_h"]
            ax.bar(j, e, width=0.62, color=st["color"], alpha=0.85, edgecolor="white", linewidth=0.8, zorder=2)
            ax.errorbar(j, e, yerr=[[e - lo], [hi - e]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
        ax.set_xticks(range(len(ws)))
        ax.set_xticklabels([f"{dlab(w)}\n{'unc.' if state(w) == 'uncovered' else state(w)}\n"
                            f"{wr.loc[w, 'Rs_W_m2']:.0f}" for w in ws], fontsize=6.5)
        ax.set_xlim(-0.6, len(ws) - 0.4)
        ax.grid(axis="x", visible=False)
        panel_label(ax, f"({'abc'[i]}) {title}")
        if i:
            ax.tick_params(axis="y", left=False)
    axes[0].set_ylabel("Evaporation rate (mm h$^{-1}$)")
    axes[0].set_ylim(0, 1.7)
    axes[1].set_xlabel("Date | cover | SP-LITE reading (W m$^{-2}$)", fontsize=7, color=INK2)
    handles = [mpl.patches.Patch(color=STATE_STYLE[k]["color"], alpha=0.85, label=STATE_STYLE[k]["label"])
               for k in ("uncovered", "V1", "V2")]
    handles.append(mpl.lines.Line2D([], [], color=INK, lw=0.8, marker="_", ms=6,
                                    label="95 % CI, measurement (HAC-120, fixed-b)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15))
    save(fig, "fig03_rates_by_regime")


# ---------------------------------------------------------------------------
def fig_suppression():
    """Fig. 4 (revised): suppression per comparison and from the window-level
    model, and the weak-minus-strong contrast against a 10 pp margin."""
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.95)
    ax = fig.add_subplot(gs[0])
    ytr = ax.get_yaxis_transform()
    labels = {"C1": "C1  18 vs 17 Aug, weak", "C2": "C2  21 vs 20 Aug, strong", "C3": "C3  24 Aug paired, strong"}
    y, yt, ylab = 0.0, [], []
    for key, stl in NORM_STYLE.items():
        m = wm.loc[key]
        ax.text(-0.02, y, f"{stl['label']}", transform=ytr, fontsize=6.9, fontweight="bold", color=INK,
                va="center", ha="right")
        y += 1
        for cid in ("C1", "C2", "C3"):
            r = sup[(sup.comparison == cid) & (sup.normaliser == key)].iloc[0]
            errbar(ax, r.S_pct, y, r.ci_lo, r.ci_hi, stl, horizontal=True)
            yt.append(y)
            ylab.append(labels[cid])
            ax.text(1.02, y, mn(f"{r.S_pct:.1f} [{r.ci_lo:.1f}, {r.ci_hi:.1f}]"), transform=ytr, fontsize=6.3,
                    va="center", color=INK2)
            y += 1
        for lab, c, lo, hi in (("Weak regime (model)", m.S_weak, m.S_weak_lo, m.S_weak_hi),
                               ("Strong regime (model)", m.S_strong, m.S_strong_lo, m.S_strong_hi),
                               ("Pooled (model)", m.S_pooled, m.S_pooled_lo, m.S_pooled_hi)):
            if lab.startswith("Pooled"):
                ax.plot([m.PI_lo, m.PI_hi], [y, y], color=stl["color"], lw=0.8, ls=(0, (1, 1.2)))
            ax.fill([lo, c, hi, c], [y, y - 0.3, y, y + 0.3], color=stl["color"], alpha=0.9 if lab.startswith("Pooled")
                    else 0.45, lw=0)
            yt.append(y)
            ylab.append(lab)
            ax.text(1.02, y, f"{c:.1f} [{lo:.1f}, {hi:.1f}]", transform=ytr, fontsize=6.3, va="center",
                    color=INK, fontweight="bold" if lab.startswith("Pooled") else "normal")
            y += 1
        y += 0.7
    ax.text(1.02, -0.9, "S [95 % CI]", transform=ytr, fontsize=6.3, color=INK2, va="center")
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab, fontsize=6.4)
    ax.set_ylim(y - 1.0, -1.4)
    ax.set_xlim(0, 95)
    ax.axvline(77.0, color=C_GREY, lw=0.6, ls=(0, (4, 3)))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Suppression S (%)")
    hs = [mpl.lines.Line2D([], [], color=INK2, marker="o", mfc="white", lw=0.9, label="pair: measurement CI"),
          mpl.patches.Polygon([[0, 0]], closed=True, color=INK2, alpha=0.6, label="window-level model: 95 % CI"),
          mpl.lines.Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.2)), label="95 % prediction interval, new pair"),
          mpl.lines.Line2D([], [], color=C_GREY, lw=0.6, ls=(0, (4, 3)), label="fractional cover 77 %")]
    ax.legend(handles=hs, loc="upper left", fontsize=6.1, bbox_to_anchor=(-0.55, -0.1), ncol=2,
              handlelength=1.6)
    panel_label(ax, "(a) Suppression by comparison and by regime")

    ax = fig.add_subplot(gs[1])
    ytr = ax.get_yaxis_transform()
    ax.axvspan(-mc_margin(), mc_margin(), color="#9E9E9E", alpha=0.15, lw=0)
    for i, (key, stl) in enumerate(NORM_STYLE.items()):
        m = wm.loc[key]
        ax.plot([m.diff90_lo, m.diff90_hi], [i, i], color=stl["color"], lw=3.4, alpha=0.45, solid_capstyle="butt")
        errbar(ax, m.diff_pp, i, m.diff_lo, m.diff_hi, stl, ms=5, horizontal=True)
        ax.text(1.03, i, mn(f"{m.diff_pp:+.1f} [{m.diff_lo:+.1f}, {m.diff_hi:+.1f}]") + f"\np = {m.interaction_p:.2f}",
                transform=ytr, fontsize=6.3, va="center")
    ax.text(1.03, -0.75, "ΔS [95 % CI]", transform=ytr, fontsize=6.3, color=INK2, va="center")
    ax.text(-mc_margin() + 0.8, -0.75, f"±{mc_margin():.0f} pp margin", fontsize=6.0, color=INK2, ha="left",
            va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(range(3))
    ax.set_yticklabels([s["label"] for s in NORM_STYLE.values()], fontsize=6.6)
    ax.set_ylim(2.8, -1.1)
    ax.set_xlim(-30, 50)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("S$_{weak}$ − S$_{strong}$ (percentage points)")
    ax.plot([], [], color=INK2, lw=3.4, alpha=0.45, label="90 % CI (equivalence test)")
    ax.plot([], [], color=INK2, lw=0.9, marker="|", ms=5, label="95 % CI")
    ax.legend(loc="lower right", fontsize=6.1, handlelength=1.6, frameon=True, framealpha=1,
              edgecolor="none")
    panel_label(ax, "(b) Regime effect, window-level model")
    save(fig, "fig04_suppression")


def mc_margin():
    return 10.0


# ---------------------------------------------------------------------------
def fig_gamma():
    """New: suppression as a function of the assumed elasticity gamma."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 70 * MM), sharey=True, gridspec_kw={"wspace": 0.1})
    # comparisons in neutral greys and line styles; colour stays reserved for
    # cover state (vermillion band = uncovered blocks)
    cst = {"C1": dict(color="#6E6E6E", ls="-", label="C1 18 vs 17 Aug (weak)"),
           "C2": dict(color="#6E6E6E", ls=(0, (5, 2)), label="C2 21 vs 20 Aug (strong)"),
           "C3": dict(color="#6E6E6E", ls=(0, (1, 1.2)), label="C3 24 Aug paired (strong)")}
    for ax, (key, title) in zip(axes, (("VPD", "(a) Normaliser: air deficit VPD"),
                                       ("dE", "(b) Normaliser: surface–air difference Δe"))):
        p = prof[prof.normaliser == key]
        for sub_, lab, color in (("all", "all blocks", "#BDBDBD"), ("uncovered", "uncovered blocks", "#F2D3C2")):
            r = el[(el.block_min == 60) & (el.subset == sub_) & (el.driver == key)].iloc[0]
            ax.axvspan(r.ppml_lo, r.ppml_hi, color=color, alpha=0.55 if sub_ == "all" else 0.6, lw=0, zorder=0)
            ax.axvline(r.gamma_ppml, color=INK2 if sub_ == "all" else C_UNC, lw=0.7)
        for cid, s in cst.items():
            ax.plot(p.gamma, p[f"S_{cid}"], color=s["color"], ls=s["ls"], lw=1.3, label=s["label"])
        ax.plot(p.gamma, p.S_pooled, color=INK, lw=2.0, label="pooled, window-level model")
        ax.axvline(1.0, color=INK, lw=0.6, ls=(0, (4, 3)))
        ax.text(0.96, 91, "γ = 1\n(assumed)", fontsize=6.2, color=INK2, va="top", ha="right")
        ax.set_xlim(0, 3)
        ax.set_ylim(40, 95)
        ax.set_xlabel("Elasticity γ in S = 1 − (E$_c$/E$_u$)(X$_u$/X$_c$)$^γ$")
        panel_label(ax, title)
    axes[0].set_ylabel("Suppression S (%)")
    hs, ls_ = axes[0].get_legend_handles_labels()
    hs += [mpl.patches.Patch(color="#BDBDBD", alpha=0.55, label="γ estimated, all blocks (PPML, 95 % CI)"),
           mpl.patches.Patch(color="#F2D3C2", alpha=0.6, label="γ estimated, uncovered blocks")]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), fontsize=6.4)
    save(fig, "figC_gamma_profile")


# ---------------------------------------------------------------------------
def fig_elasticity():
    """Within-window elasticity of the rate to VPD and to dE (60 min blocks)."""
    b = blocks60[blocks60.E > 0].copy()        # log scale for display; the fit (PPML) uses all blocks
    b["lnE"] = np.log(b.E)
    fig, axes = plt.subplots(1, 2, figsize=(W15, 72 * MM), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (drv, col, title) in zip(axes, (("VPD", "VPD_kPa", "(a) Air deficit VPD"),
                                            ("dE", "dE_kPa", "(b) Surface–air difference Δe"))):
        b["lnX"] = np.log(b[col])
        d = b.assign(x=b.lnX - b.groupby("win").lnX.transform("mean"),
                     y=b.lnE - b.groupby("win").lnE.transform("mean"))
        d["state"] = [state(w) for w in d.win]
        for st_key, lab in (("uncovered", "Uncovered blocks"), ("V1", "Cover V1 blocks"), ("V2", "Cover V2 blocks")):
            stl = STATE_STYLE[st_key]
            dd = d[d.state == st_key]
            ax.plot(dd.x, dd.y, linestyle="none", marker=stl["marker"], ms=3.6 if st_key == "uncovered" else 3.2,
                    mfc=stl["color"] if st_key == "uncovered" else "white", mec=stl["color"], mew=0.7, alpha=0.9,
                    label=lab)
        xs = np.linspace(-0.45, 0.45, 50)
        ax.plot(xs, xs, color=INK2, lw=0.8, ls=(0, (4, 3)), label="γ = 1 (proportional)")
        for subset, color, ls in (("uncovered", C_UNC, "-"), ("all", INK, (0, (1, 1.2)))):
            r = el[(el.block_min == 60) & (el.subset == subset) & (el.driver == drv)].iloc[0]
            ax.plot(xs, r.gamma_ppml * xs, color=color, lw=1.4, ls=ls,
                    label=f"{subset}: γ = {r.gamma_ppml:.2f} [{r.ppml_lo:.2f}, {r.ppml_hi:.2f}]")
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xlabel(f"ln {'VPD' if drv == 'VPD' else 'Δe'} − window mean")
        panel_label(ax, title)
        ax.legend(loc="upper left", fontsize=6.0, handlelength=1.6)
    axes[0].set_ylabel("ln E − window mean  (60 min blocks)")
    save(fig, "figB_elasticity")


# ---------------------------------------------------------------------------
def fig_advection():
    """Fig. 5 (revised)."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 74 * MM), gridspec_kw={"width_ratios": [0.8, 1.35], "wspace": 0.3})
    ax = axes[0]
    for i, (w, fill) in enumerate((("W09", True), ("W14", False))):
        lo, hi = ci(w)
        e = wr.loc[w, "E_mm_h"]
        ax.bar(i, e, width=0.55, color=C_V2 if fill else "white", edgecolor=C_V2, linewidth=1.1,
               hatch=None if fill else "////", zorder=2)
        ax.errorbar(i, e, yerr=[[e - lo], [hi - e]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=3)
    r = advc[(advc.contrast.str.startswith("covered V2-26")) & (advc.normaliser == "none")].iloc[0]
    ax.annotate("", xy=(1, 0.40), xytext=(0, 0.40), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax.text(0.5, 0.415, f"×{r.ratio:.2f}\nmeasurement [{r.meas_lo:.2f}, {r.meas_hi:.2f}]\n"
                        f"between-day [{r.window_lo:.2f}, {r.window_hi:.2f}]", ha="center", fontsize=6.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"26 Aug\n{wr.loc['W09', 'Rs_W_m2']:.0f} W m$^{{-2}}$\nu = {wr.loc['W09', 'u_m_s']:.2f} m s$^{{-1}}$",
                        f"2 Sep\n{wr.loc['W14', 'Rs_W_m2']:.0f} W m$^{{-2}}$\nu = 0"])
    ax.set_ylim(0, 0.52)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Evaporation rate (mm h$^{-1}$)")
    panel_label(ax, "(a) Cover V2, same SP-LITE reading")

    ax = axes[1]
    # both series are cover V2: colour = cover, fill = fan (filled on, open off)
    series = {"strong advection, V2 26 plates": (["W09", "W10", "W11"], True, "u = 3.2–3.6 m s$^{-1}$"),
              "calm (u = 0), V2 26 plates": (["W12", "W13", "W14", "W15"], False, "u = 0")}
    offs = {"W09": (6, 3), "W10": (6, -9), "W11": (6, 4), "W12": (-30, 2), "W13": (6, -9), "W14": (6, -10),
            "W15": (6, 2)}
    for key, (ws, filled, lab) in series.items():
        d = wr.loc[ws]
        rr = reg[(reg.series == key) & (reg.x == "Rs_W_m2")].iloc[0]
        if not filled:   # a line only for n = 4; with n = 3 a fitted line would suggest a trend
            xr = np.linspace(d.Rs_W_m2.min(), d.Rs_W_m2.max(), 50)
            ax.plot(xr, rr.intercept + rr.slope * xr, color=C_V2, lw=0.8, ls=(0, (4, 2)), alpha=0.8)
        for w in ws:
            lo, hi = ci(w)
            e = wr.loc[w, "E_mm_h"]
            ax.errorbar(wr.loc[w, "Rs_W_m2"], e, yerr=[[e - lo], [hi - e]], fmt="none", ecolor=C_V2,
                        elinewidth=0.9, capsize=2)
            ax.plot(wr.loc[w, "Rs_W_m2"], e, marker="o" if filled else "s", ms=5, mfc=C_V2 if filled else "white",
                    mec=C_V2, mew=1.0, lw=0)
            ax.annotate(dlab(w), (wr.loc[w, "Rs_W_m2"], e), xytext=offs[w], textcoords="offset points",
                        fontsize=6.2, color=INK2)
        ax.plot([], [], color=C_V2, marker="o" if filled else "s", ls="none" if filled else (0, (4, 2)),
                mfc=(C_V2 if filled else "white"), label=f"{lab} (n = {len(d)}): R² = {rr.R2:.2f}, p = {rr.p:.2f}")
    ax.set_xlim(100, 1000)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("SP-LITE reading (W m$^{-2}$; silicon band, not total lamp radiation)")
    ax.set_ylabel("Evaporation rate (mm h$^{-1}$)")
    ax.legend(loc="upper left", fontsize=6.2)
    panel_label(ax, "(b) Covered rate against the irradiance index")
    save(fig, "fig05_advection")


# ---------------------------------------------------------------------------
def fig_energy():
    """Fig. 6 (revised): LE and LE + storage against the SP-LITE reading.

    No 1:1 line: the SP-LITE reading is a silicon-band index of lamp output,
    so LE/Rs is not an energy partition. Colour and marker = cover state,
    fill = fan (filled on, open off); weak-fan windows are labelled.
    """
    fig, ax = plt.subplots(figsize=(W15, 88 * MM))
    for w in wr.index:
        stl = STATE_STYLE[state(w)]
        e = energy.loc[w]
        fan = wr.loc[w, "regime"] != "calm"
        ax.errorbar(e.Rs_SP_LITE, e.LE_W_m2, yerr=[[e.LE_W_m2 - e.LE_lo], [e.LE_hi - e.LE_W_m2]],
                    fmt="none", ecolor=stl["color"], elinewidth=0.8, capsize=1.8)
        ax.plot([e.Rs_SP_LITE] * 2, [e.LE_W_m2, e.LE_W_m2 + e.G_probe1_W_m2], color=stl["color"], lw=0.7,
                ls=(0, (1, 1.2)))
        ax.plot(e.Rs_SP_LITE, e.LE_W_m2 + e.G_probe1_W_m2, marker="^", ms=4.2, mfc="white", mec=stl["color"],
                mew=0.8, lw=0)
        ax.plot(e.Rs_SP_LITE, e.LE_W_m2, marker=stl["marker"], ms=5.2, mfc=stl["color"] if fan else "white",
                mec=stl["color"], mew=1.0, lw=0)
        if wr.loc[w, "regime"] == "weak":
            xy = {"W01": (30, 170), "W03": (240, 405), "W02": (178, 35)}[w]
            ax.text(*xy, f"{dlab(w)} (weak fan)", fontsize=6.0, color=INK2, ha="left", va="center")
    hs = [mpl.lines.Line2D([], [], color=STATE_STYLE[k]["color"], marker=STATE_STYLE[k]["marker"], lw=0,
                           label=STATE_STYLE[k]["label"]) for k in ("uncovered", "V1", "V2")]
    hs += [mpl.lines.Line2D([], [], color=INK2, marker="o", lw=0, label="fan on (filled)"),
           mpl.lines.Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, label="fan off (open)"),
           mpl.lines.Line2D([], [], color=INK2, marker="^", mfc="white", ls=(0, (1, 1.2)), lw=0.7,
                            label="LE + heat storage G (probe 1)"),
           mpl.lines.Line2D([], [], color=INK2, marker="|", ms=6, lw=0, label="95 % CI of LE")]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.4)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1450)
    ax.set_xlabel("SP-LITE reading (W m$^{-2}$; silicon band)")
    ax.set_ylabel("Latent heat flux LE (W m$^{-2}$)")
    panel_label(ax, "LE and heat storage against the irradiance index")
    save(fig, "fig06_LE_vs_irradiance_index")


# ---------------------------------------------------------------------------
def fig_26_28():
    """Fig. 7 (revised): 26-28 August, raw and normalised rates on one scale."""
    ws = ["W09", "W10", "W11"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 70 * MM), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    for w in ws:
        lo, hi = ci(w)
        e = wr.loc[w, "E_mm_h"]
        errbar(ax, wr.loc[w, "Rs_W_m2"], e, lo, hi, STATE_STYLE["V2"], ms=5)
        ax.annotate(f"{dlab(w)}\nVPD {wr.loc[w, 'VPD_kPa']:.2f} kPa\nΔe {wr.loc[w, 'dE_kPa']:.2f} kPa",
                    (wr.loc[w, "Rs_W_m2"], e), xytext=(8, -4), textcoords="offset points", fontsize=6.2,
                    color=INK2, va="top")
    ax.set_xlim(200, 900)
    ax.set_ylim(0.18, 0.42)
    ax.set_xlabel("SP-LITE reading (W m$^{-2}$)")
    ax.set_ylabel("Evaporation rate (mm h$^{-1}$)")
    panel_label(ax, "(a) Cover V2, u ≈ 3.2–3.6 m s$^{-1}$")

    ax = axes[1]
    offs = {"none": -0.2, "VPD": 0.0, "dE": 0.2}
    for key, stl in NORM_STYLE.items():
        nv = [1 if key == "none" else wr.loc[w, mc.NORMALISERS[key]] for w in ws]
        vals = np.array([wr.loc[w, "E_mm_h"] / n_ for w, n_ in zip(ws, nv)])
        bs = np.array([boot[w] / n_ for w, n_ in zip(ws, nv)])
        m = vals.mean()
        for i in range(3):
            lo, hi = pct_ci(bs[i] / bs.mean(axis=0))
            errbar(ax, i + offs[key], vals[i] / m, lo, hi, stl)
        ax.plot([], [], color=stl["color"], marker=stl["marker"], mfc=stl["mfc"], lw=0,
                label=f"{stl['label']}: range/mean {rrange.loc[key, 'relative_range_pct']:.0f} %")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{dlab(w)}\n{wr.loc[w, 'Rs_W_m2']:.0f} W m$^{{-2}}$" for w in ws])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.6, 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Value / mean of the three windows")
    ax.legend(loc="upper left", fontsize=6.3)
    panel_label(ax, "(b) Raw and normalised rates, indexed to their mean")
    save(fig, "fig07_26-28Aug")


# ---------------------------------------------------------------------------
def fig_v1_v2():
    """Fig. 8 (revised): plate geometry and thermal state."""
    fig = plt.figure(figsize=(W2, 68 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    for i, w in enumerate(("W06", "W07", "W08")):
        st = STATE_STYLE[state(w)]
        e = wr.loc[w, "E_mm_h"]
        lo, hi = ci(w)
        ax.bar(i, e, width=0.6, color=st["color"], alpha=0.85, edgecolor="white", zorder=2)
        ax.errorbar(i, e, yerr=[[e - lo], [hi - e]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Uncovered\n24 Aug", "Cover V1\n24 Aug", "Cover V2\n25 Aug"], fontsize=6.6)
    ax.set_ylabel("Evaporation rate (mm h$^{-1}$)")
    ax.set_ylim(0, 1.7)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "(a) 872–916 W m$^{-2}$, ≈3.3 m s$^{-1}$")

    ax = fig.add_subplot(gs[1])
    data = [mc.window_slice(con, wr.loc[w, "start"], wr.loc[w, "end"]).T108_C_Avg.to_numpy() for w in ("W07", "W08")]
    parts = ax.violinplot(data, positions=[1, 0], orientation="horizontal", widths=0.7, showextrema=False)
    for pc, col in zip(parts["bodies"], (C_V1, C_V2)):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
        pc.set_edgecolor(col)
    ax.boxplot(data, positions=[1, 0], orientation="horizontal", widths=0.18, showfliers=False,
               medianprops=dict(color=INK, lw=1), boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7),
               capprops=dict(lw=0.7))
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["V1\n24 Aug", "V2\n25 Aug"])
    ax.axvspan(data[1].max(), data[0].min(), color="#F2E0D6", alpha=0.8, lw=0)
    ax.text(0.5 * (data[1].max() + data[0].min()), 0.5, f"gap\n{data[0].min() - data[1].max():.1f} K",
            ha="center", va="center", fontsize=6.3, color="#8A4B2A")
    ax.set_xlabel("Water temperature, probe 1 (°C)")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "(b) No overlap in thermal state")

    ax = fig.add_subplot(gs[2])
    for i, (key, stl) in enumerate(NORM_STYLE.items()):
        r = advc[(advc.contrast.str.startswith("cover V2 (25 Aug)")) & (advc.normaliser == key)].iloc[0]
        ax.plot([r.window_lo, r.window_hi], [i, i], color=stl["color"], lw=4.0, alpha=0.3, solid_capstyle="butt")
        errbar(ax, r.ratio, i, r.meas_lo, r.meas_hi, stl, ms=5, horizontal=True)
        ax.text(1.03, i, f"{r.ratio:.2f}\n[{r.window_lo:.2f}, {r.window_hi:.2f}]", fontsize=6.3, va="center",
                ha="left", transform=ax.get_yaxis_transform())
    ax.axvline(1, color=INK2, lw=0.7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Raw", "VPD-norm.", "Δe-norm."])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0.25, 1.85)
    ax.set_xlabel("Rate ratio V2 / V1")
    ax.plot([], [], color=INK2, lw=0.9, marker="|", ms=5, label="measurement 95 % CI")
    ax.plot([], [], color=INK2, lw=4.0, alpha=0.3, label="between-day 95 % CI")
    ax.legend(loc="lower right", fontsize=5.9, handlelength=1.4)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "(c) V2 relative to V1")
    save(fig, "fig08_V1_V2_thermal")


# ---------------------------------------------------------------------------
def fig_micrometer():
    """Supplementary: micrometer against level transducer."""
    long_iv = micdf[micdf.duration_h >= 2]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 64 * MM), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    ax.plot([0, 0.6], [0, 0.6], color=INK2, lw=0.8, ls=(0, (4, 3)))
    for r in long_iv.itertuples():
        ax.plot([r.E_transducer] * 2, [r.E_micrometer, r.E_micrometer_mass], color="#BDBDBD", lw=0.8, zorder=1)
    ax.plot(long_iv.E_transducer, long_iv.E_micrometer, "o", mfc="white", mec=INK2, ms=4.5, label="as read (height)")
    ax.plot(long_iv.E_transducer, long_iv.E_micrometer_mass, "o", color=C_V1, ms=4.5, label="mass-equivalent")
    a_raw = agr[(agr.set == "intervals >= 2 h") & (agr.micrometer == "E_micrometer")].iloc[0]
    a_m = agr[(agr.set == "intervals >= 2 h") & (agr.micrometer == "E_micrometer_mass")].iloc[0]
    ax.text(0.02, 0.585, f"Lin CCC {a_raw.ccc:.2f} → {a_m.ccc:.2f}\n[{a_m.ccc_lo:.2f}, {a_m.ccc_hi:.2f}]\n"
                         f"mean ratio {a_raw.geo_mean_ratio:.2f} → {a_m.geo_mean_ratio:.2f}\nn = {int(a_m.n)} intervals ≥ 2 h\n"
                         "open: as read (height)\nfilled: mass-equivalent",
            fontsize=6.2, va="top")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Transducer rate (mm h$^{-1}$)")
    ax.set_ylabel("Micrometer rate (mm h$^{-1}$)")
    panel_label(ax, "(a) Paired rates")

    ax = axes[1]
    x, y = long_iv.E_transducer.to_numpy(), long_iv.E_micrometer_mass.to_numpy()
    ax.plot((x + y) / 2, x - y, "o", color=C_V1, ms=4.5)
    ax.axhline(a_m.bias_mm_h, color=INK, lw=0.9)
    for v in (a_m.loa_lo, a_m.loa_hi):
        ax.axhline(v, color=INK2, lw=0.7, ls=(0, (4, 3)))
    ax.axhline(0, color="#BDBDBD", lw=0.6)
    ax.text(0.06, a_m.bias_mm_h + 0.008, mn(f"bias {a_m.bias_mm_h:+.3f}"), fontsize=6.2, ha="left")
    ax.text(0.06, a_m.loa_hi + 0.008, mn(f"+1.96 SD: {a_m.loa_hi:+.3f}"), fontsize=6.2, ha="left", color=INK2)
    ax.text(0.06, a_m.loa_lo - 0.022, mn(f"-1.96 SD: {a_m.loa_lo:+.3f}"), fontsize=6.2, ha="left", color=INK2)
    ax.set_xlim(0.05, 0.5)
    ax.set_ylim(-0.2, 0.3)
    ax.set_xlabel("Mean of the two rates (mm h$^{-1}$)")
    ax.set_ylabel("Transducer − micrometer (mm h$^{-1}$)")
    panel_label(ax, "(b) Bland–Altman, mass-equivalent")

    ax = axes[2]
    dd = long_iv.E_transducer - long_iv.E_micrometer_mass
    xs = np.linspace(0, 1.4, 50)
    for lab, ycol, colr, ls in (("mass-equivalent", "E_micrometer_mass", C_V1, "-"),
                                ("as read", "E_micrometer", INK2, (0, (4, 2)))):
        d_ = long_iv.E_transducer - long_iv[ycol]
        lr = stats.linregress(long_iv.dTw_dt_K_h, d_)
        ax.plot(xs, lr.intercept + lr.slope * xs, color=colr, lw=1.0, ls=ls,
                label=f"{lab}: {lr.slope:.2f} mm K$^{{-1}}$, p = {lr.pvalue:.2f}")
        ax.plot(long_iv.dTw_dt_K_h, d_, "o", ms=4.2, mfc=colr if ycol == "E_micrometer_mass" else "white",
                mec=colr, lw=0)
    ax.axhline(0, color="#BDBDBD", lw=0.6)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.12, 0.44)
    ax.set_xlabel("Warming rate of the water (K h$^{-1}$)")
    ax.set_ylabel("Transducer − micrometer (mm h$^{-1}$)")
    ax.legend(loc="upper left", fontsize=6.0)
    panel_label(ax, "(c) Difference against warming")
    save(fig, "figS1_micrometer_agreement")


# ---------------------------------------------------------------------------
def fig_sensitivity():
    """Supplementary: settling time, window bounds and reference window."""
    fig = plt.figure(figsize=(W2, 70 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1.0], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    adopted = {"W02": (pd.Timestamp("2026-08-18 11:20") - pd.Timestamp(mc.DEPLOYMENTS["W02"])).seconds / 60,
               "W07": (pd.Timestamp("2026-08-24 13:25") - pd.Timestamp(mc.DEPLOYMENTS["W07"])).seconds / 60}
    for w, g in settle.groupby("win"):
        base = wr.loc[w, "E_mm_h"]
        col = C_V1 if w == "W07" else C_SKY
        ax.plot(g.minutes_after_step, g.E_mm_h / base, color=col, lw=1.1, ls="-" if w == "W07" else (0, (3, 1.5)),
                label=f"{dlab(w)} (E = {base:.3f})")
        hw = g.cv_fixed_b * g.se_hac120
        ax.fill_between(g.minutes_after_step, (g.E_mm_h - hw) / base, (g.E_mm_h + hw) / base,
                        color=col, alpha=0.12, lw=0, label="95 % CI (HAC-120, fixed-b)" if w == "W07" else None)
        ax.axvline(adopted[w], color=col, lw=0.6, ls=(0, (1, 1.5)))
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_xlim(0, 90)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel("Window start, minutes after the level step")
    ax.set_ylabel("Rate / adopted rate")
    ax.legend(loc="upper right", fontsize=6.2)
    panel_label(ax, "(a) Settling time")

    ax = fig.add_subplot(gs[1])
    labels = {"C1": "18/17 Aug", "C2": "21/20 Aug", "C3": "24 Aug", "C4": "25/24 Aug\n(V2)"}
    offs = {"none": -0.22, "VPD": 0.0, "dE": 0.22}
    for i, cid in enumerate(("C1", "C2", "C3", "C4")):
        for key, stl in NORM_STYLE.items():
            r = bounds[(bounds.comparison == cid) & (bounds.normaliser == key)].iloc[0]
            x = i + offs[key]
            ax.plot([x, x], [r.S_min, r.S_max], color=stl["color"], lw=0.8)
            ax.plot([x, x], [r.S_p05, r.S_p95], color=stl["color"], lw=3.2, alpha=0.55, solid_capstyle="butt")
            ax.plot(x, r.S_baseline, marker=stl["marker"], mfc=stl["mfc"], mec=stl["color"], ms=4, lw=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels([labels[c] for c in ("C1", "C2", "C3", "C4")], fontsize=6.5)
    ax.set_ylim(45, 82)
    ax.set_ylabel("Suppression S (%)")
    ax.grid(axis="x", visible=False)
    hs = [mpl.lines.Line2D([], [], color=s["color"], marker=s["marker"], mfc=s["mfc"], lw=0, label=s["label"])
          for s in NORM_STYLE.values()]
    ax.legend(handles=hs, loc="lower left", fontsize=6.2)
    panel_label(ax, "(b) Window bounds (256 variants)")

    ax = fig.add_subplot(gs[2])
    pairs = refs[["regime", "covered", "uncovered"]].drop_duplicates().reset_index(drop=True)
    for j, p in pairs.iterrows():
        for key, stl in NORM_STYLE.items():
            r = refs[(refs.covered == p.covered) & (refs.uncovered == p.uncovered) & (refs.normaliser == key)].iloc[0]
            errbar(ax, r.S_pct, j + offs[key], r.ci_lo, r.ci_hi, stl, ms=3.8, horizontal=True)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([("24 Aug paired" if p.covered == "W07" and p.uncovered == "W06" else
                         f"{wr.loc[p.covered, 'start'].day}/{wr.loc[p.uncovered, 'start'].day} Aug")
                        + f" ({p.regime})" for p in pairs.itertuples()], fontsize=6.2)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_xlim(35, 85)
    ax.set_xlabel("Suppression S (%)")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "(c) Reference window")
    save(fig, "figS2_sensitivity")


# ---------------------------------------------------------------------------
def fig_night():
    """Supplementary: night-time behaviour of the level transducer."""
    fig = plt.figure(figsize=(W2, 80 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], hspace=0.15, wspace=0.32)
    n = nights[nights.start.dt.strftime("%Y-%m-%d") == "2026-09-01"].iloc[0]
    d = con[(con.TIMESTAMP >= n.start) & (con.TIMESTAMP <= n.end)]
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(d.TIMESTAMP, d.nivel_tanque_Avg, ".", ms=1.2, color=C_GREY, rasterized=True)
    ax1.plot(d.TIMESTAMP, d.nivel_tanque_Avg.rolling(61, center=True).mean(), color=C_V2, lw=1.4,
             label="61 min running mean")
    ax1.set_ylabel("Level (mm)")
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc="upper right", fontsize=6.2)
    panel_label(ax1, "(a) Night of 1–2 Sep: covered, lamps and fan off")
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for col, lab, colr, ls in (("T108_C_Avg", "probe 1", INK, "-"), ("T108_C_2_Avg", "probe 2", "#6E6E6E", (0, (5, 2))),
                               ("T108_C_3_Avg", "probe 3", "#9E9E9E", (0, (1, 1.2)))):
        ax2.plot(d.TIMESTAMP, d[col], color=colr, lw=1.0, ls=ls, label=lab)
    ax2.set_ylabel("Water T (°C)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.legend(loc="upper right", fontsize=6.0, ncol=3)
    ax2.set_xlabel("Local time")

    ax = fig.add_subplot(gs[:, 1])
    mk = {"uncovered": ("o", C_UNC), "V1": ("s", C_V1), "V2": ("D", C_V2)}
    for r in nights.itertuples():
        m, c = mk[r.cover]
        ax.plot(r.dT1_dt_K_h, r.E_app_mm_h, marker=m, ms=5, mfc=c if r.cover != "V2" or r.start >= pd.Timestamp("2026-08-26")
                else "white", mec=c, lw=0)
        key = f"{r.start.day}/{r.start.month}"
        off = {"21/8": (5, 4), "28/8": (6, -9), "18/8": (-18, 4), "17/8": (-4, -10)}.get(key, (4, 2))
        ax.annotate(key, (r.dT1_dt_K_h, r.E_app_mm_h), xytext=off, textcoords="offset points",
                    fontsize=5.8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    f6 = night_fit[(night_fit.subset.str.startswith("V2, 26")) & (night_fit.temperature == "dT1_dt_K_h")].iloc[0]
    f7 = night_fit[(night_fit.subset == "V2, all") & (night_fit.temperature == "dT1_dt_K_h")].iloc[0]
    ax.text(0.02, 0.98, f"E$_{{app}}$ = k Δe + c dT/dt\nV2, 26 plates (6 nights): c = {f6.c_mm_per_K:.2f} "
                        f"[{f6.c_lo:.2f}, {f6.c_hi:.2f}]\nall V2 (7 nights): c = {f7.c_mm_per_K:.2f} "
                        f"[{f7.c_lo:.2f}, {f7.c_hi:.2f}]", transform=ax.transAxes, fontsize=6.1, va="top", ha="left")
    hs = [mpl.lines.Line2D([], [], marker=m, color=c, lw=0, label={"uncovered": "uncovered", "V1": "cover V1",
                                                                    "V2": "cover V2, 26 plates"}[k])
          for k, (m, c) in mk.items()]
    hs.append(mpl.lines.Line2D([], [], marker="D", color=C_V2, mfc="white", lw=0, label="cover V2, 27 plates"))
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.0)
    ax.set_ylim(-0.012, 0.15)
    ax.set_xlabel("dT/dt of the water, probe 1 (K h$^{-1}$)")
    ax.set_ylabel("Apparent night-time rate (mm h$^{-1}$)")
    panel_label(ax, "(b) All dark periods")
    save(fig, "figS3_night_transducer")


# ---------------------------------------------------------------------------
def fig_coverage():
    """Supplementary: coverage of the 95 % rate intervals (coverage_check.py)."""
    path = T / "DS00_ci_coverage.csv"
    if not path.exists():
        print("  (figS4 skipped: run coverage_check.py first)")
        return
    cv = pd.read_csv(path)
    meth = [("cov_ols", "OLS, independent errors"), ("cov_hac_andrews_t", "HAC, Andrews AR(1) bandwidth"),
            ("cov_mbb60", "Moving-block bootstrap, 60 min"), ("cov_hac60_normal", "HAC 60 min, normal"),
            ("cov_sieve", "AR-sieve bootstrap"), ("cov_hac60_fixedb", "HAC 60 min, fixed-b (trend)"),
            ("cov_hac120_fixedb", "HAC 120 min, fixed-b (trend) — adopted")]
    fig, ax = plt.subplots(figsize=(W15, 62 * MM))
    for i, (col, lab) in enumerate(meth):
        v = cv[col].to_numpy()
        ax.plot(v, np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), "o", ms=3, color=C_V1, alpha=0.6)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.4)
    ax.axvline(0.95, color=C_UNC, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.953, -0.7, "nominal 0.95", fontsize=6.2, color=C_UNC)
    ax.set_yticks(range(len(meth)))
    ax.set_yticklabels([m[1] for m in meth], fontsize=6.6)
    ax.set_ylim(len(meth) - 0.5, -0.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Empirical coverage of the 95 % interval for the rate (one point per window)")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "Simulated records: fitted line + AR(p) noise + 0.1 mm quantisation")
    save(fig, "figS4_ci_coverage")


if __name__ == "__main__":
    print("Writing figures:")
    fig_method()
    fig_rates()
    fig_suppression()
    fig_gamma()
    fig_elasticity()
    fig_advection()
    fig_energy()
    fig_26_28()
    fig_v1_v2()
    fig_micrometer()
    fig_sensitivity()
    fig_night()
    fig_coverage()
