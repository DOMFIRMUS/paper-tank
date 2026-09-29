#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - Fig. 1, schematic of the experimental system.

(a) Vertical section through the rig: lamp array, fan, Class A pan with the
floating hexagonal plates and the position of every sensor used in the
analysis. (b) Plan view of the pan with a 27-plate tessellation (77.0 %
cover). (c) The hydrostatic step that follows plate deployment, which is
excluded from every rate. Numbered call-outs are explained in the key.

The plan view is to scale (pan 1.185 m, plates 110 mm side); the section is
schematic, with the vertical scale exaggerated.
Written to ../results/figures/ with the same conventions as make_figures.py
(174 mm wide, vector PDF + 1200 dpi PNG + preview).

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
Project: MAISSER-PE, FACEPE APQ-1428-3.01/25
License: MIT
Usage:   python make_fig01_schematic.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Circle, RegularPolygon

import maisser_core as mc

MM = 1 / 25.4
W2 = 174 * MM

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.pad_inches": 0.02, "mathtext.default": "regular",
})

INK, INK2 = "#1A1A1A", "#595959"
WATER = "#CFE8F6"
STEEL = "#8C8C8C"
PLATE, PLATE_E = "#FFFFFF", "#0072B2"
LAMP = "#E69F00"
AIR = "#009E73"
SENS = "#D55E00"

R_PAN = 592.5                        # pan radius, mm
A_HEX = 110.0                        # plate side, mm

KEY = [
    "Lamp array: 18 infrared lamps × 250 W, dimmer-controlled",
    "Fan: u = 0, ≈1.9–2.2 or 3.2–3.6 m s$^{-1}$ over the pan",
    "Rigid floating hexagonal plates, 110 mm side",
    "Air temperature and humidity (HMP45C) on the lamp frame",
    "Pyranometer on a float at the surface (SP-LITE; LI200X)",
    "Water temperature, three thermistors (108)",
    "Water level, vented pressure transducer on the floor (QDY30A)",
    "Cup anemometer at the rim (034B)",
    "Micrometer on a stilling well (independent level check)",
    "CR1000 datalogger: 20 s scan, 1 min storage",
]


def callout(ax, n, x, y):
    ax.add_patch(Circle((x, y), 1.75, fc="white", ec=INK, lw=0.6, zorder=10))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=11)


def leader(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=INK2, lw=0.5, zorder=9)


def section(ax):
    """Schematic section in drawing units (x 0-100, y 0-66)."""
    xl, xr, yb, yt, yw = 22.0, 78.0, 10.0, 28.0, 25.0
    # stand
    for x in (25, 75):
        ax.plot([x, x], [yb, 3.5], color=STEEL, lw=1.3)
    ax.plot([21, 79], [3.5, 3.5], color=STEEL, lw=1.3)
    # water and pan
    ax.add_patch(Rectangle((xl, yb), xr - xl, yw - yb, fc=WATER, ec="none", zorder=1))
    ax.plot([xl, xl, xr, xr], [yt, yb, yb, yt], color=STEEL, lw=2.4, solid_joinstyle="miter", zorder=3)
    # plates (left part of the surface drawn bare to show the float and the thermistors)
    for x0 in np.arange(35.0, 77.0, 6.9):
        ax.add_patch(Rectangle((x0, yw - 0.9), 6.3, 1.7, fc=PLATE, ec=PLATE_E, lw=0.8, zorder=5))
    callout(ax, 3, 60.5, 31.5)
    leader(ax, 60.5, 29.8, 60.5, yw + 0.8)
    # lamps and rays
    yl = 58.0
    ax.plot([16, 84], [yl + 4.2, yl + 4.2], color=STEEL, lw=1.5)
    for x in np.linspace(27, 73, 6):
        ax.add_patch(Polygon([[x - 3.2, yl], [x + 3.2, yl], [x + 1.6, yl + 3.6], [x - 1.6, yl + 3.6]],
                             closed=True, fc=LAMP, ec=INK, lw=0.5, zorder=4))
        for dx in (-1.6, 0, 1.6):
            ax.add_patch(FancyArrowPatch((x + dx, yl - 0.8), (x + 1.4 * dx, 33.5), arrowstyle="-|>",
                                         mutation_scale=4.5, color=LAMP, lw=0.55, alpha=0.75, zorder=2))
    callout(ax, 1, 88.5, 60.2)
    leader(ax, 86.8, 60.6, 84.0, 62.2)
    # HMP45C shield hanging from the frame
    ax.plot([17, 17], [yl + 4.2, yl + 1.5], color=STEEL, lw=0.8)
    ax.add_patch(Rectangle((15.2, yl - 3.0), 3.6, 4.5, fc="#EFEFEF", ec=INK, lw=0.6, zorder=6))
    for k in range(4):
        ax.plot([15.2, 18.8], [yl - 2.0 + 0.9 * k] * 2, color=INK2, lw=0.35, zorder=7)
    callout(ax, 4, 10.5, 56.0)
    leader(ax, 12.2, 56.0, 15.2, 56.2)
    # fan and flow
    ax.add_patch(Rectangle((2.0, 26.0), 5.0, 15.0, fc="#E6E6E6", ec=INK, lw=0.7, zorder=4))
    ax.add_patch(Circle((4.5, 33.5), 2.2, fc="none", ec=INK2, lw=0.6, zorder=5))
    for yy in (28.5, 31.0, 33.5, 36.0, 38.5):
        ax.add_patch(FancyArrowPatch((8.0, yy), (31.0, 27.2 + 0.25 * (yy - 28.5)), arrowstyle="-|>",
                                     mutation_scale=6, color=AIR, lw=0.9, zorder=6,
                                     connectionstyle="arc3,rad=0.06"))
    callout(ax, 2, 4.5, 44.5)
    # pyranometer on its float
    ax.add_patch(Rectangle((26.5, yw - 0.7), 4.0, 1.2, fc="#BDBDBD", ec=INK, lw=0.5, zorder=6))
    ax.add_patch(Rectangle((27.6, yw + 0.5), 1.8, 1.0, fc=SENS, ec=INK, lw=0.5, zorder=7))
    callout(ax, 5, 24.0, 44.0)
    leader(ax, 24.6, 42.4, 28.3, yw + 1.6)
    # thermistors
    for x, y in ((30.5, 21.5), (32.5, 17.0), (34.5, 12.8)):
        ax.plot(x, y, marker="o", ms=3.4, mfc=SENS, mec=INK, mew=0.5, zorder=8)
    ax.plot([29.2, 36.0], [23.8, 11.8], color=INK2, lw=0.4, ls=(0, (2, 1.5)), zorder=7)
    callout(ax, 6, 16.5, 17.0)
    leader(ax, 18.2, 17.0, 31.8, 17.0)
    # level transducer on the floor, cable to the logger
    ax.plot(52, yb + 0.9, marker="D", ms=4.2, mfc=SENS, mec=INK, mew=0.5, zorder=8)
    ax.plot([52, 52, 88], [yb, 7.0, 7.0], color=INK2, lw=0.5, zorder=2)
    ax.plot([88, 88], [7.0, 9.0], color=INK2, lw=0.5, zorder=2)
    callout(ax, 7, 52, 15.5)
    leader(ax, 52, 13.8, 52, yb + 1.9)
    # anemometer at the rim
    ax.plot([80.5, 80.5], [yt - 2, 33.0], color=STEEL, lw=0.9)
    ax.plot([78.5, 82.5], [33.0, 33.0], color=INK, lw=0.6)
    for x in (78.5, 82.5):
        ax.add_patch(Circle((x, 33.0), 0.6, fc=SENS, ec=INK, lw=0.4, zorder=6))
    callout(ax, 8, 87.0, 37.0)
    leader(ax, 85.4, 36.2, 83.0, 33.6)
    # stilling well and micrometer
    ax.add_patch(Rectangle((68.2, yb + 1.0), 3.0, 21.5, fc="none", ec=INK2, lw=0.6, ls=(0, (2, 1.4)),
                           zorder=6))
    ax.plot([69.7, 69.7], [yw + 0.1, 37.0], color=INK, lw=1.0, zorder=7)
    ax.add_patch(Rectangle((68.3, 37.0), 2.8, 2.2, fc="#D9D9D9", ec=INK, lw=0.6, zorder=7))
    callout(ax, 9, 66.0, 44.0)
    leader(ax, 66.8, 42.4, 68.9, 39.2)
    # logger
    ax.add_patch(Rectangle((88.0, 9.0), 10.0, 13.0, fc="#F4F4F4", ec=INK, lw=0.7, zorder=4))
    ax.text(93.0, 15.5, "CR1000", ha="center", va="center", fontsize=7.5, rotation=90, zorder=6)
    callout(ax, 10, 93.0, 25.5)
    # dimensions
    ax.annotate("", (xl, 6.0), (xr, 6.0), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6,
                                                          shrinkA=0, shrinkB=0))
    ax.text(38.0, 6.5, "Ø 1.185 m", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.annotate("", (xr + 3.2, yb), (xr + 3.2, yt), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6,
                                                                    shrinkA=0, shrinkB=0))
    ax.text(xr + 4.0, 19.0, "0.25 m", ha="left", va="center", fontsize=7.5, color=INK2, rotation=90)
    ax.text(50, 0.3, "Class A evaporation pan, free surface 1.1029 m$^2$ (vertical scale exaggerated)",
            ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    ax.set_title("(a) Vertical section", loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=2)


def key(ax):
    ax.axis("off")
    half = (len(KEY) + 1) // 2
    for i, txt in enumerate(KEY):
        col, row = divmod(i, half)
        x, y = 0.01 + 0.52 * col, 0.92 - row * 0.21
        ax.text(x, y, str(i + 1), transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
                fontweight="bold", bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=INK, lw=0.6))
        ax.text(x + 0.022, y, txt, transform=ax.transAxes, ha="left", va="center", fontsize=7.5)


def plate_centres(n):
    """Centres of flat-topped hexagons (side a) on a touching hexagonal lattice,
    keeping those that lie entirely inside the pan. The lattice offset
    (44.0, 95.3) mm was found by a grid search; it is the one that fits the
    27 plates of the experiment."""
    a = A_HEX
    dx, dy = 1.5 * a, np.sqrt(3) * a
    ang = np.arange(6) * np.pi / 3
    pts = []
    for i in range(-8, 9):
        for j in range(-8, 9):
            x = i * dx + 44.0
            y = j * dy + (dy / 2 if i % 2 else 0) + 95.26
            if np.all(np.hypot(x + a * np.cos(ang), y + a * np.sin(ang)) <= R_PAN):
                pts.append((x, y))
    assert len(pts) >= n
    return pts[:n]


def plan(ax):
    ax.add_patch(Circle((0, 0), R_PAN, fc=WATER, ec=STEEL, lw=2.0))
    for x, y in plate_centres(27):
        ax.add_patch(RegularPolygon((x, y), 6, radius=A_HEX * 0.97, orientation=np.pi / 6,
                                    fc=PLATE, ec=PLATE_E, lw=0.6))
    frac = 27 * 1.5 * np.sqrt(3) * (A_HEX / 1000) ** 2 / (np.pi * (R_PAN / 1000) ** 2)
    for yy in (180, 0, -180):
        ax.add_patch(FancyArrowPatch((-R_PAN - 300, yy), (-R_PAN - 50, yy), arrowstyle="-|>",
                                     mutation_scale=6, color=AIR, lw=0.9))
    ax.text(0, -R_PAN - 60, f"27 plates: {100 * frac:.1f} % cover\n26 plates: {100 * frac * 26 / 27:.1f} %",
            ha="center", va="top", fontsize=7.5)
    ax.set_xlim(-R_PAN - 320, R_PAN + 20)
    ax.set_ylim(-R_PAN - 300, R_PAN + 20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) Plan view, to scale", loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=2)


def step(ax):
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, 5.8)
    ax.axvspan(20, 40, color="#9E9E9E", alpha=0.22, lw=0)
    ax.plot([0, 20], [0.55, 0.3], color=SENS, lw=1.1)
    ax.plot([20, 20], [0.3, 4.1], color=INK, lw=0.8, ls=(0, (1.5, 1)))
    ax.plot([20, 40], [4.1, 3.95], color=INK2, lw=0.8)
    ax.plot([40, 60], [3.95, 3.7], color=PLATE_E, lw=1.1)
    ax.annotate("", (18.5, 0.35), (18.5, 4.1), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6,
                                                           shrinkA=0, shrinkB=0))
    ax.text(21.5, 2.2, "buoyancy\nstep\n3.2–3.9 mm", ha="left", va="center", fontsize=7.5)
    ax.text(30, 5.7, "excluded", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.text(9.5, 0.8, "uncovered", ha="center", va="bottom", fontsize=7.5, color=SENS)
    ax.text(51, 4.1, "covered", ha="center", va="bottom", fontsize=7.5, color=PLATE_E)
    ax.set_xlabel("time", fontsize=7.5, labelpad=1)
    ax.set_ylabel("water level", fontsize=7.5, labelpad=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.set_title("(c) Plate deployment", loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=2)


def main():
    fig = plt.figure(figsize=(W2, 112 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.35, 1], height_ratios=[2.55, 1], wspace=0.04, hspace=0.12)
    gsr = gs[0, 1].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.35)
    section(fig.add_subplot(gs[0, 0]))
    plan(fig.add_subplot(gsr[0]))
    step(fig.add_subplot(gsr[1]))
    key(fig.add_subplot(gs[1, :]))
    name = "fig01_schematic"
    # rescale the canvas until the saved (tight) PDF page is exactly 174 mm wide
    import pymupdf
    pdf = mc.FIGURES / f"{name}.pdf"
    for _ in range(8):
        fig.savefig(pdf, bbox_inches="tight")
        with pymupdf.open(pdf) as d:
            width = d[0].rect.width / 72
        if abs(width - W2) < 0.2 * MM:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * W2 / width, h * W2 / width)
    fig.savefig(mc.FIGURES / f"{name}.png", dpi=1200, bbox_inches="tight")
    fig.savefig(mc.FIGURES / f"{name}_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} written")


if __name__ == "__main__":
    main()
