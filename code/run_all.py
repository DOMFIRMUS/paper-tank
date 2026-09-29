#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - run the whole pipeline from the raw record.

  python run_all.py

1. process_evaporation.py   consolidated series and window rates
2. statistical_analysis.py  uncertainty, window-level model, tests, sensitivity
3. coverage_check.py        coverage of the rate intervals by simulation
4. make_figures.py          figures (PDF + 1200 dpi PNG)
5. make_fig01_schematic.py  schematic of the experimental system (Fig. 1)
6. make_tables.py           formatted tables (docx, xlsx)

Random numbers use fixed seeds, so every number is reproducible.
"""

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ("process_evaporation.py", "statistical_analysis.py", "coverage_check.py",
               "make_figures.py", "make_fig01_schematic.py", "make_tables.py"):
    print(f"\n######## {script}")
    runpy.run_path(str(HERE / script), run_name="__main__")
