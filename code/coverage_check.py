#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAISSER-PE - coverage of the rate confidence intervals by simulation.

For every window an autoregressive model AR(p) (order by AIC, p <= 40) is
fitted to the residuals of the level regression. Synthetic level records are
then built as the fitted line plus simulated AR(p) noise, rounded to the
0.1 mm FP2 quantum exactly as the logger stores them, and each estimator's
95 % interval is checked against the known rate, including the AR-sieve
bootstrap adopted for the measurement CIs (its AR order is re-selected and
refitted on every synthetic record). The fitted AR models understate the
persistence of short, near-unit-root series, so these coverages are, if
anything, optimistic.

Writes ../results/tables/DS00_ci_coverage.csv. Takes a few minutes.

Author:  Jean Firmino Cardoso (jean.firmino@ufpe.br)
License: MIT
Usage:   python coverage_check.py
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

import maisser_core as mc

warnings.filterwarnings("ignore")
SEED = 20260924
N_REP = 200
N_BOOT = 499


def main():
    rng = np.random.default_rng(SEED)
    con = mc.load_consolidated()
    rows = []
    for win, start, end, *_ in mc.WINDOWS:
        sub = mc.window_slice(con, start, end)
        t = mc.hours_since_start(sub.TIMESTAMP)
        y = sub.nivel_tanque_Avg.to_numpy()
        n = len(y)
        tr = mc.ols_trend(t, y)
        e = np.asarray(tr["resid"])
        sel = ar_select_order(e, maxlag=40, ic="aic", trend="n")
        p = max(sel.ar_lags) if sel.ar_lags else 0
        ar = AutoReg(e, lags=p, trend="n").fit() if p else None
        phi = ar.params if p else np.array([])
        sig = np.sqrt(ar.sigma2) if p else np.std(e)
        fitted = np.asarray(tr["fit"].fittedvalues)
        true_e = tr["E"]
        hits = {k: 0 for k in ("ols", "hac_andrews_t", "hac60_normal", "hac60_fixedb",
                               "hac120_fixedb", "mbb60", "sieve")}
        cv60, cv120 = mc.fixed_b_cv_trend(n, 60), mc.fixed_b_cv_trend(n, 120)
        sieve_sd = []
        ests, se60 = [], []
        tc = t - t.mean()
        stt = np.sum(tc ** 2)
        for _ in range(N_REP):
            burn = 600
            z = lfilter([1.0], np.r_[1.0, -phi], rng.normal(0, sig, n + burn))[burn:]
            ys = np.round((fitted + z) * 10) / 10
            trs = mc.ols_trend(t, ys)
            eh = trs["E"]
            es = np.asarray(trs["resid"])
            ests.append(eh)
            err = abs(eh - true_e)
            hits["ols"] += err <= 1.96 * trs["se_ols"]
            rho = mc.lag1_autocorrelation(es)
            tq = stats.t.ppf(0.975, max(n * (1 - rho) / (1 + rho) - 2, 2))
            hits["hac_andrews_t"] += err <= tq * mc.hac_slope_se(t, es, mc.andrews_bandwidth(es))
            s60 = mc.hac_slope_se(t, es, 60)
            se60.append(s60)
            hits["hac60_normal"] += err <= 1.96 * s60
            hits["hac60_fixedb"] += err <= cv60 * s60
            hits["hac120_fixedb"] += err <= cv120 * mc.hac_slope_se(t, es, 120)
            # the adopted interval: AR order re-selected and refitted on each record
            sb, _ = mc.sieve_rates(t, ys, N_BOOT, rng)
            sieve_sd.append(np.std(sb, ddof=1))
            slo, shi = np.percentile(sb, [2.5, 97.5])
            hits["sieve"] += slo <= true_e <= shi
            idx = mc.circular_block_indices(n, 60, N_BOOT, rng)
            bs = eh - (es[idx] @ tc) / stt
            lo, hi = np.percentile(bs, [2.5, 97.5])
            hits["mbb60"] += lo <= true_e <= hi
        rows.append({"win": win, "n": n, "ar_order": p,
                     "true_sd_E": float(np.std(ests, ddof=1)),
                     "mean_se_hac60": float(np.mean(se60)),
                     "mean_se_sieve": float(np.mean(sieve_sd)),
                     **{f"cov_{k}": v / N_REP for k, v in hits.items()}})
        print(win, rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(mc.TABLES / "DS00_ci_coverage.csv", index=False, float_format="%.4g")
    print("\nMedian coverage across windows:")
    print(out.filter(like="cov_").median().round(3).to_string())
    print("Minimum coverage across windows:")
    print(out.filter(like="cov_").min().round(3).to_string())


if __name__ == "__main__":
    main()
