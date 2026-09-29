# paper-tank — MAISSER-PE evaporation-suppression experiment

Raw data, analysis code, figures and tables from an experiment on **evaporation suppression by modular hexagonal floating covers**. The measurements were made on an instrumented Class A evaporation pan under a radiative simulator, with forced advection. Recife, Brazil, August–September 2026.

- **Project:** MAISSER-PE — *Materiais Inovadores e Sustentáveis para Supressão da Evaporação em Reservatórios no Semiárido de Pernambuco*
- **Funding:** FACEPE, grant APQ-1428-3.01/25
- **Institution:** Departamento de Energia Nuclear (DEN), Universidade Federal de Pernambuco, Recife, PE, Brazil (8.0476° S, 34.8769° W, 10 m a.s.l.)
- **Contact:** Jean Firmino Cardoso, jean.firmino@ufpe.br (ORCID 0000-0001-6092-713X)
- **Dataset DOI (Zenodo, first version):** 10.5281/zenodo.22844940

Everything in `data/processed/` and `results/` is regenerated from the raw logger files with one command (see *Reproduce*).

---

## Repository layout

```
code/
  run_all.py                  runs the whole pipeline in order
  maisser_core.py             shared definitions: windows, comparisons, estimators
  process_evaporation.py      raw TOA5 -> consolidated series and window rates
  statistical_analysis.py     uncertainty, window-level model, tests, sensitivity
  coverage_check.py           simulation of confidence-interval coverage
  make_figures.py             figures (vector PDF + 1200 dpi PNG)
  make_fig01_schematic.py     schematic of the experimental system (Fig. 1)
  make_tables.py              formatted tables (docx, xlsx)
  requirements.txt
  datalogger/codigo_sensores_v8.CR1   CRBasic program run by the CR1000
  legacy/process_evaporation_v1.py    first-version script (provenance only)
data/
  raw/                        CR1000 TOA5 exports, unmodified; manual micrometer readings
  commissioning/              instrument set-up records of 12-13 Aug (used in no result)
  processed/                  window_rates.csv (consolidated_series.csv is regenerated)
results/
  figures/                    *.pdf (vector) and *_preview.png
  tables/                     T3-T6 and DS00-DS25 (CSV), MAISSER-PE_tables.docx/.xlsx,
                              analysis_log.txt, results_summary.json
docs/
  photos/                     20 photographs of the apparatus (captions in PHOTO_CAPTIONS.md)
  CR1000_wiring_diagram.pdf   wiring diagram
  CR1000_wiring_text.pdf      terminal-by-terminal wiring list
LICENSE-code.txt              MIT (code/)
LICENSE-data.txt              CC BY 4.0 (data/, results/, docs/)
```

## Reproduce

```
cd code
pip install -r requirements.txt
python run_all.py
```

Requires Python ≥ 3.9 with numpy, pandas, scipy, statsmodels, matplotlib, python-docx and openpyxl. The full run takes about 5 minutes; about 4 of them are the coverage simulation. Random numbers use fixed seeds, so the results are reproducible. The 1200 dpi PNG versions of the figures are not versioned; `make_figures.py` writes them next to the PDFs.

---

## Experimental system

| Item | Specification |
|---|---|
| Evaporating surface | Class A pan, 1.185 m internal diameter, 0.25 m deep; free surface 1.1029 m² |
| Floating cover | rigid hexagonal plates, 110 mm side (0.031437 m² each); 27 plates = 77.0 %, 26 plates = 74.1 % |
| Plate versions | V1 (baseline geometry), V2 (modified geometry, from 25 Aug) |
| Radiative simulator | 18 infrared drying lamps, 250 W each (4500 W), dimmer-controlled |
| Forced advection | fan discharging horizontally across the water surface |
| Datalogger | Campbell Scientific CR1000, 20 s scan, 1 min storage (program `code/datalogger/codigo_sensores_v8.CR1`) |
| Water level | QDY30A hydrostatic (vented) pressure transducer, single-ended channel |
| Irradiance | Kipp & Zonen SP-LITE in the water-surface plane; LI-COR LI200X as secondary |
| Air T and RH | Vaisala HMP45C, mounted at the top of the rig until 10 Sep |
| Water temperature | three Campbell 108 thermistors |
| Air velocity | Campbell 034A/034B cup anemometer |
| Independent level check | micrometer on an acrylic stilling well (manual readings) |

## Raw data

### Logger exports (`data/raw/*.dat`)
Five TOA5 exports from the same program and with the same columns, deposited as exported. Together they form a continuous 1 min series from 14 Aug 10:09 to 11 Sep 18:00. Consecutive exports overlap at their boundaries; the processing concatenates them and keeps the first occurrence of each timestamp.

| File | Period | Records |
|---|---|---|
| `CR1000_tanque_TOA5_2026-08-14.dat` | 14 Aug 10:09–18:19 | 361 |
| `CR1000_tanque_TOA5_2026-08-17.dat` | 17 Aug 08:30–17:29 | 540 |
| `CR1000_tanque_TOA5_2026-08-17_2026-08-18.dat` | 17 Aug 17:30 – 18 Aug 17:50 | 1 461 |
| `CR1000_tanque_TOA5_2026-08-18_2026-09-10.dat` | 18 Aug 17:51 – 10 Sep 16:15 | 31 737 |
| `CR1000_tanque_TOA5_2026-09-10_2026-09-11.dat` | 10 Sep 16:16 – 11 Sep 18:00 | 79 |

Gaps longer than 5 min:
- 14 Aug 10:12–12:22;
- 14 Aug 18:19 – 17 Aug 08:30 (weekend, system off);
- 9 Sep 16:50 – 10 Sep 14:14;
- 10 Sep 17:28 – 11 Sep 17:55.

**On 10 Sep the sensors were repositioned** and pH and conductivity probes were added. Records from that date onwards describe a different instrument geometry and are used in no analysis window.

TOA5 has four header lines: metadata, column names, units and aggregation. Timestamps are local time (UTC−3) and mark the end of each 1 min interval.

| Column | Unit | Meaning |
|---|---|---|
| `TIMESTAMP`, `RECORD` | — | time stamp and record number |
| `BattV_Avg`, `PTemp_C_Avg` | V, °C | logger battery and panel temperature |
| `irradiancia_Avg` | W m⁻² | SP-LITE irradiance |
| `WS_ms_Avg` | m s⁻¹ | air velocity over the pan |
| `T108_C_Avg`, `T108_C_2_Avg`, `T108_C_3_Avg` | °C | water temperature, thermistors 1 (reference), 2 and 3 |
| `nivel_tanque_Avg` / `_Max` / `_Min` / `_Std` | mm | water level (primary variable) and its within-minute statistics |
| `AirTC_Avg` | °C | air temperature |
| `RH_Min` | % | relative humidity, **minimum** of the interval, as logged |
| `ETos`, `Rso` | mm, — | on-board reference ET outputs; not used |
| `irradiancia_licor_Avg`, `irradiacaototal_Tot` | W m⁻², MJ m⁻² | LI200X irradiance and total |

**Missing-data sentinels.** `-99.999`, `NAN` and, in the commissioning files, `-100` mean that no reading was available. The processing converts them to missing values.

### Micrometer readings (`data/raw/micrometer_readings.csv`)
Fifty-five manual readings of the water surface on eleven days between 17 Aug and 3 Sep. Larger values mean a higher surface. Columns: `date`, `time`, `reading_mm`, `n_plates`, `cover_version`, `note`.

Two readings bracket a buoyancy step directly: plates deployed on 18 Aug 10:14 → 10:45 (+3.907 mm) and plates removed on 1 Sep 13:38 → 13:44 (−3.220 mm).

Two intervals should not be used:
- 17 Aug, the 12:20 reading, which is non-monotonic;
- 21 Aug 07:49 → 08:05, which spans a refill.

### Commissioning records (`data/commissioning/`)
Five short exports from 12–13 Aug, made while the system was being set up (program versions v1, v7 and v8). They support no result and are kept as provenance: the level transducer first appears in v7, and the HMP45C returns `-100` throughout 12 Aug.

## Experimental windows

Covered windows start after the buoyancy step that follows plate deployment and its settling. Refills fall outside every window.

| ID | Start | End | Cover | Plates | Fan regime |
|---|---|---|---|---|---|
| W01 | 2026-08-17 09:20 | 17:26 | none | 0 | weak (≈1.9–2.2 m s⁻¹) |
| W02 | 2026-08-18 11:20 | 18:20 | V1 | 27 | weak |
| W03 | 2026-08-19 10:20 | 15:41 | none | 0 | weak |
| W04 | 2026-08-20 14:35 | 18:24 | none | 0 | strong (3.2–3.6 m s⁻¹) |
| W05 | 2026-08-21 11:12 | 15:44 | V1 | 27 | strong |
| W06 | 2026-08-24 07:57 | 13:00 | none | 0 | strong |
| W07 | 2026-08-24 13:25 | 18:01 | V1 | 27 | strong |
| W08 | 2026-08-25 11:25 | 17:18 | V2 | 27 | strong |
| W09 | 2026-08-26 09:47 | 18:49 | V2 | 26 | strong |
| W10 | 2026-08-27 09:22 | 17:04 | V2 | 26 | strong |
| W11 | 2026-08-28 07:57 | 16:55 | V2 | 26 | strong |
| W12 | 2026-08-31 08:45 | 17:37 | V2 | 26 | fan off |
| W13 | 2026-09-01 08:35 | 13:35 | V2 | 26 | fan off |
| W14 | 2026-09-02 07:56 | 16:30 | V2 | 26 | fan off |
| W15 | 2026-09-03 08:08 | 16:39 | V2 | 26 | fan off |

The weak-fan windows also ran at a much lower lamp setting than the strong-fan windows.

## Analysis in brief

- **Rate of a window:** the negative OLS slope of the 1 min level record. Its measurement uncertainty is a Newey–West standard error (120 min bandwidth) with fixed-b critical values simulated for a trend slope. `coverage_check.py` compares seven interval methods in simulated records that have the dependence of the real residuals (DS00, Fig. S4).
- **Suppression:** S = 100 (1 − E_covered / E_uncovered). It is reported raw and with each rate divided by the air vapour pressure deficit (VPD) or by the surface–air vapour pressure difference Δe = e_s(T_w) − e_a of its own window.
- **Primary inference:** a window-level model, ln(E/X) ~ cover × regime, with the window (day) as the experimental unit (Table 4). It also gives a mixed model on block rates, a minimum detectable difference, an equivalence test and a prediction interval.
- **Checks:**
  - within-window elasticity of E to VPD and Δe (Table 5, Fig. B) and the suppression profile against the elasticity (Fig. C);
  - exact tests for the paired run of 24 Aug (Table 6);
  - sensitivity to window bounds, settling time, reference window, thermistor and sensor accuracy;
  - micrometer against transducer agreement;
  - a night-time check of the level transducer.

`results/tables/analysis_log.txt` lists every number. The README sheet of `results/tables/MAISSER-PE_tables.xlsx` describes each table file.

## Known limitations of the data

- Measurements were made indoors under infrared lamps, so absolute rates are not open-field rates.
- **Irradiance:** the SP-LITE and LI200X are silicon cells (400–1100 nm) calibrated to sunlight. Most of the lamps' output lies beyond 1100 nm, so the logged irradiance is an index of lamp output, not the radiant load. Lamp counts noted in the field are not used.
- **Level transducer:** its temperature response has not been bench-tested. On some nights the level rose slightly while the covered pan could only lose water. See Table S7 and Fig. S3.
- **Water temperature:** positions of the three thermistors before 10 Sep are not documented. Thermistor 2 reads up to ~11 K warmer than the others under the lamps and behaves as a near-surface probe.
- **Air temperature and humidity:** the HMP45C sat at the top of the rig, near the lamps, until 10 Sep. Only the per-minute *minimum* of RH was logged.
- **Level record resolution:** stored as FP2, with a 0.1 mm quantum; the per-scan ADC step is ≈0.14 mm.
- **Replication:** treatments are replicated in time, across windows and days. There is no uncovered window with the fan off, and no uncovered reference for the 26-plate V2 cover.
- **The 14 Aug record** is interrupted by a refill and yields no window.

## Licence

- Code (`code/`): MIT, see `LICENSE-code.txt`.
- Data, results and documentation (`data/`, `results/`, `docs/`): CC BY 4.0, see `LICENSE-data.txt`.

## How to cite

Please cite the Zenodo dataset (DOI above) and this repository.
