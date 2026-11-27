# JoG evaluation (JOGE-D-26-00146, ESM2.0)

Evaluation of the 5-day GRACE-C / NGGM / MAGIC data assimilation into W3RA (Brahmaputra, Danube) for the revision.
Runs anywhere; set the data roots below (defaults: the PyGLDA v1 data root `/work/data_for_w3`). Results go to `results/`.

## Layout
```
jog_eval/config.py     roots, names, best configuration, paper period (epochs 634..875 = 2003-09-05..2006-12-27)
jog_eval/metrics.py    NSE, KGE, RMSE, r; definition='paper' (Appendix B) or 'published_code' (ESAsing/paper_RSE)
jog_eval/sources.py    truth / observations (0.5 deg global files), OL / DA (daily 0.1 deg -> 5-day, 0.5 deg),
                       basin series, products of runs made with the v1 data layout
jog_eval/evaluate.py   load one basin on common cells, Table-1 numbers, improvements
jog_eval/stats.py      block bootstrap, split sample, band-pass errors, record length
scripts/01_reproduce_table1.py   Table 1 and metrics.h5 from the published outputs (both definitions)
scripts/02_bootstrap_ci.py       R1.3  95 % CIs of the headline numbers and mission differences
scripts/03_split_sample.py       R1.2  choose the configuration on 2003-09..2004, evaluate on 2005-2006 (and reverse)
scripts/04_band_errors.py        R1.4 / R4.8  model, observation and DA error per frequency band
scripts/05_extension_R35.py      R3.5  1996-2006 Brahmaputra extension, CIs, record-length sensitivity
scripts/06_compare_variants.py   R3.4, R4.1, R4.6, R1.1, seeds: variant vs seeded baseline
scripts/07_recompute_all.py      every published grid number with both definitions (Table 1, abstract, Sect. 5.2,
                                 Figs 7-11, Table 2 via --results-root Analysis_cov_residuals) + side-by-side table
tests/test_eval.py               metrics = published functions, 5-day alignment = published, up-scaling = published
```

## Inputs (environment variables, defaults for UCloud)
| variable | default | content |
|---|---|---|
| `JOG_ESM2_ROOT` | `/work/data_for_w3/GRACE/ESA_SING/ESM2.0` | `globe/Global_EWHA_all/Global_EWH_05_{HIS,<mission>_<alpha>}.hdf5`, `globe/global_mask/<basin>/0.5/mask.h5`, `<basin>/truth/basin_signal*.hdf5` |
| `JOG_RESULTS_ROOT` | `<JOG_ESM2_ROOT>/Analysis` | published products `<basin>/{OL,DA}_mean/<case>/TotalWater.<year>.h5`, `Mask/<case>.h5` |
| `PYGLDA_V1_DATA` | `/work/data_for_w3` | new runs (`w3ra/cases`, `w3ra/domains`) |
| `JOG_OUT` | `results/` | outputs |

## Conventions (as in the paper)
- anomalies: every series is de-meaned over the evaluated period;
- truth = ESM2.0 HIS; OL/DA ensemble means averaged over each truth epoch and up-scaled to 0.5 deg by 5x5 block means;
- spatial means of NSE/KGE with values < -1 set to -1; RMSE = std of the difference (centred);
- Table 1 "improvement" = (mission - GRACE-C) and its ratio to (GRACE-C - model).

## Order of use
1. `python tests/test_eval.py` (no data needed except basin_signal for the epoch dates).
2. `python scripts/01_reproduce_table1.py --all-configs --metrics-h5 <ESAsing>/paper_RSE/res/metrics.h5`
   must reproduce metrics.h5 exactly with `published_code`; then compare with Table 1.
3. `python scripts/07_recompute_all.py --metrics-h5 <metrics.h5>` (all 180 configurations, HPF 0/1), then
   `--results-root <ESM2.0>/Analysis_cov_residuals --tag cov_residual --best-only` for Table 2.
4. Steps 02-04 on the published outputs (no new runs needed).
5. Steps 05-06 when the revision runs exist.
