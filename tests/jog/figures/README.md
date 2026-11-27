# Figure scripts (JoG revision)

pygmt figures in the style of the published manuscript. Locations come from environment variables (see `fig_paths.py`):
`JOG_FIG_OUT` (output), `JOG_FIG_INPUTS` (prepared inputs), `JOG_RESULTS` / `JOG_RESULTS_V1` (outputs of `tests/jog/eval`),
`JOG_SHP` (basin shapefiles), `JOG_ESA_SING` (published ESA-SING folder). Environment: pygmt >= 0.13, geopandas, h5py.

| script | figure files |
|---|---|
| `plot_fig_revised.py` + `run_figs.py` | MetricIntro, MetricIntro60days, metric_comparison, mission_com_metric_{Brahmaputra,Danube}_v2, mission_com_metric_Brahmaputra_60days_v2 |
| `plot_fig_revised.py` + `run_cov_figs.py` | realistic_cov_DA, realistic_cov_DA_Danube |
| `brahma_cov_fig.py` | Brahma_cov |
| `response_surface_fig.py` | response_surface, response_surface_nse_rmse |
| `band_errors_fig.py` | band_errors |
| `eof_export.py` (data root) + `eof3_fig.py` | eof3_Brahmaputra, eof3_Danube |
| `inc_shares.py` (data root) + `increments_fig.py` | increments |
| `robustness_fig.py`, `si_small_figs.py` | robustness, SI_extension_b, SI_periodic_b, SI_seeds_b, SI_mismatch_shares |

`plot_fig_revised.py` is the published plotting code (ESAsing `paper_RSE/plot_fig_2.py`) with the metric maps read from the
`metrics_paper.h5` files of `tests/jog/eval/scripts/07_recompute_all.py` instead of recomputed; it imports `spatial_analysis` and
`paper_RSE.Analysis2` of the ESAsing code (add it to `PYTHONPATH`).
