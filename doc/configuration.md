# PyGLDA v1 — configuration guide

Describes the code with the "run cleanly" changes of V1 reviews 14-16 (data root, settings run copy, start-date
check, error logs, and the data layout domains/units/cases/aggregates with `made_with.json` reuse rules; numbers
unchanged, bit-identical to commit `f3f5263`). Items marked **[issue n]** behave
differently from what one would expect; see `known_issues.md`.

An experiment is described by a driver script and four JSON files in one settings folder:

| file | purpose | read by |
|---|---|---|
| driver, e.g. `demo/demo_3.py` | case name, basin and shapefile, ensemble size, the three periods, which stages run | `src_FlowControl/Regional_DA.py` (class `RDA`) |
| `settings/<folder>/setting.json` | W3RA run: domain box, initialisation, period, input folders, output folder and variables | `src_GHM/config_settings.py` |
| `settings/<folder>/pre_process.json` | cropping of the global inputs (forcing, climatology, parameters, mask) to the box | `src_GHM/preprocess.py` |
| `settings/<folder>/perturbation.json` | ensemble generation: perturbation of forcing and parameters | `src_DA/Perturbation.py` |
| `settings/<folder>/DA_setting.json` | assimilation: period, basin, state vector, GRACE product and folders, filter name | `src_DA/configure_DA.py` |

`settings/SummerSchool/Installation` is the smallest complete example (Brahmaputra, 3 sub-basins, 2005, 3 members).
`demo/demo_3.py` is the Danube example (`DRB`, 3 sub-basins, or `GDRB`, 19 grid units; 2 members, 2005, sample data);
`settings/demo_3/DA_setting.json` still names basin `Tile81` from an earlier run; the driver's basin replaces it in
the run copy (the files in `settings/` are no longer written). Ready-made Danube settings in the current format:
`settings/Danube_DRB`, `settings/Danube_GDRB` with the driver `demo/demo_danube.py` (CSR mascons, 30 members,
2002-2006, `${DATA_ROOT}` paths). The published Danube runs
(30 members) are the UCloud cases `RDA_DRB_BasinScale` and `RDA_GDRB_BasinScale` (results in `/work/data_for_w3/w3ra/res`).

## 0. How the driver and the JSON files interact

The driver sets class attributes of `RDA` and calls the stages:

```python
RDA.setting_dir = '../settings/demo_3'
RDA.ens = 30                       # perturbed members; MPI needs ens + 1 ranks
RDA.case = 'case_study_demo3'      # names all folders of this experiment
RDA.basin = 'DRB'
RDA.shp_path = 'DRB_3_shapefiles/DRB_subbasins.shp'   # relative to <data_root>/basin/shp (or absolute)
RDA.cold_begin_time, RDA.cold_end_time = '2005-01-01', '2005-12-31'      # spin-up (single run)
RDA.warm_begin_time, RDA.warm_end_time = '2005-01-02', '2005-12-31'      # open loop
RDA.resume_begin_time, RDA.resume_end_time = '2005-04-01', '2005-12-31'  # assimilation
```

**Data root.** `RDA.data_root` (default: environment variable `PYGLDA_V1_DATA`; without either the stages stop with
a clear error) is the root of `ERA5/`, `GRACE/`, `basin/` and `w3ra/`. Paths in the JSON files may be written as
`${DATA_ROOT}/...`. The repository holds no data: shapefiles are in `<data_root>/basin/shp/` (formerly the repository
`data/basin/shp`, `data/subbasins` and `res/`), the GRACE land mask in `<data_root>/GRACE/masks/`. A settings file or
driver that still names `../data/...` or `../res/...` stops with an error that gives the new place.

**Run copy.** At the start of every stage the four JSON files of `RDA.setting_dir` are copied, with `${DATA_ROOT}`
substituted, to `<data_root>/w3ra/cases/<case>/settings/` (legacy layout: `<data_root>/w3ra/runs/<case>/settings/`), and the
paths of the shared folders are pointed to the layout (section 7); the stage and `generate_settings(mode)` read and write only
that copy, so the files in `settings/` are never written (issue 9 fixed). `generate_settings` sets these keys so that
the copy agrees with the driver:

| file | keys written by the driver |
|---|---|
| `pre_process.json` | `date_begin`, `date_end` (from the stage period) |
| `setting.json` | `bounds.lat/lon` (shapefile box + 0.2°), `bounds.prefix` = case, `input.*` = `pre_process.out_dir`, `init.spec` = case, `init.mode`; `init.spinup` (cold) or `run.fromdate/todate` and `init.date` = day before (warm/resume) |
| `perturbation.json` | `dir.in`, `dir.out` = `<pre_process.out_dir>/<case>`, `ensemble` |
| `DA_setting.json` | `basic.ensemble`, `fromdate`, `todate`, `basin`; `basic.basin_shp` = the driver's `shp_path` if its file name starts with `<basin>_`, else the only `<basin>_…subbasins….shp` below `<data_root>/basin/shp/` (several byte-identical copies are accepted; different ones stop with a list) |

It also records the folder layout in `setting.json` (`output.layout` = `domains` or `legacy`, `output.stage` =
`OL`/`DA`). Everything else is taken as written. `RDA.config_external_data()` (optional, for old settings with fixed
paths) rewrites, in the run copy only, every path that contains `/w3ra/`, `/GRACE/` or `/ERA5/` so that it starts with
the data root (`RDA.data_root`, else the deprecated `RDA.external_data_path`).

**Result folders.** `RDA.res_output` / `RDA.figure_output` are the roots of the collected results and figures
(layout: `cases/<case>/res` and `cases/<case>/figure`; legacy default `<data_root>/w3ra/res` and `…/figure` plus the
case name; issue 13 fixed). The old module
variables `Regional_DA.res_output/figure_output` are still honoured, with a deprecation warning, if a driver sets them.

**Layout.** `RDA.legacy_layout = False` (default) uses the layout domains/units/cases/aggregates (section 7);
`True` keeps the old places (`crop_input/<case>`, shared `state_<case>_ensemble_<k>` folders, `../temp`, `../log/OL`,
masks in `<data_root>/w3ra/legacy_repo/basin_mask`), needed to read the old cases.

**Errors.** Every stage is guarded: an exception on any rank prints the traceback with the rank and calls
`MPI.COMM_WORLD.Abort(1)`, so no rank waits at a barrier until the job ends. Wall time and peak memory (ru_maxrss)
per stage and rank go to `<data_root>/w3ra/cases/<case>/log/resources.csv`.

## 1. Stages

| stage (`RDA.`) | ranks | what it does | main outputs |
|---|---|---|---|
| `prepare_Forcing()` | 1 | crops global ERA5 forcing, climatology, parameters and land mask to the box (Metview); only months not yet in the domain | `domains/<domain>/input/{forcing,clim,par,mask}` |
| `prepare_GRACE(...)` / `prepare_GRACE_Mascon(...)` | 1 | basin and grid TWS of GRACE (SH grids or CSR mascons) and the basin covariance from the DDK3 samples; only months not yet there | `units/<unit_set>/grace/<product>/` |
| `single_run()` | 1 | spin-up (`init.mode = cold`) over the cold period; writes the initial state for the warm start; reused if done | `domains/<domain>/spinup/init/<domain>_state.<YYYYMMDD>.h5` |
| `OL_run(skip_perturbation)` | ens + 1 | perturbs forcing and parameters (unless skipped or the ensemble exists), runs the open-loop ensemble (unless it exists), basin series, collects; prepares DA: OL mean for the datum and a state sample for the NaN mask | `domains/<domain>/ensembles/<ensemble>/`, `domains/<domain>/OL/<ensemble>/ens_kk/`, `cases/<case>/res/…OL…`, `cases/<case>/grace/ol_mean.hdf5` |
| `DA_run(skip_obs_perturbation)` | ens + 1 | checks the start date (below), perturbs GRACE (unless skipped), builds the design matrix (unless it exists), runs the EnKF from the OL state of the day before, basin series, collects; empties `tmp/` | `cases/<case>/DA/ens_kk/` (issue 5 fixed), `cases/<case>/grace/obs/`, `cases/<case>/res/…DA…` |

**DA start date.** Before anything runs, `DA_run` checks that `resume_begin_time` is the first day of an
observation window and that every member's OL state of the day before exists; otherwise it stops and names the first
valid start date (e.g. with an open loop from 2002-04-01 and CSR mascons the first possible DA start is 2002-05-01).

Member 0 is the unperturbed run; members 1…ens are perturbed. Run the MPI stages with
`mpiexec -n <ens+1> python <driver>.py` from `demo/` (all relative paths start at `../`).

Note on the flags: `demo_3.py` calls `OL_run(skip_perturbation=prepare)`, so `prepare=True` **skips** the
perturbation (it assumes the perturbed inputs exist).

## 2. `setting.json` — the W3RA run

| block | keys | notes |
|---|---|---|
| `bounds` | `lat [north, south]`, `lon [west, east]`, `prefix` | written by the driver |
| `init` | `mode` (`cold`, `warm`, `resume`), `date`, `spinup [from, to]`, `dir`, `spec` | `cold`: spin-up from default states; `warm`/`resume`: start from `<dir>/<spec>_state.<date>.h5` |
| `run` | `fromdate`, `todate`, `res` (0.1), `exttype` (`ERA5`), `save_states_every_day`, `save_output_every_day` | DA needs `save_states_every_day: true` |
| `input` | `meteo`, `clim`, `pars`, `mask_fn` | base folder; the code appends `<prefix>/forcing`, `/clim`, `/par`, `/mask` (member 0 and single runs) or `ens_forcing`, `ens_par` (members) |
| `output` | `dir`, `var{name: bool}` | daily outputs: `<dir>/output_<prefix>[_ensemble_k]/output.YYYYMMDD.h5`; states `<dir>/state_<prefix>[_ensemble_k]/state.YYYYMMDD.h5` |

## 3. `pre_process.json` — cropping

`date_begin`, `date_end` (`YYYY-MM`, written by the driver), `out_dir` (cropped inputs, the same folder is used as
`input.*` of `setting.json`), `mask_dir`, `forcing_dir` (ERA5 `Raw_W3`, monthly GRIB), `clim_dir`, `par_dir`
(global W3RA files `pyW3RA/{mask,clim,par}`).

## 4. `perturbation.json` — the ensemble

| key | meaning |
|---|---|
| `ensemble` | number of perturbed members (written by the driver) |
| `dir.in`, `dir.out` | cropped inputs of the case; perturbed files go to `ens_forcing`/`ens_par` (written by the driver) |
| `correlation.forcing` | `isSpatialCorrelated`, `isTemporalCorrelated`, `customize` |
| `correlation.par` | `isSpatialCorrelated`, `customize` |
| `forcing.<var>` | for `2t`, `ssrd`, `tp`: `is_perturbed`, `perturb_method` (`additive`/`multiplicative`), `error_distribution` (`gaussian`/`triangle`), `coefficients` (std or half-width) |
| `par.<name>` | same four keys for each of the 44 W3RA parameters |

Default of the examples: `2t` additive Gaussian 2 K; `ssrd` additive triangular 4.32 MJ m⁻² d⁻¹; `tp`
multiplicative Gaussian 30 %; parameters `ER_frac_ref`, `FsoilEmax`, `InitLoss`, `PrefR`, `S0FC`, `S_sls`, `SdFC`,
`SsFC`, `beta` multiplicative triangular 30 %, all others unperturbed. The random draws are not seeded **[issue 2]**.

## 5. `DA_setting.json` — the assimilation

Four blocks, all required. Inside `basic`, `model` and `method` a missing key takes the default of
`configure_DA.py` (`process()` merges each block over its defaults). `obs.GRACE` is replaced as a whole, so all of
its keys must be given.

### 5.1 `basic`
`fromdate`, `todate`, `basin`, `basin_shp`, `ensemble` (written by the driver); `NaNmaskStatesDir` (a state file of
the open loop, `<case>_state.h5`, is copied here to mask cells without data: the last OL day, chosen by date; the
legacy layout keeps the old arbitrary `listdir()[-1]`, issue 11); `layer` — the state
vector, one switch per W3RA store: `S0`, `Ss`, `Sd` (top, shallow, deep soil), `Sr` (surface/river), `Sg`
(groundwater), `Mleaf` (canopy; ×4 in TWS), `FreeWater`, `DrySnow` (snow). Two hydrological response units per cell
are combined with `Fhru`.

### 5.2 `obs`
`name` (`GRACE`), `dir` (perturbed observations, `ens_<k>` per member) and the block `GRACE`:

| key | meaning |
|---|---|
| `kind` | `Mascon_monthly` (CSR RL06.3 mascons), `SH_monthly` (SaGEA SH grids), `ESA_SING`, `ESA_SING_ESM3` (5-daily simulated, future missions) |
| `EWH_grid_dir` | input grids (mascon NetCDF or SH grids) |
| `cov_dir` | SaGEA DDK3 Monte Carlo samples, one file per month (`YYYY-MM_0.npy`) |
| `preprocess_res` | basin TWS, grid TWS and covariance of the basin units |
| `OL_mean` | open-loop mean per unit (`<case>_<basin>.hdf5`), added back to the perturbed anomalies **[issue 3]** |
| `aux_for_time_epochs` | where the time epochs of each solution are read |

Observation perturbation (`GRACE_perturbation.py`): for each month, `ensemble` draws from N(obs, R) with R the basin
covariance; the temporal mean of the unperturbed series is removed and the open-loop mean is added. Not seeded,
not centred **[issue 2]**.

### 5.3 `model`
`name`: `w3ra_v0` (only value used).

### 5.4 `method`
`fusion_method` selects the filter class (`src_DA/filter_registry.py`, issue 4 fixed): `EnKF` (plain perturbed-observation
EnKF, no shrinkage), `EnKF_localization_v1` (default, section 6), `EnKF_localization_v2`, `EnKF_domain_localization`,
`EnKF_adaptive`, `EnSQRA`, `EnSQRA_V2`. The old names `EnKF_v0`, `EnKF_v1`, `EnKF_v2` never chose a filter; they run
`EnKF_localization_v1` as before, with a warning. An unknown name stops the run with the list of valid names.
`inflation` (default 1.1) multiplies the forecast anomalies; with exactly 1.0 the inflation step is skipped, so
`EnKF` + 1.0 is the filter of the published ESA-SING runs of 2025 (`settings/JoG_ESA_SING/README.md`). The run log
(`DA_log_rank<k>.txt` / stdout of rank 0) names the filter and the inflation.

## 6. The default filter (`src_DA/EnKF.py`, class `EnKF_localization_v1`)

- Window: one GRACE solution (monthly, or 5-daily for `ESA_SING`). The model runs through the window, the
  window-mean state is compared with the observation, and the analysis increment of the window mean is added to the
  stored daily states of every day of the window (no rerun). After the update, every negative storage is set to 0.0001 **[issue 8]**.
- The ensemble anomalies of the forecast are inflated by `method.inflation` (default 1.1) before the update.
- Both covariances in observation space are shrunk towards their diagonal, C' = (1 − α) C + α diag(C):
  α = 0.10 for the forecast covariance H P Hᵀ and α = 0.50 for the observation error covariance R, i.e. the
  correlations of R between units are halved **[issue 8]**. K = A (HA)ᵀ (H P Hᵀ + R)⁻¹ / (N − 1), solved with
  `lstsq`; each member is updated with its own perturbed observation. The applied increment is the analysis minus
  the *uninflated* forecast, so the inflation itself is kept in the states.
- Member 0 (the unperturbed run) takes no part in the update and is not changed: in a DA run it stays an open loop.
- The time bookkeeping requires `fromdate` to coincide with the start of a GRACE window (now checked before the run,
  see section 1); days after the last window are not assimilated **[issue 6]**.
- The time of every analysis (update and whole window) is logged to `<data_root>/w3ra/cases/<case>/log/analysis_times.csv`.
- No per-window log of forecast HX, innovation or increment is written **[issue 7]**.

## 7. Data layout and outputs

Everything a run writes is below `<data_root>/w3ra/` (`src_FlowControl/run_context.py`, class `Layout`). Folders that
several cases can share are named by driver attributes, recorded in `cases/<case>/case.json`:

| attribute | default | names |
|---|---|---|
| `RDA.domain` | case name | model box: cropped inputs, spin-up, ensembles, open loops |
| `RDA.ensemble` | `n<ens>` | perturbed forcing/parameters and the open loop run with them |
| `RDA.unit_set` | basin name | observation units (shapefile): masks, design matrices, prepared GRACE |
| `RDA.grace_product` | from `obs.GRACE.kind` (`CSR_mascon`, `SH_monthly`, `ESA_SING`, `ESA_SING_ESM3`) | prepared GRACE product of the units |
| `RDA.obs_input` | None | with `DA_run(skip_obs_perturbation=True)`: an existing perturbed-obs file, copied into the case (e.g. `${DATA_ROOT}/GRACE/ESA_SING/Europe_obs_GRACE.hdf5`) |

```
<data_root>/
  ERA5/  GRACE/{SaGEA,aux,masks,ESA_SING,...}  basin/shp/<name>/...      model-independent input (never written by runs)
  w3ra/global/        pyW3RA/ matlab/ perturbation_sample/ basin_selection/
  w3ra/domains/<domain>/
      input/{forcing,clim,par,mask}            cropped inputs; months added when missing
      spinup/{state,output}/  spinup/init/<domain>_state.<YYYYMMDD>.h5   spinup/products/SR/
      ensembles/<ensemble>/{ens_forcing,ens_par}
      OL/<ensemble>/ens_00..ens_NN/{state,output}/   OL/<ensemble>/nan_sample.h5   OL/<ensemble>/products/OL_mean/
  w3ra/units/<unit_set>/
      mask/  DM/<domain>__<ensemble>/DM.hdf5  grace/<product>/   (basin TWS, grid TWS, covariance)
  w3ra/cases/<case>/
      case.json  settings/  log/  tmp/ (emptied after a successful DA)
      grace/ol_mean.hdf5  grace/obs/ (perturbed obs)   DA/ens_00..ens_NN/{state,output}/
      res/  figure/  products/{DA_mean,DA_std,mask}/
  w3ra/aggregates/<global_run>/   GDA_{DA,OL,GR}_map.h5, SR/   (global tile-wise DA, see below)
```

**Sharing and `made_with.json`.** Every shared folder (domain, ensemble, open loop, unit set, GRACE product, design
matrix) holds a `made_with.json` with the inputs that define it: domain = box, resolution, forcing source, cropped
months; ensemble = the full `perturbation.json` content, member count, seed note; open loop = ensemble, members, period,
initial state; unit set = shapefile path and md5, resolution; GRACE product = kind, input folders, months, diagonal or
full covariance; design matrix = units shapefile md5, domain, ensemble, state layers. A stage reuses a folder only if
its definition matches; a mismatch stops with an error naming the differing keys (e.g. `perturbation.forcing.tp.
coefficients`) and asking for another name. Nothing is silently reused or overwritten. Missing months (forcing, GRACE)
are added and `made_with.json` is updated. Every reuse is noted in `cases/<case>/log/reuse.log`. Two cases on the same
domain, ensemble and units therefore run only their own DA; a second case with another perturbation needs another
`RDA.ensemble`.

**Legacy layout** (`RDA.legacy_layout = True`): the old places, unchanged: `w3ra/crop_input/<case>`,
`w3ra/output/{state,output}_<case>[_ensemble_<k>]`, `w3ra/initial_states`, `w3ra/states_sample`, `GRACE/{output,obs,OLmean}`,
`w3ra/res|figure/<case>`, `../temp/DM.hdf5`, `../log/OL/`; settings run copy in `w3ra/runs/<case>/settings`.

**Post-processing.** `RDA.activate()` registers the data root and the layout of the case set in the driver (without
running a stage) and returns its run copy of the settings. Then `src_postprocessing/DataManager.py` with `out_dir=None`
writes to the layout: `dataManager_ensemble_statistic` -> `cases/<case>/products/DA_mean`, `DA_std`;
`reduce_datasize(save_mask=True)` -> `products/mask`; `dataManager_ensemble_member(ens, stage='OL')` ->
`domains/<domain>/OL/<ensemble>/products/OL_mean`; `dataManager_single_run` -> `domains/<domain>/spinup/products/SR`.
With an explicit `out_dir` the old `<out_dir>/<case>` is used. `cleanup()` in the layout deletes only the case's
`DA/` members and `tmp/`, never shared domain data. The visualization methods of `RDA` read the run copy, not
`RDA.setting_dir`. `src_DA/Analysis.py` save/load functions default to `cases/<case>/res` (legacy `../temp`).

**Global tile-wise DA** (`src_FlowControl/global_DA.py`, class `GDA`): `GDA.legacy_layout = True` (default) keeps the
old places. With `False`, every tile is a case `<GDA.global_run>_Tile<n>` with its own domain and unit set, settings run
copy in `cases/<case>/settings`, and `src_auxiliary/GlobalAggregation.py` writes the global maps and the SR mosaic to
`aggregates/<global_run>/`. The tile shapefile is `basin/shp/global_shp_new/Tile<n>_subbasins.shp`. This mode is
not yet tested with a tile run.

| what | where (layout) |
|---|---|
| basin series per member | `basin_ts_<OL|DA>.h5` in each member's `output/`, collected by `post_processing` into `cases/<case>/res/` |
| logs | `cases/<case>/log/`: `OL_log_rank<k>.txt`, `OL_signal_log_rank<k>.txt`, `DA_log_rank<k>.txt`, `DA_signal_log_rank<k>.txt`, `resources.csv`, `analysis_times.csv`, `reuse.log`, `errors.log` (legacy: `../log/OL/`) |
| error log of all runs | `<repo>/log/errors.log` and `<repo>/log/errors/` (see Logs below) |

### Logs

Two places, with different jobs:

- **Code log `py-w3ra/log/`** (git-ignored, found from the package path, not from the working directory) collects the
  **errors of all runs** started from this checkout. When a stage fails on any rank, the guard writes one detail file
  `log/errors/<YYYYmmdd-HHMMSS>_<case>_<stage>_rank<k>.txt` (time, host and UCloud job id, case, stage, rank/size, data
  root, run-copy settings folder, git commit with a dirty flag, python and numpy versions, full traceback) and appends
  one tab-separated line (time, case, stage, rank, exception, detail file) to `log/errors.log`; then it calls
  `MPI.COMM_WORLD.Abort(1)`. One file per error, so several ranks never write the same file. On UCloud this is
  `/work/W3RA_Fan/py-w3ra/log/`; the Syncthing link does not bring it back to the desktop, so read it on UCloud.
- **Case log `<data_root>/w3ra/cases/<case>/log/`** holds **everything of one case**: rank logs of OL and DA,
  `resources.csv`, `analysis_times.csv`, `reuse.log` (every shared folder reused, every month added), and the same error
lines (`errors.log`), so a case folder is complete on its own.

The old `log/OL/` is used only with `legacy_layout = True`. Test only: the environment variable
`PYGLDA_V1_FAIL_TEST=<stage>:<rank>` (stage = `forcing`, `prepare_GRACE_Mascon`, `prepare_GRACE`, `spinup`, `ol`, `da`)
raises an exception in that stage on that rank, to test the abort and the error logs; never set it in production.
