# PyGLDA v1 — configuration guide

Describes the code at commit `8c775b5`. Items marked **[issue n]** behave differently from what one would expect;
see `known_issues.md`.

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
the JSON files in `settings/demo_3` were last rewritten for basin `Tile81` **[issue 9]**. The published Danube runs
(30 members) are the UCloud cases `RDA_DRB_BasinScale` and `RDA_GDRB_BasinScale` (results in `/work/data_for_w3/w3ra/res`).

## 0. How the driver and the JSON files interact

The driver sets class attributes of `RDA` and calls the stages:

```python
RDA.setting_dir = '../settings/demo_3'
RDA.ens = 30                       # perturbed members; MPI needs ens + 1 ranks
RDA.case = 'case_study_demo3'      # names all folders of this experiment
RDA.basin = 'DRB'
RDA.shp_path = '../data/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp'
RDA.cold_begin_time, RDA.cold_end_time = '2005-01-01', '2005-12-31'      # spin-up (single run)
RDA.warm_begin_time, RDA.warm_end_time = '2005-01-02', '2005-12-31'      # open loop
RDA.resume_begin_time, RDA.resume_end_time = '2005-04-01', '2005-12-31'  # assimilation
```

At the start of every stage, `generate_settings(mode)` **rewrites the JSON files in the settings folder in place**
**[issue 9]** so that they agree with the driver:

| file | keys written by the driver |
|---|---|
| `pre_process.json` | `date_begin`, `date_end` (from the stage period) |
| `setting.json` | `bounds.lat/lon` (shapefile box + 0.2°), `bounds.prefix` = case, `input.*` = `pre_process.out_dir`, `init.spec` = case, `init.mode`; `init.spinup` (cold) or `run.fromdate/todate` and `init.date` = day before (warm/resume) |
| `perturbation.json` | `dir.in`, `dir.out` = `<pre_process.out_dir>/<case>`, `ensemble` |
| `DA_setting.json` | `basic.ensemble`, `fromdate`, `todate`, `basin`; `basic.basin_shp` = the single file under `data/basin/shp/` named `<basin>_…subbasins….shp` (the driver's `shp_path` is not used here) |

Everything else in the JSON files is taken as written. `RDA.config_external_data()` (optional) rewrites every path
that contains `/w3ra/`, `/GRACE/` or `/ERA5/` so that it starts with `RDA.external_data_path`.

The result folders `res_output` and `figure_output` are module variables of `Regional_DA.py`
(`/work/data_for_w3/w3ra/res` and `…/figure`). Setting variables of the same name in a driver has no effect
**[issue 13]**; set `Regional_DA.res_output` instead.

## 1. Stages

| stage (`RDA.`) | ranks | what it does | main outputs |
|---|---|---|---|
| `prepare_Forcing()` | 1 | crops global ERA5 forcing, climatology, parameters and land mask to the box (Metview) | `<pre_process.out_dir>/<case>/{forcing,clim,par,mask}` |
| `prepare_GRACE(...)` / `prepare_GRACE_Mascon(...)` | 1 | basin and grid TWS of GRACE (SH grids or CSR mascons) and the basin covariance from the DDK3 samples | `obs.GRACE.preprocess_res` |
| `single_run()` | 1 | spin-up (`init.mode = cold`) over the cold period; writes the initial state for the warm start | `<init.dir>/<case>_state.<YYYYMMDD>.h5` |
| `OL_run(skip_perturbation)` | ens + 1 | perturbs forcing and parameters (unless skipped), runs the open-loop ensemble, basin series, collects; prepares DA: OL mean for the datum and a state sample for the NaN mask | `state_<case>_ensemble_<k>/`, `output_<case>_ensemble_<k>/`, `<res_output>/<case>/…OL…`, `OL_mean/<case>_<basin>.hdf5` |
| `DA_run(skip_obs_perturbation)` | ens + 1 | perturbs GRACE (unless skipped), builds the design matrix, runs the EnKF, basin series, collects | daily states (same folders as the OL **[issue 5]**), `<res_output>/<case>/…DA…` |

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
the open loop, `<case>_state.h5`, is copied here to mask cells without data **[issue 11]**); `layer` — the state
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
`fusion_method`: `EnKF_v0`, `EnKF_v1`, `EnKF_v2`. **Read but not used** — `run_DA` always uses
`EnKF_localization_v1` **[issue 4]**.

## 6. The filter as run (`src_DA/EnKF.py`, class `EnKF_localization_v1`)

- Window: one GRACE solution (monthly, or 5-daily for `ESA_SING`). The model runs through the window, the
  window-mean state is compared with the observation, and the analysis increment of the window mean is added to the
  stored daily states of every day of the window (no rerun). After the update, every negative storage is set to 0.0001 **[issue 8]**.
- The ensemble anomalies of the forecast are inflated by 1.1 before the update **[issue 8]**.
- Both covariances in observation space are shrunk towards their diagonal, C' = (1 − α) C + α diag(C):
  α = 0.10 for the forecast covariance H P Hᵀ and α = 0.50 for the observation error covariance R, i.e. the
  correlations of R between units are halved **[issue 8]**. K = A (HA)ᵀ (H P Hᵀ + R)⁻¹ / (N − 1), solved with
  `lstsq`; each member is updated with its own perturbed observation. The applied increment is the analysis minus
  the *uninflated* forecast, so the inflation itself is kept in the states.
- Member 0 (the unperturbed run) takes no part in the update and is not changed: in a DA run it stays an open loop.
- The time bookkeeping requires `fromdate` to coincide with the start of a GRACE window; days after the last window
  are not assimilated **[issue 6]**.
- No per-window log of forecast HX, innovation or increment is written **[issue 7]**.

## 7. Outputs

| what | where |
|---|---|
| daily states / outputs per member | `<output.dir>/state_<case>_ensemble_<k>/`, `output_<case>_ensemble_<k>/` |
| basin series per member | `basin_ts_<OL|DA>.h5` (states of each unit, cos-latitude weighted), collected by `post_processing` into `<res_output>/<case>/` |
| design matrix | `../temp/` (shared by all cases **[issue 12]**) |
| logs | `../log/OL/log_<rank>.txt`, `signal_log_<rank>.txt` (OL and DA use the same names) |
