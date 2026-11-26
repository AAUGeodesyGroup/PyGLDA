# PyGLDA v1 — settings reference

Every key of the four JSON files that the code reads, at commit `8c775b5`. "driver" = written by
`generate_settings()` at the start of each stage (see `configuration.md` §0); a value in the file is overwritten.
Defaults are the class defaults in `src_GHM/config_settings.py` and `src_DA/configure_DA.py`; they apply to a key
missing inside a block that is present (except `obs.GRACE`, which must be complete). Paths are absolute unless
noted; relative paths start at `demo/`.

## setting.json (`config_settings`)

| key | meaning | values | default |
|---|---|---|---|
| `bounds.lat` | box [north, south] | degrees | driver (shapefile box + 0.2°) |
| `bounds.lon` | box [west, east] | degrees | driver |
| `bounds.prefix` | name of the case folders | text | driver (= case) |
| `init.mode` | start type | `cold`, `warm`, `resume` | driver (stage) |
| `init.date` | date of the initial state file | `YYYY-MM-DD` | driver (day before `run.fromdate`) |
| `init.spinup` | spin-up period [from, to] | dates | driver (cold stage) |
| `init.dir` | folder of initial states `<spec>_state.<YYYYMMDD>.h5` | path | required |
| `init.spec` | name in the initial-state file | text | driver (= case) |
| `run.fromdate`, `run.todate` | model period | `YYYY-MM-DD` | driver (warm/resume) |
| `run.res` | grid resolution | `0.1` (used) | 0.05 |
| `run.exttype` | forcing | `ERA5` | `ERA5` |
| `run.save_states_every_day` | write daily states (needed by DA) | bool | false |
| `run.save_output_every_day` | write daily outputs `output.var` | bool | false |
| `input.meteo`, `input.clim`, `input.pars`, `input.mask_fn` | base folder of the cropped inputs; `<prefix>/forcing|clim|par|mask` (or `ens_forcing`, `ens_par` for members) is appended | path | driver (= `pre_process.out_dir`) |
| `output.dir` | folder of `state_<prefix>[_ensemble_k]` and `output_<prefix>[_ensemble_k]` | path | required |
| `output.var.<name>` | write this output variable | bool for `Stot`, `ETtot`, `Qtot`, `Qnet`, `E0`, `Ei`, `Et`, `Es`, `Er`, `RSn`, `RLn`, `LE`, `H`, `albedo`, `LAI`, `Ssnow`, `z_g`, `S0`, `Sroot`, `Ssoil`, `Sg`, `Ts`, `fsnow`, `Sr` | all false |

## pre_process.json

| key | meaning | default |
|---|---|---|
| `date_begin`, `date_end` | months to crop (`YYYY-MM`) | driver |
| `out_dir` | cropped inputs (`<out_dir>/<case>/{forcing,clim,par,mask,ens_forcing,ens_par}`) | required |
| `mask_dir`, `clim_dir`, `par_dir` | global W3RA files (`pyW3RA/mask`, `clim_01..12.h5`, `par.h5`) | required |
| `forcing_dir` | ERA5 daily forcing, monthly GRIB `W3_ERA5_daily_YYYYMM.grib` | required |

## perturbation.json (`src_DA/Perturbation.py`)

| key | meaning | values |
|---|---|---|
| `ensemble` | perturbed members | driver |
| `dir.in`, `dir.out` | cropped inputs of the case | driver (`<out_dir>/<case>`) |
| `correlation.forcing.isSpatialCorrelated` / `isTemporalCorrelated` / `customize` | error correlation of the forcing perturbation | bool |
| `correlation.par.isSpatialCorrelated` / `customize` | error correlation of the parameter perturbation | bool |
| `forcing.<2t|ssrd|tp>.is_perturbed` | perturb this forcing | bool |
| `forcing.<var>.perturb_method` | | `additive`, `multiplicative` |
| `forcing.<var>.error_distribution` | | `gaussian` (coefficient = std), `triangle` (coefficient = half-width) |
| `forcing.<var>.coefficients` | list with one value | number |
| `par.<name>.*` | same four keys for each of the 44 W3RA parameters | as above |
| (no key) | random seed | not available: draws are unseeded **[issue 2]** |

## DA_setting.json (`configure_DA`)

### basic
| key | meaning | values | default |
|---|---|---|---|
| `fromdate`, `todate` | assimilation period; `fromdate` must be the first day of a GRACE window **[issue 6]** | `YYYY-MM-DD` | driver |
| `basin` | unit-set name (also used to find the shapefile) | text | driver |
| `basin_shp` | shapefile of the units | path (relative to `demo/`) | driver (search under `data/basin/shp/`) |
| `ensemble` | perturbed members (`ensemble + 1` MPI ranks) | integer | driver |
| `NaNmaskStatesDir` | folder that receives `<case>_state.h5` to mask cells without data | path | required |
| `layer.<store>` | store in the state vector | bool for `S0`, `Ss`, `Sd`, `Sr`, `Sg`, `Mleaf`, `FreeWater`, `DrySnow` | all true |

### obs
| key | meaning | values | default |
|---|---|---|---|
| `name` | observation type | `GRACE` | — |
| `dir` | perturbed observations (`ens_0 … ens_N`, `cov`, `time_epoch`, …) | path | required |
| `GRACE.kind` | product and time step | `Mascon_monthly`, `SH_monthly`, `ESA_SING`, `ESA_SING_ESM3` | required |
| `GRACE.EWH_grid_dir` | input grids | path | required |
| `GRACE.cov_dir` | DDK3 Monte Carlo samples | path | required |
| `GRACE.preprocess_res` | prepared basin TWS / covariance | path | required |
| `GRACE.OL_mean` | open-loop mean per unit `<case>_<basin>.hdf5` | path | required |
| `GRACE.aux_for_time_epochs` | time epochs of the solutions | path | required |

### model
| key | values | default |
|---|---|---|
| `name` | `w3ra_v0` | `w3ra_v0` |

### method
| key | values | default |
|---|---|---|
| `fusion_method` | `EnKF_v0`, `EnKF_v1`, `EnKF_v2` (parsed, **not used** **[issue 4]**) | `EnKF_v0` |

## Constants in the code (not settings yet **[issue 8]**)

| constant | value | where |
|---|---|---|
| inflation of the forecast anomalies | 1.1 | `src_DA/EnKF.py:35` |
| shrinkage of H P Hᵀ / of R (`EnKF_localization_v1`) | 0.10 / 0.50 | `src_DA/EnKF.py:377, 381` |
| floor for negative storages after the update | 0.0001 | `src_DA/EnKF.py:230` |
| box margin around the shapefile | 0.2° | `Regional_DA.py:set_box` |
| model grid | 0.1° | `DA_GRACE.py` (`basin_shp_process(res=0.1)`) |
