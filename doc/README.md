# PyGLDA v1 (py-w3ra) — documentation

PyGLDA v1 assimilates GRACE/GRACE-FO terrestrial water storage (TWS) into the W3RA model (0.1°, daily, ERA5
forcing) with an ensemble Kalman filter run in parallel with MPI (one rank per member). Reference: Yang et al.
(2025), GMD 18, 6195–6217, doi:10.5194/gmd-18-6195-2025.

These pages describe the code **as it is at commit `8c775b5` (19 Mar 2026)**. They are updated together with every
change of behaviour; each change is listed in `known_issues.md` with the commit that fixed it.

| page | content |
|---|---|
| `installation.md` | pinned conda environment, one-line install, checks, desktop and HPC (UCloud) |
| `configuration.md` | how an experiment is set up: driver, the four JSON files, stages, folders, outputs |
| `settings_reference.md` | every key of the four JSON files with meaning, values and default |
| `known_issues.md` | verified issues of the current code, their effect, the agreed fix and its status |

Repository layout:

| folder | content |
|---|---|
| `src_GHM` | W3RA model (`W3RA_timestep_model.py`, `hotrun.py`), configuration (`config_settings.py`), input cropping (`preprocess.py`) |
| `src_OBS` | GRACE preparation: SH grids (`prepare_GRACE.py`), CSR mascons (`prepare_GRACE_mascon.py`), time epochs (`obs_auxiliary.py`), perturbation of the observations (`GRACE_perturbation.py`) |
| `src_DA` | EnKF variants (`EnKF.py`, `EnSQRA.py`), forcing/parameter perturbation (`Perturbation.py`), basin mask and design matrix (`shp2mask.py`, `ObsDesignMatrix.py`), configuration (`configure_DA.py`), basin series (`Analysis.py`) |
| `src_FlowControl` | stages and drivers: `SingleModel`, `OpenLoop`, `DA_GRACE`, `Regional_DA` (class `RDA`), `global_DA`; data root, layout and error logs (`run_context.py`) |
| `src_postprocessing` | figures, products (`DataManager.py`) |
| `demo/` | example drivers and the summer-school notebooks |
| `settings/` | one folder of JSON files per case |
| `tests/regression/` | regression harness (fixed seeds) and bit-by-bit comparison |
| `installation/` | environment files, install script and checks |
| `doc/` | this documentation |
| `temp/` | scratch (git-ignored): coordination notes, test logs |
| `log/` | error log of all runs from this checkout (git-ignored) |

No data in the repository: shapefiles, masks and all model data are below the data root (`installation.md` §6).

Version 2 (WaterGAP 2.2e) lives in a separate repository; where v1 is being aligned with it, `known_issues.md` says so.
