# PyGLDA v1 — known issues

Verified in the code at commit `8c775b5` (26 Nov 2026; #10 and #14 corrected after the installation test). "Fix" is the proposal under discussion with Fan; nothing is
changed yet. Each fix will be behind a setting that restores today's behaviour, and every fix is tested against a
frozen baseline. Status: **open** / **agreed** / **fixed in <commit>**.

| # | issue | where | effect | proposed fix | status |
|---|---|---|---|---|---|
| 1 | `np.roll(data, 360)` without `axis` on the 2-D mascon grid | `src_OBS/prepare_GRACE_mascon.py` ~94, 187 | the array is rolled as one flat vector, so the western half of every row ends up one row (0.5°) too far south; Europe/Danube east of 0° unaffected | `np.roll(data, 360, axis=1)` | open |
| 2 | observation perturbation and forcing/parameter perturbation are not seeded; observation ensemble not centred | `src_OBS/GRACE_perturbation.py`; `src_DA/Perturbation.py` | runs are not reproducible; ensemble mean of the observations ≠ observation | seeded generator (`seed` setting), centring option (default on) | open |
| 3 | open-loop datum = mean over all OL days | `src_FlowControl/DA_GRACE.py:129-169` (`gather_OLmean`) | constant offset against anomalies defined over the GRACE months | mean over the GRACE windows (v2 definition) | open |
| 4 | `method.fusion_method` parsed but ignored | `DA_GRACE.py:441` | filter is always `EnKF_localization_v1` | dispatch by name; unknown name = error | open |
| 5 | DA writes its daily states into the open-loop folders `state_<case>_ensemble_<k>` | `src_GHM/config_settings.py:105-106` | open-loop states are overwritten by the DA run | separate `OL` and `DA` folders | open |
| 6 | fragile window bookkeeping | `src_DA/EnKF.py:246-341` (`helper_resolve_time`) | `fromdate` must start a window; runs past the last window rely on a `-1` sentinel | assimilate only complete windows with data; log skipped ones | open |
| 7 | no per-window log of forecast HX, innovation, increment per store | `src_DA/EnKF.py` | χ², gain, water added/removed cannot be computed after a run | `windows.csv`, `filter_summary.json` (v2 format) | open |
| 8 | constants in the code: inflation 1.1, shrinkage 0.10/0.50, floor 0.0001 | `src_DA/EnKF.py:35, 230, 377, 381` | undocumented, not configurable | settings with today's values as defaults | open |
| 9 | the JSON files in `settings/` are rewritten in place at every stage | `generate_settings` in `DA_GRACE.py`, `OpenLoop.py`, `SingleModel.py`; `RDA.config_external_data` | tracked files change with every run; no record of what a run used | run copy per case (v2 style) | open |
| 10 | two legacy modules are still imported (`GRACE_perturbation_old`, `EnKF_fixed_time_old`); only the base class `DA_GRACE.run_DA` uses them, the driver uses `DA_GRACE_flexibile` | `DA_GRACE.py:7, 254` | dead code paths, no failure (imports pass) | move the old modules and filter classes to `src_DA/legacy/` | open |
| 11 | the state sample for the NaN mask is `os.listdir(...)[-1]` | `DA_GRACE.py:get_states_sample_for_mask` | arbitrary day (listdir order is not sorted) | take the last date explicitly | open |
| 12 | design matrix saved to `../temp` without case name | `DA_GRACE.py:prepare_design_matrix`, `run_DA` | two cases run from the same checkout overwrite each other | `temp/<case>/` or the run folder | open |
| 13 | `res_output`, `figure_output` are module variables of `Regional_DA.py`; values set in a driver are ignored | `Regional_DA.py:10-12, set_box` | demos write to `/work/data_for_w3/w3ra/…` even on the desktop | `RDA.res_output`, `RDA.figure_output` attributes | open |
| 14 | the installation test `demo/demo_test_installation.py` is a full OL+DA demo with a dead data path (`…/SummerSchool/External Data`); the spec-file recipe installs h5py with pip (other HDF5 than netCDF4, not reproducible) | `README.md`, `installation/linux-spec-file.txt` | installation cannot be tested or reproduced as documented | `installation/` (environment.yml, lock, install.sh, checks) + `doc/installation.md` | fixed, not committed |
| 15 | `OL_run` / `DA_run` logs of OL and DA share names `../log/OL/*_<rank>.txt` | `Regional_DA.py`, `DA_GRACE.py` | DA log overwrites OL log | per-stage log folder | open |

## Behaviour that is intended but easy to miss
- Member 0 is unperturbed and is not updated in a DA run (it stays an open loop).
- The applied increment includes the 1.1 inflation (analysis minus the uninflated forecast).
- The window-mean increment is added equally to every stored day of the window; the model is not rerun.
- R is shrunk by 0.5 towards its diagonal before the update.
