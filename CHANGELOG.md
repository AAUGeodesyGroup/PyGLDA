# Changelog

## v1.2 (2026-11-27)

* **Seeded perturbations** - new setting `run.seed` (`null` = unseeded, as before). With an integer every random draw (forcing,
  parameter and observation perturbations) comes from its own stream `SeedSequence([seed, purpose, member, key])`, so runs are
  bit-reproducible and independent of the number of members or MPI ranks.
* **Innovation diagnostics** - new setting `output.save_innovations` writes per DA window the unperturbed observations, the forecast
  H X mean/variance, R, the analysis H X and the increments per storage compartment (`cases/<case>/DA/innovations.h5`).
* **Optional perturbation bounds** - `min_factor` (multiplicative factors) and `min_value` (perturbed additive forcing fields) in
  `perturbation.json`; absent keys leave the draws unchanged.
* **JoG revision package** - drivers (`tests/jog/jog_run.py`, `r11_prepare.py`, `r35_run.py`, `snow_check.py`), the ESA-SING settings
  (`settings/JoG_ESA_SING/*`), the evaluation package (`tests/jog/eval/`), the figure scripts (`tests/jog/figures/`) and the data-package
  assembly (`tests/jog/zenodo/`).
* Smaller fixes: the seed is read once and broadcast to all ranks; the perturbation month follows `run.forcing_day`; documentation of
  the new settings (`doc/settings_reference.md`) and status updates (`doc/known_issues.md`).
* **Defaults are bit-identical to the previous release** (regression harness `tests/regression`: default run and seeded run with
  innovations compared file by file).
