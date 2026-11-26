# JoG ESA-SING settings (paper JOGE-D-26-00146, revision)

Best configuration of the published runs: alpha 1e18, 1-degree units, basins Brahmaputra and Danube, missions GRACE-C-like
(GRACEC), NGGM, MAGIC; one folder per mission and basin. Built from the published settings (settings/<MISSION>/1e18, Apr/Jul
2025 runs) with `${DATA_ROOT}` paths; only these differ from them:

- `method.fusion_method = "EnKF"`, `method.inflation = 1.0`: the plain perturbed-observation EnKF without inflation or
  covariance shrinkage, i.e. the filter of the published runs (src_DA/data_assimilation_flexible_time.DataAssimilation at
  commits abf1302/16e1fe2/3674684; see V1 review 19 Proof A). Today's default would be EnKF_localization_v1 with 1.1.
- `obs.GRACE.kind = "ESA_SING"`, `aux_for_time_epochs = ${DATA_ROOT}/GRACE/aux/ESA_5daily` (5-day windows), and
  `EWH_grid_dir`/`cov_dir` = the ESM2.0 inputs `${DATA_ROOT}/GRACE/ESA_SING/ESM2.0/<basin>/gravity_signal_cov` (V1 review 18).
  `preprocess_res` = the prepared products of the published runs `${DATA_ROOT}/GRACE/output/<MISSION>/1e18`.
- `run.forcing_day = "same"`: model day d uses ERA5 day d, as the code of 2025; since commit 8c775b5 (19 Mar 2026) the
  default is "next" (day d+1, ERA5 accumulation convention), which changes the OL by tens of mm after one month
  (V1 review 19, Proof B).
- Everything else as published: 30 members, perturbation of 2t/ssrd/tp and the 9 parameters (perturbation.json), state
  layers, spin-up 2000-01-01..2002-08-31, OL 2002-09-01..2006-12-31, DA 2003-09-01..2006-12-31 (driver demo/demo_ESA_SING.py).

Shared ensemble: all published cases of a basin used ONE perturbed forcing/parameter ensemble, now in
`${DATA_ROOT}/w3ra/crop_input/MAGIC_1e19_1degree_<basin>/{ens_forcing,ens_par}` (2002-09..2006-12, 31 groups ens_0..ens_30;
demo_ESA_SING.demo_batch_run renamed that folder from case to case). `perturbation.json dir` names it for the record; the
code itself places the ensemble per domain (configuration.md section 7), so a rerun imports those files into
`w3ra/domains/<domain>/ensembles/<ensemble>/` first (as tests/jog/reproduce_ol_member.py does for one month).
Observation perturbations of the published runs were not kept (overwritten, unseeded): a DA rerun is statistically, not
bitwise, equal to the published one. No runs with these settings yet.
