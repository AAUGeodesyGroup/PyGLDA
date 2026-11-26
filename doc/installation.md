# PyGLDA v1 — installation guide

The Python environment for PyGLDA v1 (py-w3ra: W3RA + EnKF + MPI), how to check it, and how to launch the code.
All files are in `installation/`:

| file | purpose |
|---|---|
| `install.sh` | creates the env (from the lock by default; by name, or under `--prefix DIR` with a private Miniforge), writes activation hooks (and `DIR/activate.sh`, `DIR/ucloud_init.sh`), runs the checks, logs |
| `environment.yml` | core env `pyglda_v1`: conda-forge only, pinned science stack, MPICH |
| `linux-64.lock.txt` | explicit lock (URL + md5 of every package) of the tested env; the default install source |
| `environment-vis.yml`, `linux-64-vis.lock.txt` | optional visualisation env `pyglda_vis` (pygmt) |
| `check_install.py`, `check_install.sh` | serial checks and the MPI check (`mpiexec -n 4`) |
| `linux-spec-file.txt`, `visualization-linux-spec-file.txt`, `windows-spec-file.txt` | the old recipe (Aug 2025), kept for reference |

Logs of `install.sh` go to `installation/logs/` (git-ignored).

---

## 1. Why the environment is pinned

- The science stack is pinned to the versions of the reference env used for the published results
  (Python 3.10.12, numpy 1.26.0, scipy 1.11.3, pandas 2.1.2, OpenBLAS 0.3.24, HDF5 1.14.2, netCDF 4.9.2 / netcdf4
  1.6.4, GDAL 3.7.2, PROJ 9.3.0, GEOS 3.12.0, shapely 2.0.2, eccodes 2.32.1, Metview 5.20.0). A W3RA run in the new
  env is **bit-identical** to the old env (92-day test, 112 files, 839 datasets, max |diff| = 0).
- **MPI is MPICH** (the old recipe used Intel MPI `impi_rt`). MPICH is the conda-forge default and is portable to
  clusters. Always start PyGLDA with the env's own `mpiexec`.
- **h5py comes from conda-forge** and uses the same HDF5 as netCDF4. The old recipe installed h5py with pip, which
  brings its own HDF5 (today h5py 3.16 with HDF5 2.0.0). Never `pip install` h5py, numpy or netCDF4 into this env.
- `setuptools` is pinned (68.2.2) because `cdsapi` imports `pkg_resources`.

## 2. Desktop install

Requirements: Linux x86_64, Miniforge (or any conda; mamba is used if present), about 3 GB of disk.

```bash
cd py-w3ra
bash installation/install.sh                 # env 'pyglda_v1' from the lock, hooks, checks (about 25 s + download)
conda activate pyglda_v1
```

Options: `--env NAME` (other name), `--solve` (solve `environment.yml` instead of using the lock; about 45 s),
`--force` (recreate an existing env), `--vis` (also `pyglda_vis`; the pygmt solve takes about 15 min),
`--no-check`, `--prefix DIR` (HPC, section 3). Re-running without `--force` keeps the env and only rewrites the
hooks and repeats the checks.

The activation hook (`$CONDA_PREFIX/etc/conda/activate.d/zz_pyglda_v1.sh`) sets `PROJ_LIB` and `PROJ_DATA`; nothing
has to be exported by hand. **Always activate the env**; calling `<env>/bin/python` directly leaves `metview` off
the PATH (see section 8).

Updating the lock after a deliberate change of `environment.yml`:

```bash
bash installation/install.sh --solve --force --env pyglda_v1
conda list -n pyglda_v1 --explicit --md5 > installation/linux-64.lock.txt
```

## 3. HPC install (env under a persistent prefix; example: UCloud)

**Status: desktop- and UCloud-tested 26 Nov 2026.** Desktop: clean PATH, private Miniforge bootstrap, 24 s.
UCloud: job with 2 vCPU / 6 GB, `install.sh --prefix /work/data_for_w3/envs` from scratch (Miniforge bootstrap,
sha256 verified, env from the lock in 118 s) took 2 min 44 s including the checks, all PASS; `ucloud_init.sh` 1 s; the
checks in a new shell 8 s, with the env's `mpiexec` (MPICH 4.3.1).

A UCloud job image (Ubuntu 24.04) has a system python 3.12 and a system `mpiexec`, but no conda and no `module`
command, and only `/work/...` survives a job. So everything goes under a **prefix** in `/work`:

```
/work/data_for_w3/envs/
├── miniforge3/          private Miniforge (bootstrapped when no conda is on PATH; release pinned, sha256 checked)
├── pyglda_v1/           the model env (from the lock)
├── pyglda_vis/          the visualisation env (with --vis)
├── activate.sh          source in every shell
├── ucloud_init.sh       select as the job "Initialization" script
├── install_<date>.log   installer logs
└── last_job_init.log    written by ucloud_init.sh at every job start
```

One-time setup, from any job that mounts `/work/data_for_w3` and the code:

```bash
cd /work/W3RA_Fan/py-w3ra          # the code (see "Code on UCloud" below)
bash installation/install.sh --prefix /work/data_for_w3/envs          # add --vis for pygmt
```

Re-running is safe: an existing env is verified (checks run), not rebuilt; `--force` recreates it. The Miniforge
release can be changed with `PYGLDA_MINIFORGE_VERSION=<tag>` (default `26.9.4-2`).

### Every new UCloud job

1. In the job form mount the two UCloud drives - `/data/data_for_w3` (appears as `/work/data_for_w3`) and
   `/code/W3RA_Fan` (appears as `/work/W3RA_Fan`, holds `py-w3ra`) - enable ssh and tmux, and select
   `/data/data_for_w3/envs/ucloud_init.sh` (= `/work/data_for_w3/envs/ucloud_init.sh`) in the **Initialization** field.
   Tested 26 Nov 2026 on DeiC Interactive HPC `cpu-amd-zen5-8-vcpu` (8 vCPU, 24 GB) and on a 2 vCPU / 6 GB job. If it was not selected, run
   `source /work/data_for_w3/envs/activate.sh` in each shell yourself.
2. Every new shell then has the env first on PATH (`python`, `mpiexec`, `metview` from the env).
   Always use the env's `mpiexec`, not the system one (`which mpiexec` must point into the env).
3. Check `cat /work/data_for_w3/envs/last_job_init.log`: a fresh timestamp, the job's cgroup limits, and both
   `rank 0 of 2` and `rank 1 of 2`.
4. Start long stages inside `tmux` (`tmux new -s <name>`; detach Ctrl+B then D; `tmux attach -t <name>`). The init
   script switches on mouse scrolling and a 50 000-line scrollback (`~/.tmux.conf`). Note: in the current UCloud image
   `~/.tmux.conf` is a link to the "Oh my tmux" main file; the two lines are appended there (job-local, harmless).

### Job size

`nproc` and `free` show the host, not the job. The real limits are in `/sys/fs/cgroup/cpu.max` and
`/sys/fs/cgroup/memory.max`; `check_install.sh` and `ucloud_init.sh` print them. A v1 OL/DA stage needs `ens + 1`
MPI ranks (31 for 30 members). The job used for the inventory (2 vCPU, 6 GB) is only enough for install tests.

### Code on UCloud

- `/work/W3RA_Fan/py-w3ra` is a **Syncthing mirror** of the desktop working copy `/home/user/codes/py-w3ra`
  (checked 26 Nov 2026: same commit `8c775b5`, identical md5, `temp/` included). This is the simplest way: no git
  operation on UCloud, no credentials.
- Git alternative: the reference repository is GitHub (since 26 Nov 2026 the only remote):
  `git -C /work clone https://github.com/AAUGeodesyGroup/PyGLDA.git py-w3ra` (public, no credentials needed to
  clone) and later `git pull`. Pushing uses `git@github.com:AAUGeodesyGroup/PyGLDA.git` with an ssh key.

## 4. What the checks verify

`bash installation/check_install.sh [ENV]` (run by `install.sh`) prints versions and reports PASS / FAIL / WARN:

| check | required | meaning |
|---|---|---|
| versions | yes | numpy, scipy, pandas, h5py, netCDF4, geopandas, shapely, pyproj/PROJ, GDAL, mpi4py + MPI library, eccodes import and report versions |
| HDF5 consistency | yes | h5py and netCDF4 use the same HDF5 (fails for a pip-installed h5py) |
| metview selfcheck | yes | `python -m metview selfcheck` finds Metview 5 (needed to crop the ERA5 GRIB forcing) |
| imports | yes for the regional DA chain | every module under `src_*` is imported; failures outside the `RDA` chain are WARN |
| h5py / netCDF4 | yes | write and read a small file |
| geopandas | yes | read `$PYGLDA_V1_DATA/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp` (without a data root: write and read a small shapefile in a temporary folder) and reproject it (PROJ database) |
| MPI | yes | `mpiexec -n 4 python -m mpi4py.bench helloworld` and an Allreduce sum on 4 ranks |
| pygmt | no | only in `pyglda_vis` |

Exit code 0 only if all required checks pass. `PYGLDA_CHECK_NP` changes the number of MPI ranks.

## 5. Running PyGLDA v1

```bash
conda activate pyglda_v1                          # UCloud: source /work/data_for_w3/envs/activate.sh
export PYGLDA_V1_DATA=/work/data_for_w3           # data root (ERA5/, GRACE/, basin/, w3ra/); or RDA.data_root in the driver
cd py-w3ra/demo                                   # all paths in the code start at '../'
python <driver>.py                                # single-process stages (prepare_Forcing, prepare_GRACE*, single_run)
mpiexec -n <ens+1> python <driver>.py             # OL_run / DA_run: one rank per member + member 0
```

Without a data root the stages stop with a clear error. Each case gets its own folder
`<data_root>/w3ra/cases/<case>/` (run copy of the settings, logs, perturbed obs, DA members, results); what several
cases can share (cropped inputs, spin-up, ensembles, open loops, units, prepared GRACE) is kept once under
`w3ra/domains/` and `w3ra/units/` and reused only if its `made_with.json` matches (`configuration.md` §7).
Example: `python demo_danube.py --basin DRB --stages forcing,grace,spinup`, then
`mpiexec -n 31 python demo_danube.py --basin DRB --stages ol,da` (settings `settings/Danube_DRB`).
How a driver and the four JSON files define an experiment: `configuration.md`.

## 6. Data

The repository holds code only. All data is below one data root (`PYGLDA_V1_DATA` or `RDA.data_root`; UCloud
`/work/data_for_w3`, desktop `/media/user/My Book/Fan/PyGLDA/External Data`). Paths in the JSON files are written as
`${DATA_ROOT}/...`.

```
<data_root>/
  ERA5/Raw_W3/                      W3_ERA5_daily_YYYYMM.grib (daily ERA5 forcing, global, ~1.6 GB/month)
  GRACE/SaGEA/signal_Mascon/        CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc
  GRACE/SaGEA/sample_DDK3/          YYYY-MM_0.npy, 2000 DDK3 samples x 180 x 360 (1 deg), ~1 GB/month
  GRACE/masks/                      GlobalLandMaskForGRACE.hdf5
  GRACE/aux/, GRACE/ESA_SING/       time epochs; simulated ESA-SING observations (input)
  basin/shp/<name>/                 shapefiles of the units (DRB_3_shapefiles, GDRB_shapefiles, Brahmaputra, ESA_SING,
                                    global_shp_new, ...; global/ = the global tile sets formerly in the repository res/)
  w3ra/global/                      pyW3RA/{mask,clim,par} (global W3RA inputs), perturbation_sample/, basin_selection/
  w3ra/domains/ units/ cases/ aggregates/   written by the code (configuration.md §7)
  w3ra/crop_input/ initial_states/ output/ res/ states_sample/, GRACE/{obs,output,OLmean}/
                                    old cases (legacy layout, read with RDA.legacy_layout = True)
  w3ra/legacy_repo/basin_mask/      basin masks formerly in the repository data/basin/mask (legacy layout)
```

`settings/templates/perturbation.json` (all variables listed, none perturbed) is written by
`perturbation.save_default_json()` from `w3ra/global/perturbation_sample`.

| set | where | content |
|---|---|---|
| summer-school sample | desktop `/media/user/My Book/Fan/SummerSchool2025` and `SummerSchool2026` `/External Data` (54-56 GB) | ERA5 2004-10..2005-12 (2026: ..2006-01), pyW3RA global inputs, GRACE for the Brahmaputra exercises |
| full set | UCloud `/work/data_for_w3` (1.7 TB) | ERA5 1994-01..2024-05, crop_input / initial_states / res of the published and ESA-SING cases; **no `sample_DDK3`** |
| DDK3 samples | desktop `PyGLDA_v2_external_data/GRACE/SaGEA/sample_DDK3` (227 GB, 2002-04..2024-03) | needed by `basin_COV` |

## 7. Visualisation env `pyglda_vis`

Figures (`demo/plotshow.py`, the `*visualization*` functions) use pygmt, which is kept out of the model env:

```bash
bash installation/install.sh --vis           # adds pyglda_vis (pygmt 0.13.0, GMT 6.6.0)
conda activate pyglda_vis
python -c "import pygmt; pygmt.show_versions()"
```

`demo/plotshow.py` also imports `mat73` in one function; it is in neither env.

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Could not run the Metview executable ('metview')` | env not activated (python called by path), so `metview` is not on PATH | `conda activate pyglda_v1` |
| `PROJ: proj_create_from_database: Open of .../share/proj failed` | same (no PROJ_LIB/PROJ_DATA) | activate the env; the hook sets both |
| `MAGPLUS_HOME: unbound variable` when activating in a script | script uses `set -u`; the magics hook reads unset variables | do not use `set -u` around `conda activate` |
| `No module named 'pkg_resources'` (cdsapi) | setuptools missing | use the lock / yml (setuptools pinned) |
| h5py and netCDF4 report different HDF5 | h5py installed with pip | reinstall from the lock; never `pip install h5py` |
| `metview -slog` prints `EXIT on ERROR (line 1)` | harmless; `-slog` is not a reliable test (exit code 0 or 1) | use `python -m metview selfcheck` |
| `/tmp/mv.<pid>.<user>` folders accumulate | Metview cache folders | remove old ones with `rm -rf /tmp/mv.*` when no Metview runs |
| `mpiexec` starts N copies of rank 0 | a system `mpiexec` (other MPI) used with the env's mpi4py | use the env's `mpiexec` (`which mpiexec` must point into the env) |
| `demo/demo_test_installation.py` points to `/media/user/My Book/Fan/SummerSchool/External Data` | path from 2025; the samples are now `SummerSchool2025` / `SummerSchool2026` | use `installation/check_install.sh`; for a run, set `RDA.external_data_path` |
