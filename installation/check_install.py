"""PyGLDA v1 (py-w3ra) installation check.

    python installation/check_install.py            # serial checks (run inside the activated env)
    mpiexec -n 4 python installation/check_install.py --mpi   # MPI part (used by check_install.sh)

Exit code 0 only if every REQUIRED check passes. OPTIONAL items (pygmt, modules outside the DA chain) are reported only.
"""
import sys
import os
import importlib
import subprocess
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# modules on the path of a regional DA run (RDA: prepare -> single_run -> OL_run -> DA_run); their import must work
REQUIRED_MODULES = [
    'src_FlowControl.Regional_DA', 'src_FlowControl.DA_GRACE', 'src_FlowControl.OpenLoop',
    'src_FlowControl.SingleModel',
    'src_GHM.preprocess', 'src_GHM.hotrun', 'src_GHM.config_settings', 'src_GHM.config_parameters',
    'src_GHM.model_initialise', 'src_GHM.ext_adapter', 'src_GHM.W3RA_timestep_model',
    'src_DA.EnKF', 'src_DA.Perturbation', 'src_DA.observations', 'src_DA.ExtractStates', 'src_DA.shp2mask',
    'src_DA.Analysis', 'src_DA.ObsDesignMatrix', 'src_DA.configure_DA',
    'src_OBS.prepare_GRACE', 'src_OBS.prepare_GRACE_mascon', 'src_OBS.GRACE_perturbation',
]
SAMPLE_SHP = REPO / 'data' / 'basin' / 'shp' / 'DRB_3_shapefiles' / 'DRB_subbasins.shp'

results = []   # (name, required, ok, detail)


def check(name, required=True):
    def deco(fn):
        try:
            detail = fn()
            results.append((name, required, True, detail or ''))
        except Exception as e:
            results.append((name, required, False, '%s: %s' % (type(e).__name__, str(e).splitlines()[0] if str(e) else '')))
        return fn
    return deco


def serial():
    print('PyGLDA v1 installation check')
    print('  python  :', sys.executable, sys.version.split()[0])
    print('  repo    :', REPO)
    print('  env     :', os.environ.get('CONDA_PREFIX', '(no CONDA_PREFIX - env not activated?)'))
    print('  PROJ_LIB:', os.environ.get('PROJ_LIB'), ' PROJ_DATA:', os.environ.get('PROJ_DATA'))
    print()

    @check('versions')
    def _():
        import numpy, scipy, pandas, h5py, netCDF4, geopandas, shapely, pyproj, mpi4py
        from mpi4py import MPI
        from osgeo import gdal
        v = ['numpy %s' % numpy.__version__, 'scipy %s' % scipy.__version__, 'pandas %s' % pandas.__version__,
             'h5py %s (HDF5 %s)' % (h5py.__version__, h5py.version.hdf5_version),
             'netCDF4 %s (HDF5 %s, netCDF %s)' % (netCDF4.__version__, netCDF4.__hdf5libversion__,
                                                 netCDF4.__netcdf4libversion__),
             'geopandas %s' % geopandas.__version__, 'shapely %s' % shapely.__version__,
             'pyproj %s (PROJ %s)' % (pyproj.__version__, pyproj.proj_version_str), 'gdal %s' % gdal.__version__,
             'mpi4py %s' % mpi4py.__version__, 'MPI: %s' % MPI.Get_library_version().splitlines()[0].strip()]
        try:
            import eccodes
            v.append('eccodes %s' % eccodes.codes_get_api_version())
        except Exception as e:
            v.append('eccodes ? (%s)' % type(e).__name__)
        print('  ' + '\n  '.join(v))
        return ''

    @check('HDF5 consistency (h5py and netCDF4 use the same HDF5)')
    def _():
        import h5py, netCDF4
        a, b = h5py.version.hdf5_version, netCDF4.__hdf5libversion__
        if a != b:
            raise RuntimeError('h5py HDF5 %s != netCDF4 HDF5 %s (pip-installed h5py?)' % (a, b))
        return a

    # before the src_* imports: src_GHM.preprocess imports metview, and a metview session in this process
    # makes the selfcheck child fail ('Could not find a valid Metview installation')
    @check('metview selfcheck')
    def _():
        # write to a file, not a pipe: metview leaves a helper process that keeps an inherited pipe open
        with tempfile.TemporaryFile(mode='w+') as fh:
            subprocess.run([sys.executable, '-m', 'metview', 'selfcheck'], stdout=fh, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=300, start_new_session=True)
            fh.seek(0)
            out = fh.read()
        if 'Your system is ready' not in out:
            raise RuntimeError(out.strip().splitlines()[-1] if out.strip() else 'no output')
        ver = [l for l in out.splitlines() if 'version' in l.lower()]
        return ver[0].strip() if ver else 'ready'

    # import every module of src_*; the modules use relative paths ('../'), so import from <repo>/demo
    cwd = os.getcwd()
    os.chdir(REPO / 'demo')
    sys.path.insert(0, str(REPO))
    all_mods = sorted('%s.%s' % (p.parent.name, p.stem) for p in REPO.glob('src_*/*.py'))
    failed_opt = []
    for m in all_mods:
        req = m in REQUIRED_MODULES
        try:
            importlib.import_module(m)
            results.append(('import %s' % m, req, True, ''))
        except Exception as e:
            last = traceback.extract_tb(e.__traceback__)[-1]
            detail = '%s: %s (%s:%s)' % (type(e).__name__, e, Path(last.filename).name, last.lineno)
            results.append(('import %s' % m, req, False, detail))
            if not req:
                failed_opt.append(m)
    missing = [m for m in REQUIRED_MODULES if m not in all_mods]
    for m in missing:
        results.append(('import %s' % m, True, False, 'module file not found'))
    os.chdir(cwd)

    @check('h5py write/read')
    def _():
        import h5py, numpy as np
        with tempfile.TemporaryDirectory() as d:
            fn = Path(d) / 't.h5'
            with h5py.File(fn, 'w') as f:
                f.create_dataset('x', data=np.arange(10.0))
            with h5py.File(fn, 'r') as f:
                assert float(f['x'][:].sum()) == 45.0
        return 'ok'

    @check('netCDF4 write/open')
    def _():
        import netCDF4, numpy as np
        with tempfile.TemporaryDirectory() as d:
            fn = str(Path(d) / 't.nc')
            ds = netCDF4.Dataset(fn, 'w')
            ds.createDimension('t', 3)
            ds.createVariable('v', 'f8', ('t',))[:] = [1.0, 2.0, 3.0]
            ds.close()
            ds = netCDF4.Dataset(fn)
            assert float(ds['v'][:].sum()) == 6.0
            ds.close()
        return 'ok'

    @check('geopandas read + reproject (%s)' % SAMPLE_SHP.name)
    def _():
        import geopandas as gpd
        g = gpd.read_file(SAMPLE_SHP)
        g2 = g.to_crs(3035)                          # needs a working PROJ database
        assert len(g) > 0 and g2.crs.to_epsg() == 3035
        return '%d sub-basins' % len(g)

    @check('pygmt (visualisation, optional)', required=False)
    def _():
        import pygmt
        return pygmt.__version__

    report(failed_opt)


def report(failed_opt):
    print()
    w = max(len(r[0]) for r in results)
    for name, req, ok, detail in results:
        if name.startswith('import ') and ok:
            continue                                  # list only failing imports
        tag = 'PASS' if ok else ('FAIL' if req else 'WARN')
        print('  [%s] %-*s  %s' % (tag, w, name, detail))
    n_imp = sum(1 for r in results if r[0].startswith('import '))
    n_imp_ok = sum(1 for r in results if r[0].startswith('import ') and r[2])
    print('  imports: %d/%d src_* modules import (failures listed above; WARN = outside the regional DA chain)'
          % (n_imp_ok, n_imp))
    bad = [r for r in results if r[1] and not r[2]]
    print()
    print('RESULT (serial): %s' % ('ALL REQUIRED CHECKS PASSED' if not bad else '%d REQUIRED CHECK(S) FAILED' % len(bad)))
    sys.exit(0 if not bad else 1)


def mpi():
    from mpi4py import MPI
    import numpy as np
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    print('Hello, World! I am process %d of %d on %s.' % (rank, size, MPI.Get_processor_name()), flush=True)
    total = comm.allreduce(np.float64(rank + 1), op=MPI.SUM)
    vec = np.zeros(3)
    comm.Allreduce(np.full(3, float(rank)), vec, op=MPI.SUM)
    expect = size * (size + 1) / 2
    ok = (total == expect) and np.allclose(vec, size * (size - 1) / 2)
    all_ok = comm.allreduce(bool(ok), op=MPI.LAND)
    if rank == 0:
        print('Allreduce sum check: %s (size %d, sum %.0f, expected %.0f; library %s)'
              % ('PASS' if all_ok else 'FAIL', size, total, expect,
                 MPI.Get_library_version().splitlines()[0].strip()), flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    if '--mpi' in sys.argv:
        mpi()
    else:
        serial()
