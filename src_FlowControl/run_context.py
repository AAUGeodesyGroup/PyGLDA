"""Run context of PyGLDA v1: data root, data layout, settings run copy, reuse of shared folders, stage guard + logs.

Data layout under <data_root>/w3ra/ (default; RDA.legacy_layout = True keeps the old places exactly as before):
    global/                       pyW3RA/, matlab/, perturbation_sample/, basin_selection/
    domains/<domain>/             model box, shared by every case on it
        input/{forcing,clim,par,mask}            cropped inputs (months cropped only if missing)
        spinup/{state,output}/, spinup/init/<domain>_state.<YYYYMMDD>.h5
        ensembles/<ensemble>/{ens_forcing,ens_par}
        OL/<ensemble>/ens_00..ens_NN/{state,output}/, OL/<ensemble>/nan_sample.h5, OL/<ensemble>/products/OL_mean/
    units/<unit_set>/             observation units, shared by every case with these units
        mask/, DM/<domain>__<ensemble>/DM.hdf5, grace/<product>/ (basin TWS, covariance)
    cases/<case>/                 one DA experiment
        settings/ log/ tmp/ case.json, grace/ol_mean.hdf5, grace/obs/, DA/ens_00..ens_NN/{state,output}/,
        res/ figure/ products/{DA_mean,DA_std,mask}/
    aggregates/<global_run>/      global tile-wise DA: GDA_{DA,OL,GR}_map.h5, SR/
Every shared folder holds made_with.json (the inputs that define it). A folder is reused only if its made_with matches;
a mismatch stops with the differing keys named (never silent reuse, never overwrite); missing months are added.

The design matrix is kept per domain and ensemble (DM/<domain>__<ensemble>/) because it depends on the domain's
parameters and the ensemble's NaN-mask sample, not only on the units.
"""
import functools
import hashlib
import json
import os
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

ENV_DATA_ROOT = 'PYGLDA_V1_DATA'
SETTING_FILES = ('setting.json', 'DA_setting.json', 'pre_process.json', 'perturbation.json')


def resolve_data_root(data_root=None) -> str:
    """data root from the driver attribute, else from the environment variable PYGLDA_V1_DATA, else an error"""
    root = data_root or os.environ.get(ENV_DATA_ROOT)
    if not root:
        raise RuntimeError('PyGLDA v1: no data root. Set RDA.data_root in the driver or the environment variable %s '
                           '(e.g. export %s=/work/data_for_w3).' % (ENV_DATA_ROOT, ENV_DATA_ROOT))
    return str(Path(root))


def substitute(txt: str, data_root: str) -> str:
    return txt.replace('${DATA_ROOT}', data_root)


# ---------------------------------------------------------------- data layout
PRODUCT_BY_KIND = {'Mascon_monthly': 'CSR_mascon', 'SH_monthly': 'SH_monthly', 'ESA_SING': 'ESA_SING',
                   'ESA_SING_ESM3': 'ESA_SING_ESM3'}


class Layout:
    """all paths of one case in the domains/units/cases/aggregates layout"""

    def __init__(self, data_root, case, domain=None, ensemble=None, unit_set=None, grace_product=None,
                 global_run=None):
        self.data_root = str(data_root)
        self.case = case
        self.domain = domain or case
        self.ensemble = ensemble
        self.unit_set = unit_set
        self.grace_product = grace_product
        self.global_run = global_run
        self.w3ra = Path(self.data_root) / 'w3ra'

    # domain
    @property
    def domain_dir(self):
        return self.w3ra / 'domains' / self.domain

    @property
    def input_dir(self):
        return self.domain_dir / 'input'

    @property
    def spinup_dir(self):
        return self.domain_dir / 'spinup'

    @property
    def init_dir(self):
        return self.spinup_dir / 'init'

    @property
    def ensemble_dir(self):
        return self.domain_dir / 'ensembles' / self.ensemble

    @property
    def ol_dir(self):
        return self.domain_dir / 'OL' / self.ensemble

    def member_dir(self, stage, k):
        base = self.ol_dir if stage == 'OL' else self.case_dir / 'DA'
        return base / ('ens_%02d' % int(k))

    @property
    def nan_sample(self):
        return self.ol_dir / 'nan_sample.h5'

    # units
    @property
    def units_dir(self):
        return self.w3ra / 'units' / self.unit_set

    @property
    def mask_dir(self):
        return self.units_dir / 'mask'

    @property
    def dm_dir(self):
        return self.units_dir / 'DM' / ('%s__%s' % (self.domain, self.ensemble))

    @property
    def grace_dir(self):
        return self.units_dir / 'grace' / self.grace_product

    # case
    @property
    def case_dir(self):
        return self.w3ra / 'cases' / self.case

    @property
    def settings_dir(self):
        return self.case_dir / 'settings'

    @property
    def log_dir(self):
        return self.case_dir / 'log'

    @property
    def tmp_dir(self):
        return self.case_dir / 'tmp'

    @property
    def ol_mean_file(self):
        return self.case_dir / 'grace' / 'ol_mean.hdf5'

    @property
    def obs_dir(self):
        return self.case_dir / 'grace' / 'obs'

    @property
    def res_dir(self):
        return self.case_dir / 'res'

    @property
    def figure_dir(self):
        return self.case_dir / 'figure'

    def products_dir(self, kind):
        """kind: DA_mean, DA_std, DA_ens_kk, mask (case), OL_mean (domain OL ensemble) or SR (domain spin-up)"""
        if kind == 'OL_mean':
            return self.ol_dir / 'products' / 'OL_mean'
        if kind == 'SR':
            return self.spinup_dir / 'products' / 'SR'
        return self.case_dir / 'products' / kind

    # aggregates (global tile-wise DA)
    @property
    def aggregate_dir(self):
        return self.w3ra / 'aggregates' / (self.global_run or 'default')

    def make_dirs(self):
        for d in (self.input_dir, self.init_dir, self.ensemble_dir, self.ol_dir, self.mask_dir, self.dm_dir,
                  self.grace_dir, self.settings_dir, self.log_dir, self.tmp_dir, self.obs_dir, self.res_dir,
                  self.figure_dir):
            d.mkdir(parents=True, exist_ok=True)

    def write_case_json(self):
        info = dict(case=self.case, domain=self.domain, ensemble=self.ensemble, unit_set=self.unit_set,
                    grace_product=self.grace_product, global_run=self.global_run, data_root=self.data_root)
        (self.case_dir / 'case.json').write_text(json.dumps(info, indent=2))


# layout of the current process (set by RDA at the start of every stage; None = legacy layout, old places)
_LAYOUT = None


def set_layout(layout):
    global _LAYOUT
    _LAYOUT = layout


def get_layout():
    return _LAYOUT


def make_run_copy(src_dir, dst_dir, data_root: str) -> Path:
    """copy the four JSON files of src_dir to dst_dir (${DATA_ROOT} substituted); src_dir is never written"""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_dir)
    if src_dir.resolve() == dst.resolve():
        return dst
    for f in SETTING_FILES:
        fn = src_dir / f
        if not fn.is_file():
            raise FileNotFoundError('PyGLDA v1: settings file %s not found (settings folder %s)' % (fn, src_dir))
        txt = substitute(fn.read_text(), data_root)
        json.loads(txt)
        (dst / f).write_text(txt)
    return dst


# ---------------------------------------------------------------- path helpers (legacy result when no layout is set)
def crop_dir(out_dir, prefix) -> Path:
    """cropped inputs {forcing,clim,par,mask}: <domain>/input (layout) or <out_dir>/<prefix> (legacy)"""
    L = _LAYOUT
    return L.input_dir if L else Path(out_dir) / prefix


def ens_input_dir(out_dir, prefix) -> Path:
    """perturbed inputs {ens_forcing,ens_par}: <domain>/ensembles/<ensemble> (layout) or <out_dir>/<prefix> (legacy)"""
    L = _LAYOUT
    return L.ensemble_dir if L else Path(out_dir) / prefix


def member_dir(base, kind: str, case: str, ens: int, stage=None) -> Path:
    """states ('state') or outputs ('output') of member <ens> in stage OL/DA; legacy <base>/<kind>_<case>_ensemble_<k>"""
    L = _LAYOUT
    if L and stage in ('OL', 'DA'):
        return L.member_dir(stage, ens) / kind
    return Path(base) / ('%s_%s_ensemble_%s' % (kind, case, ens))


def single_dir(base, kind: str, case: str) -> Path:
    """spin-up (single run) states/outputs: <domain>/spinup/<kind> (layout) or <base>/<kind>_<case> (legacy)"""
    L = _LAYOUT
    return L.spinup_dir / kind if L else Path(base) / ('%s_%s' % (kind, case))


def nan_sample_path(nan_dir, case) -> Path:
    L = _LAYOUT
    return L.nan_sample if L else Path(nan_dir) / ('%s_state.h5' % case)


def ol_mean_path(ol_mean_dir, case, basin) -> Path:
    L = _LAYOUT
    return L.ol_mean_file if L else Path(ol_mean_dir) / ('%s_%s.hdf5' % (case, basin))


def init_name(case) -> str:
    """name in the initial-state file <name>_state.<YYYYMMDD>.h5: the domain (layout) or the case (legacy)"""
    L = _LAYOUT
    return L.domain if L else case


def dm_dir(legacy_default='../temp') -> str:
    L = _LAYOUT
    if L is None:
        return legacy_default
    L.dm_dir.mkdir(parents=True, exist_ok=True)
    return str(L.dm_dir)


def log_dir():
    L = _LAYOUT
    return L.log_dir if L else None


def find_ens_dir(base, kind: str, case: str, ens: int, stage=None) -> Path:
    """reader used since review 14: the member folder of the current layout (or the legacy one)"""
    return member_dir(base, kind, case, ens, stage=stage)


# ---------------------------------------------------------------- data root of the process and model-independent data
# <data_root>/basin/shp/<unit_set or name>/ (shapefiles; global/ = tile sets), <data_root>/GRACE/masks/,
# <data_root>/w3ra/global/{pyW3RA,perturbation_sample,basin_selection}/, <data_root>/w3ra/legacy_repo/basin_mask/
_DATA_ROOT = None


def set_data_root(root):
    global _DATA_ROOT
    _DATA_ROOT = str(root) if root else None


def get_data_root() -> str:
    """the data root of this process: set by RDA (or GDA), else PYGLDA_V1_DATA, else a clear error"""
    if _LAYOUT is not None:
        return _LAYOUT.data_root
    return resolve_data_root(_DATA_ROOT)


def basin_shp_dir() -> Path:
    return Path(get_data_root()) / 'basin' / 'shp'


def basin_path(rel) -> Path:
    """<data_root>/basin/shp/<rel> (e.g. 'global/global_shp_new', 'DRB_3_shapefiles/DRB_subbasins.shp')"""
    return basin_shp_dir() / rel


def resolve_shp(path) -> str:
    """a shapefile given absolute or relative to <data_root>/basin/shp; the old repository paths ../data/... and
    ../res/... stop with a clear error naming the new place"""
    p = str(path)
    if '${DATA_ROOT}' in p:
        p = substitute(p, get_data_root())
    for old, new in (('../data/basin/shp/', ''), ('../data/subbasins/', None), ('../res/', 'global/')):
        if p.startswith(old):
            rest = p[len(old):]
            if new is None:
                rest = '%s/%s' % (Path(rest).name.split('_subbasins')[0], Path(rest).name)
            else:
                rest = new + rest
            raise RuntimeError('PyGLDA v1: %s - the repository data/ and res/ folders moved to the data root. Use '
                               '"%s" (relative to <data_root>/basin/shp) or "%s".' % (p, rest, basin_path(rest)))
    if Path(p).is_absolute():
        return p
    return str(basin_path(p))


_SHAPEFILE = None


def set_shapefile(path):
    """the shapefile of the current case (RDA.shp_path, resolved); DA_GRACE uses it instead of searching by basin"""
    global _SHAPEFILE
    _SHAPEFILE = str(path) if path else None


def find_basin_shp(basin) -> str:
    """shapefile of <basin>: the registered one if its name is <basin>_..., else the only <basin>_*subbasins*.shp below
    <data_root>/basin/shp (several hits are accepted only if byte-identical)"""
    import os
    if _SHAPEFILE and Path(_SHAPEFILE).name.split('_')[0] == basin and Path(_SHAPEFILE).is_file():
        return _SHAPEFILE
    target = []
    for root, dirs, files in os.walk(basin_shp_dir()):
        for file_name in files:
            if file_name.split('_')[0] == basin and file_name.endswith('.shp') and 'subbasins' in file_name:
                target.append(os.path.join(root, file_name))
    target.sort()
    if len(target) > 1:
        def sig(shp):
            return tuple(file_md5(Path(shp).with_suffix(x)) if Path(shp).with_suffix(x).is_file() else ''
                         for x in ('.shp', '.shx', '.dbf'))
        if len({sig(t) for t in target}) == 1:
            return target[0]
    if len(target) != 1:
        raise RuntimeError('PyGLDA v1: basin "%s": %d different shapefiles below %s (%s); set RDA.shp_path to the one '
                           'to use' % (basin, len(target), basin_shp_dir(), ', '.join(target) or 'none'))
    return target[0]


def grace_land_mask() -> str:
    return str(Path(get_data_root()) / 'GRACE' / 'masks' / 'GlobalLandMaskForGRACE.hdf5')


def global_path(rel) -> Path:
    """<data_root>/w3ra/global/<rel> (pyW3RA, perturbation_sample, basin_selection)"""
    return Path(get_data_root()) / 'w3ra' / 'global' / rel


def legacy_mask_dir() -> str:
    """basin masks of the legacy layout (formerly the repository data/basin/mask)"""
    return str(Path(get_data_root()) / 'w3ra' / 'legacy_repo' / 'basin_mask')


# ---------------------------------------------------------------- made_with.json: reuse only if the inputs match
MADE_WITH = 'made_with.json'


def file_md5(fn) -> str:
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _flatten(d, prefix=''):
    out = {}
    for k, v in (d or {}).items():
        key = '%s.%s' % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def made_with_check(folder, what: str, defining: dict, months=None):
    """compare the defining inputs of a shared folder with its made_with.json.
    Returns ('new', months) without made_with.json, ('reuse', missing_months) if the defining inputs match, and raises
    RuntimeError naming the differing keys otherwise."""
    fn = Path(folder) / MADE_WITH
    if not fn.is_file():
        return 'new', list(months or [])
    old = json.loads(fn.read_text())
    a, b = _flatten(old.get('defining', {})), _flatten(json.loads(json.dumps(defining)))
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff:
        raise RuntimeError('PyGLDA v1: %s %s was made with other inputs; differing keys: %s. Use another name '
                           '(RDA.ensemble / RDA.domain / RDA.unit_set / RDA.grace_product) instead of reusing it.'
                           % (what, folder, ', '.join(diff[:12]) + (' ...' if len(diff) > 12 else '')))
    have = set(old.get('months', []))
    return 'reuse', [m for m in (months or []) if m not in have]


def made_with_write(folder, what: str, defining: dict, months=None, extra=None):
    """write/update made_with.json (months are merged with the ones already recorded)"""
    fn = Path(folder) / MADE_WITH
    Path(folder).mkdir(parents=True, exist_ok=True)
    old = json.loads(fn.read_text()) if fn.is_file() else {}
    allm = sorted(set(old.get('months', [])) | set(months or []))
    rec = dict(what=what, defining=json.loads(json.dumps(defining)), months=allm,
               created=old.get('created', time.strftime('%Y-%m-%d %H:%M:%S')),
               updated=time.strftime('%Y-%m-%d %H:%M:%S'), extra=extra or old.get('extra', {}))
    fn.write_text(json.dumps(rec, indent=2))


def note(msg: str):
    """one line on stdout and in <case>/log/reuse.log (reuse decisions etc.)"""
    line = '%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    L = _LAYOUT
    if L is not None:
        try:
            L.log_dir.mkdir(parents=True, exist_ok=True)
            with open(L.log_dir / 'reuse.log', 'a') as f:
                f.write(line + '\n')
        except Exception:
            pass


# ---------------------------------------------------------------- stage guard and error log
REPO = Path(__file__).resolve().parents[1]
CODE_LOG = REPO / 'log'                    # git-ignored; collects the errors of all runs from this checkout
FAIL_TEST_ENV = 'PYGLDA_V1_FAIL_TEST'      # TEST ONLY: "<stage>:<rank>" raises an exception in that stage on that rank


def _rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


def _size():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size()
    except Exception:
        return 1


def _host_info():
    import re
    import socket
    host = socket.gethostname()
    m = re.match(r'j-(\d+)-job', host)                      # UCloud job hostname j-<job id>-job-<n>
    return host, (m.group(1) if m else '')


def _git_info():
    import subprocess
    try:
        rev = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return '%s%s' % (rev or '?', ' (dirty)' if dirty else '')
    except Exception:
        return '?'


def record_error(stage: str, tb_text: str, exc: BaseException = None, case='', data_root='', run_settings=''):
    """write one detail file <repo>/log/errors/<YYYYmmdd-HHMMSS>_<case>_<stage>_rank<k>.txt and append one
    tab-separated line to <repo>/log/errors.log and to <data_root>/w3ra/runs/<case>/log/errors.log (time, case, stage,
    rank, exception, detail file). One file per error, so ranks never write the same file. Never raises."""
    try:
        r, n = _rank(), _size()
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        stamp = time.strftime('%Y%m%d-%H%M%S')
        host, job = _host_info()
        try:
            import numpy
            npv = numpy.__version__
        except Exception:
            npv = '?'
        safe_case = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in (case or 'nocase'))
        det_dir = CODE_LOG / 'errors'
        det_dir.mkdir(parents=True, exist_ok=True)
        det = det_dir / ('%s_%s_%s_rank%d.txt' % (stamp, safe_case, stage, r))
        k = 1
        while det.exists():                                 # same second, same rank: keep both
            det = det_dir / ('%s_%s_%s_rank%d_%d.txt' % (stamp, safe_case, stage, r, k)); k += 1
        det.write_text('\n'.join([
            'time: %s' % now, 'host: %s' % host, 'ucloud_job: %s' % (job or '-'), 'case: %s' % (case or '-'),
            'stage: %s' % stage, 'rank: %d of %d' % (r, n), 'data_root: %s' % (data_root or '-'),
            'run_settings: %s' % (run_settings or '-'), 'git: %s' % _git_info(),
            'python: %s' % sys.version.split()[0], 'numpy: %s' % npv, 'cwd: %s' % os.getcwd(), '', tb_text]))
        msg = ('%s: %s' % (type(exc).__name__, exc)).replace('\t', ' ').replace('\n', ' ')[:300] if exc else '-'
        line = '\t'.join([now, case or '-', stage, str(r), msg, str(det)]) + '\n'
        targets = [CODE_LOG / 'errors.log']
        if _LAYOUT is not None:                             # case log only in the new layout (legacy: code log)
            targets.append(_LAYOUT.log_dir / 'errors.log')
        for fn in targets:
            fn.parent.mkdir(parents=True, exist_ok=True)
            with open(fn, 'a') as f:                         # one short line per error
                f.write(line)
        return det
    except Exception as e:
        print('PyGLDA v1: error log not written: %s' % e, file=sys.stderr, flush=True)
        return None


def log_resources(log_dir, stage: str, wall_s: float):
    if log_dir is None:
        return
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fn = log_dir / 'resources.csv'
        line = '%s,%s,%d,%.2f,%.1f\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), stage, _rank(), wall_s,
                                         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        new = not fn.exists()
        with open(fn, 'a') as f:
            if new:
                f.write('time,stage,rank,wall_s,maxrss_MB\n')
            f.write(line)
    except Exception as e:                                  # the log must never stop a run
        warnings.warn('resources.csv not written: %s' % e)


def stage_guard(stage_name: str, log_dir_fn=None, context_fn=None):
    """decorator for a stage: on an exception print the traceback with the rank, record it in the error logs
    (record_error) and call MPI Abort, so that no other rank waits forever at a barrier; on success append wall time and
    ru_maxrss to <run_dir>/log/resources.csv. context_fn() -> dict(case=, data_root=, run_settings=) for the error log.
    TEST ONLY: environment variable PYGLDA_V1_FAIL_TEST=<stage>:<rank> raises an exception in that stage on that rank."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                ft = os.environ.get(FAIL_TEST_ENV, '')
                if ft and ft == '%s:%d' % (stage_name, _rank()):
                    raise RuntimeError('%s=%s: test exception (test only)' % (FAIL_TEST_ENV, ft))
                out = fn(*args, **kwargs)
            except SystemExit:
                raise
            except BaseException as exc:
                r = _rank()
                sys.stdout = sys.__stdout__
                tb_text = traceback.format_exc()
                msg = 'PyGLDA v1: stage %s failed on rank %d:\n%s' % (stage_name, r, tb_text)
                print(msg, file=sys.stderr, flush=True)
                print(msg, flush=True)
                ctx = {}
                try:
                    ctx = context_fn() if context_fn else {}
                except Exception:
                    pass
                record_error(stage_name, tb_text, exc, **ctx)
                try:
                    from mpi4py import MPI
                    MPI.COMM_WORLD.Abort(1)
                except Exception:
                    pass
                raise
            log_resources(log_dir_fn() if log_dir_fn else None, stage_name, time.time() - t0)
            return out
        return wrapper
    return deco
