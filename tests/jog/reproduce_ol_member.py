"""Reproduce published open-loop members with the current code (JoG ESA-SING revision).

    cd <repo>/tests/jog
    python reproduce_ol_member.py --data-root /work/data_for_w3 [--basin Brahmaputra] [--month 2002-09] [--ens 5]

Steps (each its own process; the OL with mpiexec -n ens+1):
  import   copy, read-only from the published folders, into a new test domain of the layout (configuration.md section 7):
           crop_input/<crop>/{forcing (month-1..month+1), clim, par, mask} -> domains/<domain>/input/
           crop_input/<crop>/{ens_forcing (month-1..month+1), ens_par/par.h5} -> domains/<domain>/ensembles/<ensemble>/
           initial_states/<published case>_state.<day before>.h5          -> domains/<domain>/spinup/init/<domain>_state...
           (md5 of every source and copy in <case>/log/import_md5.txt; nothing is written into the published folders)
  ol       RDA.OL_run(skip_perturbation=True) for members 0..ens over the month (settings/JoG_ESA_SING/<MISSION>_<basin>)
  compare  basin series of the new Res_OL_<basin>1degree.h5df vs the published res/<published case>/Res_OL_*.h5df:
           max |diff| per member and store over the days of the month
Member k of the ensemble files is group ens_k, so members 0..ens of a 30-member ensemble can be rerun with ens+1 ranks.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def month_shift(month, k):
    d = datetime.strptime(month + '-01', '%Y-%m-%d')
    y, m = d.year, d.month + k
    while m < 1:
        y, m = y - 1, m + 12
    while m > 12:
        y, m = y + 1, m - 12
    return '%04d-%02d' % (y, m)


def month_days(month):
    b = datetime.strptime(month + '-01', '%Y-%m-%d')
    e = datetime.strptime(month_shift(month, 1) + '-01', '%Y-%m-%d') - timedelta(days=1)
    return b.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d')


def names(a):
    basin = a.basin + '1degree'
    return dict(basin=basin, case=a.case or 'jog_proofB_%s_%s' % (a.basin, a.month.replace('-', '')),
                domain=a.domain or 'jog_proofB_%s' % basin, ensemble='published_%s' % a.crop.split('_')[0].lower(),
                crop=a.crop or 'MAGIC_1e19_1degree_%s' % a.basin,
                published=a.published or 'GRACEC_1e18_1degree_%s' % a.basin,
                settings=a.settings or str(REPO / 'settings' / 'JoG_ESA_SING' / ('GRACEC_%s' % a.basin)))


def setup_rda(a, n):
    sys.path.insert(0, str(REPO))
    from src_FlowControl.Regional_DA import RDA
    b, e = month_days(a.month)
    RDA.data_root = a.data_root
    RDA.setting_dir = n['settings']
    RDA.case, RDA.basin, RDA.ens = n['case'], n['basin'], a.ens
    RDA.shp_path = 'ESA_SING/Grid_1/%s_subbasins.shp' % n['basin']
    RDA.domain, RDA.ensemble, RDA.unit_set, RDA.grace_product = n['domain'], n['ensemble'], n['basin'], 'ESA_SING'
    RDA.cold_begin_time, RDA.cold_end_time = '2000-01-01', (datetime.strptime(b, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    RDA.warm_begin_time, RDA.warm_end_time = b, e
    RDA.resume_begin_time, RDA.resume_end_time = b, e
    RDA.isSet = False
    return RDA


def do_import(a, n):
    RDA = setup_rda(a, n)
    RDA._settings()
    from src_FlowControl import run_context as rc
    L = rc.get_layout()
    w3 = Path(a.data_root) / 'w3ra'
    crop = w3 / 'crop_input' / n['crop']
    day_before = (datetime.strptime(a.month + '-01', '%Y-%m-%d') - timedelta(days=1)).strftime('%Y%m%d')
    pairs = []
    for m in (month_shift(a.month, -1), a.month, month_shift(a.month, 1)):
        if (crop / 'forcing' / ('%s.h5' % m)).is_file():
            pairs.append((crop / 'forcing' / ('%s.h5' % m), L.input_dir / 'forcing' / ('%s.h5' % m)))
    for sub in ('clim', 'par', 'mask'):
        for f in sorted((crop / sub).iterdir()):
            pairs.append((f, L.input_dir / sub / f.name))
    for m in (month_shift(a.month, -1), a.month, month_shift(a.month, 1)):   # the model reads the next day's forcing
        if (crop / 'ens_forcing' / ('%s.h5' % m)).is_file():
            pairs.append((crop / 'ens_forcing' / ('%s.h5' % m), L.ensemble_dir / 'ens_forcing' / ('%s.h5' % m)))
    pairs.append((crop / 'ens_par' / 'par.h5', L.ensemble_dir / 'ens_par' / 'par.h5'))
    pairs.append((w3 / 'initial_states' / ('%s_state.%s.h5' % (n['published'], day_before)),
                  L.init_dir / ('%s_state.%s.h5' % (L.domain, day_before))))
    lines = []
    for src, dst in pairs:
        if dst.exists():
            raise RuntimeError('%s exists: use a new --domain' % dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ms, md = md5(src), md5(dst)
        assert ms == md, (src, dst)
        lines.append('%s  %s  ->  %s' % (ms, src, dst))
    (L.log_dir / 'import_md5.txt').write_text('\n'.join(lines) + '\n')
    rc.note('imported %d published files (read-only copies) into %s, ensemble %s; md5 in import_md5.txt'
            % (len(pairs), L.domain_dir, L.ensemble_dir))
    print('imported %d files' % len(pairs))


def do_ol(a, n):
    RDA = setup_rda(a, n)
    RDA.OL_run(skip_perturbation=True)


def do_compare(a, n):
    import h5py
    import numpy as np
    w3 = Path(a.data_root) / 'w3ra'
    new = w3 / 'cases' / n['case'] / 'res' / ('Res_OL_%s.h5df' % n['basin'])
    pub = w3 / 'res' / n['published'] / ('Res_OL_%s.h5df' % n['basin'])
    out = []
    with h5py.File(new, 'r') as hn, h5py.File(pub, 'r') as hp:
        tn, tp = hn['time'][:], hp['time'][:]
        i0 = int(np.argmin(np.abs(tp - tn[0])))
        nd = len(tn)
        dt = float(np.max(np.abs(tp[i0:i0 + nd] - tn)))
        out.append('new %s (%d days), published %s days %d..%d, max |time diff| %.3e' % (new, nd, pub, i0, i0 + nd - 1, dt))
        units = sorted([k for k in hn if k.startswith('basin_')], key=lambda x: int(x.split('_')[1]))
        stores = list(hn[units[0]].keys())
        for k in range(a.ens + 1):
            worst = {}
            for s in stores:
                d = max(float(np.max(np.abs(hn[u][s][str(k)][:] - hp[u][s][str(k)][i0:i0 + nd]))) for u in units)
                rel = max(float(np.max(np.abs(hp[u][s][str(k)][i0:i0 + nd]))) for u in units)
                worst[s] = (d, rel)
            out.append('member %2d: ' % k + ', '.join('%s %.3e (of %.3e)' % (s, d, r) for s, (d, r) in worst.items()))
            tws = worst.get('TWS', (np.nan, np.nan))[0]
            out.append('           TWS max|diff| %.3e -> %s' % (tws, 'BIT-IDENTICAL' if tws == 0 else 'differs'))
    txt = '\n'.join(out)
    print(txt)
    (w3 / 'cases' / n['case'] / 'log' / 'proofB_compare.txt').write_text(txt + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--basin', choices=['Brahmaputra', 'Danube'], default='Brahmaputra')
    ap.add_argument('--month', default='2002-09')
    ap.add_argument('--ens', type=int, default=5, help='members 0..ens are rerun (ens+1 MPI ranks)')
    ap.add_argument('--crop', help='published crop_input folder with the shared ensemble (default MAGIC_1e19_1degree_<basin>)')
    ap.add_argument('--published', help='published case to compare with (default GRACEC_1e18_1degree_<basin>)')
    ap.add_argument('--settings', help='default settings/JoG_ESA_SING/GRACEC_<basin>')
    ap.add_argument('--case')
    ap.add_argument('--domain')
    ap.add_argument('--steps', default='import,ol,compare')
    ap.add_argument('--step', help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.data_root = str(Path(a.data_root).resolve())
    a.crop = a.crop or 'MAGIC_1e19_1degree_%s' % a.basin
    n = names(a)
    if a.step:
        dict(importdata=do_import, ol=do_ol, compare=do_compare)[a.step](a, n)
        return
    me = [sys.executable, '-u', str(Path(__file__).resolve()), '--data-root', a.data_root, '--basin', a.basin,
          '--month', a.month, '--ens', str(a.ens), '--crop', a.crop, '--published', n['published'],
          '--settings', n['settings'], '--case', n['case'], '--domain', n['domain']]
    for s in a.steps.split(','):
        cmd = (['mpiexec', '-n', str(a.ens + 1)] if s == 'ol' else []) + me + ['--step', 'importdata' if s == 'import' else s]
        print('[%s] step %s' % (datetime.now().strftime('%H:%M:%S'), s), flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO / 'demo')).returncode
        if rc != 0:
            sys.exit('step %s failed (rc %d)' % (s, rc))
    print('PROOF_B_DONE %s' % n['case'])


if __name__ == '__main__':
    main()
