"""Compare every HDF5 output of two PyGLDA v1 cases dataset by dataset (bit identity).

    python compare.py --data-root /work/data_for_w3 --case-a reg_base_a --case-b reg_base_b [--report out.txt]

Files of a case: w3ra/crop_input/<case>/, w3ra/output/{state,output}_<case>*/, w3ra/initial_states/<case>_state.*,
w3ra/states_sample/<case>_state.h5, w3ra/res/<case>/, GRACE/{output,obs,OLmean}/<case>/, w3ra/runs/<case>/{DM,mask}/.
Each file is keyed by its path with the case name replaced by <CASE>. Layouts are mapped onto each other:
 - new layout `state_<case>_DA_ensemble_k/state.D` and `state_<case>_OL_ensemble_k/state.D` -> key of the old shared
   folder `state_<case>_ensemble_k/state.D`, the DA file winning where both exist (the old layout overwrote the OL day
   with the DA day); an OL day overwritten in the old layout is reported as 'only in one case (OL kept, new layout)';
 - `output_<case>_{OL,DA}_ensemble_k/` -> `output_<case>_ensemble_k/`.
Datasets are compared for keys, shape, dtype and values (NaN == NaN). String datasets that look like paths are skipped.
Exit code 0 only if every common dataset is bit-identical and no file is missing from case B that case A has (except
the documented OL-kept files).
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

SUFFIXES = ('.h5', '.hdf5', '.h5df')


def case_files(root: Path, case: str):
    pats = ['w3ra/crop_input/%s' % case, 'w3ra/res/%s' % case, 'GRACE/output/%s' % case, 'GRACE/obs/%s' % case,
            'GRACE/OLmean/%s' % case, 'w3ra/runs/%s/DM' % case, 'w3ra/runs/%s/mask' % case]
    files = []
    for p in pats:
        d = root / p
        if d.is_dir():
            files += [f for f in d.rglob('*') if f.suffix in SUFFIXES]
    out = root / 'w3ra' / 'output'
    if out.is_dir():
        for d in out.iterdir():
            if d.is_dir() and re.match(r'(state|output)_%s(_OL|_DA)?(_ensemble_\d+)?$' % re.escape(case), d.name):
                files += [f for f in d.rglob('*') if f.suffix in SUFFIXES]
    for pat in ('w3ra/initial_states/%s_state.*' % case, 'w3ra/states_sample/%s_state.h5' % case):
        files += [f for f in root.glob(pat) if f.suffix in SUFFIXES]
    return files


def keyed_layout(root: Path, case: str):
    """files of a case in the domains/units/cases layout, keyed by the old-layout path (<CASE> for the case)"""
    import json
    info = json.loads((root / 'w3ra' / 'cases' / case / 'case.json').read_text())
    w = root / 'w3ra'
    D, E, U, P = info['domain'], info['ensemble'], info['unit_set'], info['grace_product']
    dom, cas = w / 'domains' / D, w / 'cases' / case
    keys, ol_kept = {}, set()

    def add(folder, prefix, rename=None):
        if not folder.is_dir():
            return
        for f in sorted(folder.rglob('*')):
            if f.suffix in SUFFIXES:
                rel = f.relative_to(folder).as_posix()
                keys[prefix + (rename(rel) if rename else rel)] = f
    add(dom / 'input', 'w3ra/crop_input/<CASE>/')
    add(dom / 'ensembles' / E, 'w3ra/crop_input/<CASE>/')
    add(dom / 'spinup' / 'state', 'w3ra/output/state_<CASE>/')
    add(dom / 'spinup' / 'output', 'w3ra/output/output_<CASE>/')
    add(dom / 'spinup' / 'init', 'w3ra/initial_states/', rename=lambda r: r.replace(D + '_state.', '<CASE>_state.', 1))
    if (dom / 'OL' / E / 'nan_sample.h5').is_file():
        keys['w3ra/states_sample/<CASE>_state.h5'] = dom / 'OL' / E / 'nan_sample.h5'
    add(cas / 'res', 'w3ra/res/<CASE>/')
    add(w / 'units' / U / 'grace' / P, 'GRACE/output/<CASE>/')
    add(cas / 'grace' / 'obs', 'GRACE/obs/<CASE>/')
    if (cas / 'grace' / 'ol_mean.hdf5').is_file():
        keys['GRACE/OLmean/<CASE>/ol_mean'] = cas / 'grace' / 'ol_mean.hdf5'
    add(w / 'units' / U / 'mask', 'units/mask/')                                   # new only
    add(w / 'units' / U / 'DM' / ('%s__%s' % (D, E)), 'units/DM/')                 # new only
    # members: OL first, DA days replace the OL days (the old layout overwrote them)
    for stage, base in (('OL', dom / 'OL' / E), ('DA', cas / 'DA')):
        if not base.is_dir():
            continue
        for m in sorted(base.glob('ens_*')):
            k = int(m.name.split('_')[1])
            for kind in ('state', 'output'):
                for f in sorted((m / kind).glob('*')) if (m / kind).is_dir() else []:
                    if f.suffix not in SUFFIXES:
                        continue
                    key = 'w3ra/output/%s_<CASE>_ensemble_%d/%s' % (kind, k, f.name)
                    if stage == 'DA' and key in keys:
                        keys[key + ' [OL]'] = keys[key]
                        ol_kept.add(key + ' [OL]')
                    keys[key] = f
    return keys, ol_kept


def keyed(root: Path, case: str):
    """{key: path}; plus the set of keys of OL files that the old layout would have overwritten"""
    if (root / 'w3ra' / 'cases' / case / 'case.json').is_file():
        return keyed_layout(root, case)
    keys, ol_kept = {}, set()
    for f in case_files(root, case):
        rel = f.relative_to(root).as_posix().replace(case, '<CASE>')
        m = re.match(r'(w3ra/output/(?:state|output)_<CASE>)_(OL|DA)(_ensemble_\d+/.*)$', rel)
        if m:
            k = m.group(1) + m.group(3)
            if m.group(2) == 'DA':
                if k in keys:                      # an OL file with the same day was there: keep it as OL-only
                    ol_kept.add(k + ' [OL]')
                    keys[k + ' [OL]'] = keys[k]
                keys[k] = f
            else:
                if k in keys:
                    ol_kept.add(k + ' [OL]')
                    keys[k + ' [OL]'] = f
                else:
                    keys[k] = f
            continue
        if rel.startswith('GRACE/OLmean/<CASE>/'):
            rel = 'GRACE/OLmean/<CASE>/ol_mean'
        keys[rel] = f
    return keys, ol_kept


def datasets(fn: Path):
    out = {}
    with h5py.File(fn, 'r') as h:
        h.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def is_pathlike(v):
    a = np.asarray(v)
    if a.dtype.kind in 'SOU':
        flat = [x.decode() if isinstance(x, bytes) else str(x) for x in a.ravel()]
        return any('/' in s for s in flat)
    return False


def compare_file(fa, fb, nan_only=False):
    """list of difference strings (empty = identical); also the max |diff| over numeric datasets.
    nan_only: compare only shape, dtype and NaN pattern (used for the NaN-mask sample, see --sample-mode)"""
    da, db = datasets(fa), datasets(fb)
    diffs, worst = [], 0.0
    for n in sorted(set(da) | set(db)):
        if n not in da or n not in db:
            diffs.append('dataset %s only in %s' % (n, 'A' if n in da else 'B'))
            continue
        a, b = np.asarray(da[n]), np.asarray(db[n])
        if is_pathlike(a) or is_pathlike(b):
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            diffs.append('%s: shape/dtype %s %s vs %s %s' % (n, a.shape, a.dtype, b.shape, b.dtype))
            continue
        if nan_only and a.dtype.kind in 'fc':
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                diffs.append('%s: NaN pattern differs' % n)
            continue
        if a.dtype.kind in 'fc':
            same = np.array_equal(a, b, equal_nan=True)
            if not same:
                d = float(np.nanmax(np.abs(a - b)))
                worst = max(worst, d)
                diffs.append('%s: max|diff| %.3e' % (n, d))
        elif not np.array_equal(a, b):
            diffs.append('%s: values differ' % n)
    return diffs, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--case-a', required=True)
    ap.add_argument('--case-b', required=True)
    ap.add_argument('--report')
    ap.add_argument('--sample-mode', choices=['nan', 'bitwise'], default='nan',
                    help='states_sample/<case>_state.h5 (NaN-mask sample): nan = compare the NaN pattern only (its day '
                         'was arbitrary before issue 11 was fixed), bitwise = compare the values')
    a = ap.parse_args()
    root = Path(a.data_root)
    ka, ol_a = keyed(root, a.case_a)
    kb, ol_b = keyed(root, a.case_b)
    lines, bad, nds, worst = [], 0, 0, 0.0
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    for k in sorted(set(ka) & set(kb)):
        nan_only = a.sample_mode == 'nan' and k.startswith('w3ra/states_sample/')
        diffs, w = compare_file(ka[k], kb[k], nan_only=nan_only)
        worst = max(worst, w)
        if diffs:
            bad += 1
            lines.append('DIFF %s: %s' % (k, '; '.join(diffs[:5]) + (' ...' if len(diffs) > 5 else '')))
    real_only_a = [k for k in only_a if k not in ol_a]
    real_only_b = [k for k in only_b if k not in ol_b]
    lines.append('files: A %d, B %d, common %d, differing %d, max|diff| %.3e' % (len(ka), len(kb), len(set(ka) & set(kb)),
                                                                               bad, worst))
    lines.append('only in A: %d (%d OL-kept)  %s' % (len(only_a), len(only_a) - len(real_only_a), real_only_a[:10]))
    lines.append('only in B: %d (%d OL-kept)  %s' % (len(only_b), len(only_b) - len(real_only_b), real_only_b[:10]))
    ok = bad == 0 and not real_only_a and len(set(ka) & set(kb)) > 0
    lines.append('states_sample compared: %s' % ('NaN pattern only' if a.sample_mode == 'nan' else 'bitwise'))
    lines.append('RESULT: %s' % ('BIT-IDENTICAL' if ok else 'DIFFERENT'))
    txt = '\n'.join(lines)
    print(txt)
    if a.report:
        Path(a.report).write_text(txt + '\n')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
