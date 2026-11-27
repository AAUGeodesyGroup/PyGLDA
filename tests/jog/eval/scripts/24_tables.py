"""Step 24 - Tables 1 and 2 in the revision format (standard metrics, Appendix B).

Rows: basin x metric (NSE, KGE, RMSE). Columns: Model | GRACE-C | NGGM | MAGIC | NGGM - GRACE-C | MAGIC - GRACE-C |
MAGIC - NGGM. Cell: mean over the available seeds, [95 % moving-block bootstrap CI of seed 1] (block 18 epochs,
1000 resamples, paired over products), (seed range min-max). Differences: mean over seeds of the paired difference,
[bootstrap CI of the seed-1 difference], (seed range). No percentages. Spatial means as the paper (NSE/KGE clipped
at -1), epochs 634..875, common 0.5 deg cells.
Table 1 A: fixed 1e18 / 1 deg, nominal seeded baselines jog_<M>_<basin>_s<k>_nominal (k = 1..3, those complete).
Table 1 B: each mission at its best alpha at 1 deg (seed 1 reruns jog_<M>_<basin>_<alpha>_1degree_s1_nominal, plus the
           1e18 baseline), best per metric on the standard definition; the alpha is given; only existing runs.
Table 2  : Brahmaputra seed 1, nominal (jog_<M>_Brahmaputra_s1_nominal) and realistic VCM (jog_<M>_Brahmaputra_s1),
           same layout for the realistic runs + change realistic - nominal per mission with paired CI.
Flag '*': NSE seed range > 0.1.
Seed disagreement: per 0.5 deg cell NSE range over the seeds (PNG per basin, all missions with >= 2 seeds).
A run counts as available when all 242 epochs exist and its newest product file is older than 5 minutes.
Outputs: results/24_tables/{table1A.csv, table1B.csv, table2.csv, tables.md, seed_range_<basin>.png, meta.json}
Usage: python 24_tables.py [--n-boot 1000] [--block 18]
"""
import time

import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import baselines as B, evaluate as E, metrics as M, sources as S, stats as ST

MET = ('nse', 'kge', 'rmse')
PAIRS = [('NGGM', 'GRACEC'), ('MAGIC', 'GRACEC'), ('MAGIC', 'NGGM')]
COLS = ['model', 'GRACEC', 'NGGM', 'MAGIC', 'NGGM-GRACEC', 'MAGIC-GRACEC', 'MAGIC-NGGM']
ALPHAS_B = C.ALPHAS


def ready(folder):
    fs = sorted(folder.glob('TotalWater.*.h5'))
    return len(fs) >= 4 and time.time() - max(f.stat().st_mtime for f in fs) > 300


def load_set(basin, epochs, names):
    """names {product: (kind, folder, mask_case)} -> data or None if a product is incomplete"""
    if not all(ready(f) for _, f, _ in names.values()):
        return None
    d = E.load_basin(basin, epochs, folders=names)
    return None if any(d['missing'].values()) else d


def point(data, keys):
    out = {}
    for k in keys:
        m = M.all_metrics(data['truth'], data['sims'][k], 'paper')
        out[k] = {n: M.spatial_mean(m[n], n) for n in MET}
    for a, b in PAIRS:
        if a in out and b in out:
            out['%s-%s' % (a, b)] = {n: out[a][n] - out[b][n] for n in MET}
    return out


def fmt(v, n):
    return ('%.3f' if n != 'rmse' else '%.1f') % v


def cell(vals, ci, n, flag=False):
    """vals: list over seeds; ci: (lo, hi) of seed 1"""
    s = fmt(np.mean(vals), n)
    if ci is not None:
        s += ' [%s, %s]' % (fmt(ci[0], n), fmt(ci[1], n))
    if len(vals) > 1:
        s += ' (%s-%s)' % (fmt(min(vals), n), fmt(max(vals), n))
    return s + ('*' if flag else '')


def table_rows(tag, basin, seeds_pts, boot, extra=None):
    rows = []
    for n in MET:
        r = {'table': tag, 'basin': basin, 'metric': n.upper()}
        for c in COLS:
            vals = [p[c][n] for p in seeds_pts if c in p]
            if not vals:
                continue
            ci = boot['ci'].get(c, {}).get(n) if boot else None
            flag = n == 'nse' and c in C.MISSIONS and len(vals) > 1 and max(vals) - min(vals) > 0.1
            r[c] = cell(vals, ci, n, flag)
            r[c + '_mean'] = round(float(np.mean(vals)), 4)
            if ci is not None:
                r[c + '_ci_low'], r[c + '_ci_high'] = round(ci[0], 4), round(ci[1], 4)
            r[c + '_seeds'] = ' '.join('%.4f' % v for v in vals)
        if extra:
            r.update(extra.get(n, {}))
        rows.append(r)
    return rows


def md(rows, cols, title):
    lines = ['### ' + title, '', '| basin | metric | ' + ' | '.join(c.replace('GRACEC', 'GRACE-C') for c in cols) + ' |',
             '|' + '---|' * (len(cols) + 2)]
    for r in rows:
        lines.append('| %s | %s | ' % (r['basin'], r['metric']) + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--block', type=int, default=18)
    args = ap.parse_args()
    out = outdir(args, '24_tables')
    epochs = np.arange(*C.PAPER_EPOCHS)
    meta = {'available': {}, 'missing': [], 'block': args.block, 'n_boot': args.n_boot}
    t1a, t1b, t2, mdtxt = [], [], [], ['# Tables 1 and 2 (standard definition) - preview\n',
                                        'Cell: mean over seeds [95 % block-bootstrap CI of seed 1] (seed range); '
                                        '* = NSE seed range > 0.1.\n']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for basin in args.basins:
        # ---- Table 1 A: seeds
        pts, maps, boot = [], {}, None
        for seed in (1, 2, 3):
            data = load_set(basin, epochs, B.folders(basin, seed))
            if data is None:
                meta['missing'] += [B.case(m, basin, seed) for m in C.MISSIONS]
                continue
            meta['available'].setdefault(basin, []).append(seed)
            pts.append(point(data, ['model'] + list(C.MISSIONS)))
            for m in C.MISSIONS:
                maps.setdefault(m, []).append(M.all_metrics(data['truth'], data['sims'][m], 'paper')['nse'])
            if seed == 1:
                boot = ST.bootstrap_table(data['truth'], data['sims'], definition='paper', block=args.block,
                                          n_boot=args.n_boot, pairs=PAIRS)
                lat, lon = data['lat'], data['lon']
            print(basin, 'seed', seed, round(time.time() - t0, 1), 's', flush=True)
        t1a += table_rows('1A', basin, pts, boot)
        # seed-range maps
        if any(len(v) > 1 for v in maps.values()):
            fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
            for ax, m in zip(axs, C.MISSIONS):
                v = np.array(maps[m])
                rng = np.nanmax(np.clip(v, -1, None), 0) - np.nanmin(np.clip(v, -1, None), 0)
                im = ax.pcolormesh(lon, lat, rng, cmap='Blues', vmin=0, vmax=0.5, shading='nearest')
                ax.set_aspect('equal')
                ok = np.isfinite(rng)
                ax.set_title('%s: NSE range over %d seeds; cells > 0.1: %d of %d' %
                             (m.replace('GRACEC', 'GRACE-C'), len(v), int((rng[ok] > 0.1).sum()), int(ok.sum())), fontsize=9)
                meta.setdefault('seed_range', {})['%s_%s' % (basin, m)] = {
                    'seeds': len(v), 'cells_gt_0.1': int((rng[ok] > 0.1).sum()), 'cells_gt_0.3': int((rng[ok] > 0.3).sum()),
                    'cells': int(ok.sum()), 'median_range': float(np.median(rng[ok]))}
            fig.colorbar(im, ax=axs, fraction=0.02, label='NSE max - min (clipped at -1)')
            fig.suptitle('%s: per-cell NSE seed range, nominal 1e18 / 1 deg' % basin, fontsize=10)
            fig.savefig(out / ('seed_range_%s.png' % basin), dpi=120)
            plt.close(fig)
        # ---- Table 1 B: best alpha per mission at 1 deg (seed 1 reruns)
        cand = {}
        for m in C.MISSIONS:
            for a in ALPHAS_B:
                case = 'jog_%s_%s_%s_1degree_s1_nominal' % (m, basin, a)
                f = S.case_products(case, 'jog_%s1degree' % basin, B.ENSEMBLE)['DA']
                if a == '1e18' and not f.exists():
                    f = S.case_products(B.case(m, basin, 1), 'jog_%s1degree' % basin, B.ENSEMBLE)['DA']
                    case = B.case(m, basin, 1)
                if f.exists() and ready(f):
                    cand.setdefault(m, {})[a] = ('DA', f, C.case_name(m, '1e18', '1degree', basin), case)
        meta.setdefault('table1B_candidates', {})[basin] = {m: sorted(v) for m, v in cand.items()}
        if all(m in cand for m in C.MISSIONS) and any(len(v) > 1 for v in cand.values()):
            names = {'model': B.folders(basin)['model']}
            for m, v in cand.items():
                for a, (k, f, mk, case) in v.items():
                    names['%s@%s' % (m, a)] = (k, f, mk)
            data = E.load_basin(basin, epochs, folders=names)
            pall = point(data, list(names))
            sel = {}
            for n in MET:
                pick = {m: (min if n == 'rmse' else max)(cand[m], key=lambda a: pall['%s@%s' % (m, a)][n]) for m in C.MISSIONS}
                sel[n] = pick
            rowsB = []
            for n in MET:
                sub = {'truth': data['truth'], 'sims': {'model': data['sims']['model'],
                                                         **{m: data['sims']['%s@%s' % (m, sel[n][m])] for m in C.MISSIONS}}}
                bt = ST.bootstrap_table(sub['truth'], sub['sims'], metric_names=(n,), definition='paper', block=args.block,
                                        n_boot=args.n_boot, pairs=PAIRS)
                p = point(sub, ['model'] + list(C.MISSIONS))
                r = table_rows('1B', basin, [p], bt)[MET.index(n)]
                r.update({'alpha_' + m: sel[n][m] for m in C.MISSIONS})
                rowsB.append(r)
            t1b += rowsB
        # ---- Table 2 (Brahmaputra, seed 1): nominal vs realistic
        if basin == 'Brahmaputra':
            nom, rea = B.folders(basin, 1, 'nominal'), B.folders(basin, 1, 'realistic')
            names = {'model': nom['model'], **{m + '_nom': nom[m] for m in C.MISSIONS}, **{m: rea[m] for m in C.MISSIONS}}
            if all(ready(f) for _, f, _ in names.values()):
                data = E.load_basin(basin, epochs, folders=names)
                pairs = PAIRS + [(m, m + '_nom') for m in C.MISSIONS]
                bt = ST.bootstrap_table(data['truth'], data['sims'], definition='paper', block=args.block,
                                        n_boot=args.n_boot, pairs=pairs)
                p = point(data, list(names))
                for m in C.MISSIONS:
                    p['%s-%s_nom' % (m, m)] = {n: p[m][n] - p[m + '_nom'][n] for n in MET}
                extra = {}
                for n in MET:
                    extra[n] = {}
                    for m in C.MISSIONS:
                        lo, hi = bt['ci']['%s-%s_nom' % (m, m)][n]
                        extra[n]['nominal_' + m] = cell([p[m + '_nom'][n]], bt['ci'][m + '_nom'][n], n)
                        extra[n]['change_' + m] = cell([p['%s-%s_nom' % (m, m)][n]], (lo, hi), n)
                t2 += table_rows('2', basin, [p], bt, extra)
            else:
                meta['missing'].append('Table 2 realistic/nominal products of Brahmaputra seed 1')
    write_csv(out / 'table1A.csv', t1a)
    if t1b:
        write_csv(out / 'table1B.csv', t1b)
    if t2:
        write_csv(out / 'table2.csv', t2)
    mdtxt.append(md(t1a, COLS, 'Table 1 A - fixed 1e18 / 1 deg, nominal VCM, seeds %s' % meta['available']))
    if t1b:
        mdtxt.append(md(t1b, COLS + ['alpha_GRACEC', 'alpha_NGGM', 'alpha_MAGIC'],
                        'Table 1 B - best alpha per mission and metric at 1 deg (seed 1)'))
    else:
        mdtxt.append('### Table 1 B\n\nnot yet available (per-alpha reruns missing): %s\n' % meta.get('table1B_candidates'))
    if t2:
        mdtxt.append(md(t2, COLS, 'Table 2 - Brahmaputra seed 1, realistic VCM'))
        mdtxt.append(md(t2, ['nominal_GRACEC', 'nominal_NGGM', 'nominal_MAGIC', 'change_GRACEC', 'change_NGGM', 'change_MAGIC'],
                        'Table 2 - nominal (same epochs) and change realistic - nominal [paired CI]'))
    (out / 'tables.md').write_text('\n'.join(mdtxt))
    meta['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'meta.json', meta)
    print('written', out, meta['run_time_s'], 's')


if __name__ == '__main__':
    main()
