"""Step 5 (R3.5) - the 1996-2006 Brahmaputra extension.

Evaluates the extended runs (v1 data layout) against the ESM2.0 truth over 1996-2006 (no 2002-07 cut), the same
metrics as Table 1 plus bootstrap CIs, and the record-length sensitivity (metric over the last N years, N = 2..11).
Usage: python 05_extension_R35.py --domain <d> --ensemble <e> --cases GRACEC=<case> MAGIC=<case> [--start 1996-01-01]
"""
from datetime import date

import numpy as np

from _common import parser, outdir, write_csv, write_json, definitions, C
from jog_eval import evaluate as E, sources as S, stats as ST


def main():
    p = parser(__doc__)
    p.add_argument('--domain', required=True)
    p.add_argument('--ensemble', required=True)
    p.add_argument('--cases', nargs='+', required=True, help='MISSION=case_name')
    p.add_argument('--mask-case', default=C.case_name('MAGIC', '1e18', '1degree', 'Brahmaputra'),
                   help='published case whose 0.1 deg mask matches the domain')
    p.add_argument('--start', default='1996-01-01')
    p.add_argument('--end', default='2006-12-31')
    args = p.parse_args()
    out = outdir(args, '05_extension')
    basin = 'Brahmaputra'
    epochs = S.epochs_between(date.fromisoformat(args.start), date.fromisoformat(args.end))
    cases = dict(x.split('=') for x in args.cases)
    first = S.case_products(next(iter(cases.values())), args.domain, args.ensemble)
    folders = {'model': ('OL', first['OL'], args.mask_case)}
    for m, c in cases.items():
        folders[m] = ('DA', S.case_products(c, args.domain, args.ensemble)['DA'], args.mask_case)
    data = E.load_basin(basin, epochs, folders=folders)
    rows, res = [], {'epochs': [int(epochs[0]), int(epochs[-1])], 'cells': data['cells'], 'missing': data['missing']}
    for d in definitions(args):
        tab = E.table(data, d)
        for prod, vals in tab.items():
            rows.append({'definition': d, 'product': prod, **{k: round(v, 4) for k, v in vals.items()}})
        res['bootstrap_' + d] = ST.bootstrap_table(data['truth'], data['sims'], definition=d,
                                                   pairs=[(m, 'GRACEC') for m in cases if m != 'GRACEC'])
        years = [2, 3, 4, 5, 7, 9, 11]
        res['record_length_' + d] = ST.record_length(data['truth'], data['sims'],
                                                     [min(len(epochs), int(y * 365.25 / 5)) for y in years], 'kge', d)
    write_csv(out / 'extension_table.csv', rows)
    write_json(out / 'extension_details.json', res)
    print('written', out)


if __name__ == '__main__':
    main()
