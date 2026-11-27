"""Step 6 (R3.4, R4.1, R4.6, R1.1, R1.3 seeds) - a revision variant against its baseline.

Both runs are read from the v1 data layout. Prints Table-1 style metrics of baseline and variant and the bootstrap
CI of variant - baseline (paired epochs). The baseline is the seeded rerun of the best configuration.
Usage: python 06_compare_variants.py --basin Danube --mission MAGIC --domain <d> --ensemble <e>
                                     --baseline <case> --variant <case> [--label R3.4]
"""
import numpy as np

from _common import parser, outdir, write_csv, write_json, definitions, C
from jog_eval import evaluate as E, sources as S, stats as ST


def main():
    p = parser(__doc__)
    p.add_argument('--basin', required=True)
    p.add_argument('--mission', required=True)
    p.add_argument('--domain', required=True)
    p.add_argument('--ensemble', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--label', default='variant')
    p.add_argument('--mask-case', default=None)
    args = p.parse_args()
    out = outdir(args, '06_variants')
    mask_case = args.mask_case or C.case_name(args.mission, C.BEST['alpha'], C.BEST['grid'], args.basin)
    base = S.case_products(args.baseline, args.domain, args.ensemble)
    var = S.case_products(args.variant, args.domain, args.ensemble)
    folders = {'model': ('OL', base['OL'], mask_case), 'baseline': ('DA', base['DA'], mask_case),
               'variant': ('DA', var['DA'], mask_case)}
    data = E.load_basin(args.basin, np.arange(*C.PAPER_EPOCHS), folders=folders)
    rows, res = [], {}
    for d in definitions(args):
        for prod, vals in E.table(data, d).items():
            rows.append({'label': args.label, 'definition': d, 'product': prod, **{k: round(v, 4) for k, v in vals.items()}})
        res[d] = ST.bootstrap_table(data['truth'], data['sims'], definition=d, pairs=[('variant', 'baseline')])
    tag = '%s_%s_%s' % (args.label, args.basin, args.mission)
    write_csv(out / ('%s.csv' % tag), rows)
    write_json(out / ('%s.json' % tag), res)
    print('written', out)


if __name__ == '__main__':
    main()
