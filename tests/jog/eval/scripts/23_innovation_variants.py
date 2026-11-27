"""Step 23 (R4.7) - normalized innovation statistics, primary and perturbed-obs variant, per case (wraps step 16).

primary   : d = y - mean(HX), y = stored unperturbed observation, nis = d^2 / (var(HX) + R)   (16_innovations.analyse)
perturbed : d = mean(perturbed obs) - mean(HX) (the innovation the update used), same normalisation
Output: results/23_innovation_variants/innovation_variants.csv (+ json); one row per case and variant.
Usage: python 23_innovation_variants.py --cases <case> [...]
"""
import argparse
import importlib
from pathlib import Path

from _common import write_csv, write_json, C

I16 = importlib.import_module('16_innovations')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--data-root', default=str(C.V1_DATA_ROOT))
    p.add_argument('--out', default=str(C.OUT_ROOT))
    a = p.parse_args()
    out = Path(a.out) / '23_innovation_variants'
    out.mkdir(parents=True, exist_ok=True)
    rows, full = [], {}
    for case in a.cases:
        x = I16.load(Path(a.data_root) / 'w3ra' / 'cases' / case / 'DA' / 'innovations.h5')
        for var, y in (('primary_unperturbed_y', x['y']), ('perturbed_obs_mean', x['obs_pert_mean'])):
            r, _, _ = I16.analyse({**x, 'y': y})
            full['%s/%s' % (case, var)] = r
            rows.append(dict(case=case, variant=var, windows=r['windows'], units=r['units'],
                             nis_mean=round(r['nis_mean'], 4), nis_ci_low=round(r['nis_ci95'][0], 4),
                             nis_ci_high=round(r['nis_ci95'][1], 4), nis_unit_median=round(r['nis_unit_median'], 4),
                             nis_unit_min=round(r['nis_unit_min'], 4), nis_unit_max=round(r['nis_unit_max'], 4),
                             share_windows_below=round(r['share_windows_below'], 4),
                             share_windows_above=round(r['share_windows_above'], 4), bias_d=round(r['bias_d'], 3),
                             rms_d=round(r['rms_d'], 3), spread_hx=round(r['spread_hx'], 3), sigma_R=round(r['sigma_R'], 3)))
    write_csv(out / 'innovation_variants.csv', rows)
    write_json(out / 'innovation_variants.json', full)
    print('written', out)


if __name__ == '__main__':
    main()
