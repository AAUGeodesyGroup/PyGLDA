"""Step 4 (R1.4, R4.8) - errors per frequency band.

RMS error (mm, spatial mean of per-cell std) of model (OL), DA and synthetic observation (obs - truth) in the bands
<30 d, 30-90 d, 90-365 d, >365 d, for each mission (best configuration) and basin. The ratio model error /
observation error per band is the quantity behind the hypothesis "DA gains most where the model is worse than the
observation".
Usage: python 04_band_errors.py
"""
import numpy as np

from _common import parser, outdir, write_csv, C
from jog_eval import evaluate as E, sources as S, stats as ST


def main():
    args = parser(__doc__).parse_args()
    out = outdir(args, '04_band_errors')
    epochs = np.arange(*C.PAPER_EPOCHS)
    rows = []
    for basin in args.basins:
        data = E.load_basin(basin, epochs)
        ok = np.isfinite(data['truth']).all(0)
        model = ST.band_errors(data['truth'], data['sims']['model'])
        for m in C.MISSIONS:
            obs, _, _ = S.obs_grid(basin, m, C.BEST['alpha'], epochs)
            obs = np.where(ok[None], obs, np.nan)
            da = ST.band_errors(data['truth'], data['sims'][m])
            ob = ST.band_errors(data['truth'], obs)
            for band in model:
                rows.append({'basin': basin, 'mission': m, 'band': band, 'model_err': round(model[band], 2),
                             'obs_err': round(ob[band], 2), 'DA_err': round(da[band], 2),
                             'model_over_obs': round(model[band] / ob[band], 3) if ob[band] else '',
                             'DA_gain_mm': round(model[band] - da[band], 2)})
    write_csv(out / 'band_errors.csv', rows)
    print('written', out)


if __name__ == '__main__':
    main()
