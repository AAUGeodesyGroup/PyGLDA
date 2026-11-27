"""Share of the absolute DA increment per compartment (mean of seeds 1-3, 1e18 / 1 deg) from the basin series of 17_increments."""
import csv, json
import numpy as np
COMP = [('S0', 'Top soil'), ('Ss', 'Shallow soil'), ('Sd', 'Deep soil'), ('Sg', 'Groundwater'), ('snow', 'Snow'),
        ('Sr', 'Surface water'), ('Mleaf', 'Canopy/biomass')]
res = {}
for b in ('Brahmaputra', 'Danube'):
    for m in ('GRACEC', 'NGGM', 'MAGIC'):
        sh = []
        for s in (1, 2, 3):
            r = list(csv.DictReader(open('inputs/inc/jog_%s_%s_s%d_nominal_series.csv' % (m, b, s))))
            ser = {c: np.array([float(x[c]) for x in r]) for c in ('S0', 'Ss', 'Sd', 'Sg', 'Sr', 'Mleaf')}
            ser['snow'] = np.array([float(x['DrySnow']) + float(x['FreeWater']) for x in r])
            tot = sum(np.abs(v).sum() for v in ser.values())
            sh.append([100 * np.abs(ser[c]).sum() / tot for c, _ in COMP])
        sh = np.array(sh)
        res['%s_%s' % (b, m)] = {'mean': sh.mean(0).tolist(), 'min': sh.min(0).tolist(), 'max': sh.max(0).tolist()}
json.dump({'compartments': [n for _, n in COMP], 'shares_pct': res}, open('inc_shares.json', 'w'), indent=1)
print('| basin | mission | ' + ' | '.join(n for _, n in COMP) + ' |')
for k, v in res.items():
    print('| %s | %s | ' % tuple(k.split('_')) + ' | '.join('%.1f' % x for x in v['mean']) + ' |')
print('max seed range (pct points):', round(max(max(np.array(v['max']) - np.array(v['min'])) for v in res.values()), 1))
