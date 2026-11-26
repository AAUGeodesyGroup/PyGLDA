import sys

sys.path.append('../')
from src_FlowControl import run_context as _rc

from pathlib import Path
from datetime import datetime, timedelta
# from src_FlowControl.DA_GRACE import DA_GRACE as DA_GRACE
from src_FlowControl.DA_GRACE import DA_GRACE_flexibile as DA_GRACE
from src_FlowControl import run_context as rc
import warnings
import os

'''deprecated: module variables used before RDA.res_output / RDA.figure_output (issue 13). A driver that still sets
them (e.g. `Regional_DA.res_output = ...`) is honoured with a warning; otherwise the roots derive from RDA.data_root.'''
res_output = '/work/data_for_w3/w3ra/res'
# figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
figure_output = '/work/data_for_w3/w3ra/figure'
_DEFAULT_MODULE_RES, _DEFAULT_MODULE_FIG = res_output, figure_output


def _log_dir():
    """<data_root>/w3ra/cases/<case>/log (layout), None for the legacy layout or without a data root"""
    if RDA.legacy_layout:
        return None
    try:
        return RDA._layout().log_dir
    except RuntimeError:
        return None


def _ctx():
    """case, data root and run-copy folder for the error log (src_FlowControl/run_context.record_error); registers
    the layout if the stage failed before doing so, so that the case log gets the error too"""
    try:
        root = rc.resolve_data_root(RDA.data_root)
        if rc.get_layout() is None and not RDA.legacy_layout:
            rc.set_layout(RDA._layout())
    except Exception:
        root = ''
    return dict(case=RDA.case, data_root=root, run_settings=RDA._run_settings or '')


def _rank_log(stage: str, kind: str, rank: int) -> str:
    """rank log file: <case>/log/<stage>_<kind>_rank<k>.txt (layout) or ../log/OL/<kind>_<k>.txt (legacy)"""
    d = _log_dir()
    if d is None:
        return '../log/OL/%s_%s.txt' % (kind, rank)
    d.mkdir(parents=True, exist_ok=True)
    return str(d / ('%s_%s_rank%s.txt' % (stage, kind, rank)))


def _months(begin_day, end_day, extra_month=False):
    """'YYYY-MM' list from begin_day to end_day ('YYYY-MM-DD'); extra_month adds the month after (forcing crop)"""
    from src_GHM.GeoMathKit import GeoMathKit
    end = datetime.strptime(end_day[:7], '%Y-%m')
    if extra_month:
        end = (end.replace(day=28) + timedelta(days=4))
    return [m.strftime('%Y-%m') for m in GeoMathKit.monthListByMonth(begin=begin_day[:7], end=end.strftime('%Y-%m'))]


def _bcast(comm, value):
    return value if comm is None else comm.bcast(value, root=0)


class RDA:
    """
    This is an instance to launch a regional data assimilation automatically. Arbitrary region can be defined.

    data_root: root of the external data (ERA5/, w3ra/, GRACE/); default = environment variable PYGLDA_V1_DATA.
      Settings may write paths as ${DATA_ROOT}/...; every stage works on a run copy of the four JSON files in
      <data_root>/w3ra/runs/<case>/settings/ (the files in setting_dir are never written).
    res_output / figure_output: roots of the collected results and figures (default <data_root>/w3ra/res and
      <data_root>/w3ra/figure); the case name is appended by set_box.
    legacy_layout: True keeps the old folder layout (see src_FlowControl/run_context.py).
    """
    setting_dir = '../settings/Ucloud_DA'
    ens = 30

    data_root = None
    legacy_layout = False
    '''shared folders a case uses (src_FlowControl/run_context.py); defaults: domain = case, ensemble = n<ens>,
    unit_set = basin, grace_product from obs.GRACE.kind (Mascon_monthly -> CSR_mascon)'''
    domain = None
    ensemble = None
    unit_set = None
    grace_product = None
    obs_input = None        # layout + DA_run(skip_obs_perturbation=True): an existing perturbed-obs file to use (copied
                            # into <case>/grace/obs/<basin>_obs_GRACE.hdf5), e.g. ${DATA_ROOT}/GRACE/ESA_SING/...
    res_output = None
    figure_output = None
    _res_root = None
    _fig_root = None
    _run_settings = None

    '''deprecated alias of data_root, used only by config_external_data when data_root is not set'''
    external_data_path = '/media/user/My Book/Fan/SummerSchool/External Data'
    # case = 'RDA_DRB_BasinScale'
    # basin = 'DRB'
    # shp_path = str(_rc.basin_path('DRB_3_shapefiles/DRB_subbasins.shp'))

    # case = 'United_States'
    # basin = 'US'
    # shp_path = str(_rc.basin_path('USgrid/US_subbasins.shp'))

    case = 'Africa'
    basin = 'Africa'
    shp_path = 'Africa/Africa_subbasins.shp'             # absolute or relative to <data_root>/basin/shp

    box = [50.5, 42, 8.5, 29.5]

    # cold_begin_time = '2000-01-01'
    # cold_end_time = '2010-01-31'

    '''for spin-up'''
    cold_begin_time = '2000-01-01'
    cold_end_time = '2001-12-31'

    '''for open-loop'''
    warm_begin_time = '2002-01-01'
    warm_end_time = '2023-03-31'
    # warm_begin_time = '2000-01-01'
    # warm_end_time = '2010-01-31'

    '''for data assimilation'''
    resume_begin_time = '2002-03-31'
    resume_end_time = '2023-03-31'

    # resume_begin_time = '2002-03-31'
    # resume_end_time = '2010-01-31'
    isSet = False

    @staticmethod
    def _layout():
        """the Layout of this case (None for the legacy layout)"""
        if RDA.legacy_layout:
            return None
        import json
        root = rc.resolve_data_root(RDA.data_root)
        product = RDA.grace_product
        if product is None:
            try:
                kind = json.loads(rc.substitute((Path(RDA.setting_dir) / 'DA_setting.json').read_text(),
                                                root))['obs']['GRACE'].get('kind')
            except Exception:
                kind = None
            product = rc.PRODUCT_BY_KIND.get(kind, kind or 'unknown_product')
        return rc.Layout(root, RDA.case, domain=RDA.domain or RDA.case, ensemble=RDA.ensemble or 'n%d' % RDA.ens,
                         unit_set=RDA.unit_set or RDA.basin, grace_product=product)

    @staticmethod
    def _rewrite_paths(settings_dir, L):
        """point the run copy to the layout: initial states, perturbed obs, prepared GRACE, OL mean, NaN-mask sample"""
        import json
        sd = Path(settings_dir)
        a = json.loads((sd / 'DA_setting.json').read_text())
        a['basic']['NaNmaskStatesDir'] = str(L.ol_dir)
        a['obs']['dir'] = str(L.obs_dir)
        a['obs']['GRACE']['preprocess_res'] = str(L.grace_dir)
        a['obs']['GRACE']['OL_mean'] = str(L.ol_mean_file.parent)
        (sd / 'DA_setting.json').write_text(json.dumps(a, indent=4))
        t = json.loads((sd / 'setting.json').read_text())
        t['init']['dir'] = str(L.init_dir)
        (sd / 'setting.json').write_text(json.dumps(t, indent=4))

    @staticmethod
    def _settings(comm=None):
        """register the layout of this process and make the run copy of the settings for this stage (rank 0 only when
        comm is given): <data_root>/w3ra/cases/<case>/settings (layout) or <data_root>/w3ra/runs/<case>/settings
        (legacy); returns its folder"""
        root = rc.resolve_data_root(RDA.data_root)
        rc.set_data_root(root)
        L = RDA._layout()
        rc.set_layout(L)
        rc.set_shapefile(rc.resolve_shp(RDA.shp_path) if RDA.shp_path else None)
        dst = L.settings_dir if L else Path(root) / 'w3ra' / 'runs' / RDA.case / 'settings'
        if comm is None or comm.Get_rank() == 0:
            if L:
                L.make_dirs()
                L.write_case_json()
            rc.make_run_copy(RDA.setting_dir, dst, root)
            if L:
                RDA._rewrite_paths(dst, L)
        if comm is not None:
            comm.barrier()
        RDA._run_settings = str(dst)
        return str(dst)

    @staticmethod
    def activate():
        """for post-processing and visualization without running a stage (e.g. DataManager with out_dir=None):
        register the data root and the layout of this case and return its run copy of the settings. An existing run
        copy (the one the stages ran with) is used as it is; it is made from RDA.setting_dir only if missing."""
        root = rc.resolve_data_root(RDA.data_root)
        rc.set_data_root(root)
        L = RDA._layout()
        rc.set_layout(L)
        rc.set_shapefile(rc.resolve_shp(RDA.shp_path) if RDA.shp_path else None)
        dst = L.settings_dir if L else Path(root) / 'w3ra' / 'runs' / RDA.case / 'settings'
        if not all((dst / f).is_file() for f in rc.SETTING_FILES):
            if L:
                L.make_dirs()
                L.write_case_json()
            rc.make_run_copy(RDA.setting_dir, dst, root)
            if L:
                RDA._rewrite_paths(dst, L)
        RDA._run_settings = str(dst)
        return str(dst)

    @staticmethod
    def _configure(demo):
        """kept for drivers written against review 14; the layout is registered by _settings()"""
        return demo

    @staticmethod
    def _ol_decision(settings):
        """(ensemble state, OL state, ensemble made_with, OL made_with) for the layout; raises on a mismatch"""
        import json
        L = rc.get_layout()
        pert = json.loads((Path(settings) / 'perturbation.json').read_text())
        pert.pop('dir', None)
        ens_def = dict(perturbation=pert, members=RDA.ens, domain=L.domain,
                       seed=os.environ.get('PYGLDA_V1_SEED_INFO', 'numpy global generator, not seeded by PyGLDA'))
        months = _months(RDA.warm_begin_time, RDA.warm_end_time)
        ens_state, missing = rc.made_with_check(L.ensemble_dir, 'ensemble', ens_def, months)
        if ens_state == 'reuse' and missing:
            raise RuntimeError('PyGLDA v1: ensemble %s has no perturbed inputs for %s; perturbing again would change '
                               'the existing members. Use a new RDA.ensemble name.' % (L.ensemble_dir, missing))
        ol_def = dict(ensemble=L.ensemble, members=RDA.ens, ol_begin=RDA.warm_begin_time, ol_end=RDA.warm_end_time,
                      spinup=[RDA.cold_begin_time, RDA.cold_end_time], domain=L.domain)
        ol_state = rc.made_with_check(L.ol_dir, 'open loop', ol_def)[0]
        if ol_state == 'reuse' and ens_state != 'reuse':
            raise RuntimeError('PyGLDA v1: open loop %s exists but its ensemble %s has no made_with.json'
                               % (L.ol_dir, L.ensemble_dir))
        rc.note('ensemble %s: %s; open loop %s: %s' % (L.ensemble, ens_state, L.ol_dir, ol_state))
        return ens_state, ol_state, ens_def, ol_def

    @staticmethod
    def _check_DA_start(demo):
        """issue 6: fromdate must be the first day of an observation window, and every member's OL state of the day
        before must exist; otherwise stop with the first valid start date. The filter itself is not changed."""
        import json
        from src_DA.configure_DA import config_DA
        dp = Path(demo.setting_dir)
        cfg = config_DA.loadjson(dp / 'DA_setting.json').process()
        aux = demo.get_obs_aux(configDA=cfg, begin_day=RDA.warm_begin_time, end_day=RDA.resume_end_time)
        starts = sorted({datetime.strptime(d.split('_')[0], '%Y-%m-%d')                              # as in EnKF.py
                         for d in aux.getTimeReference()['duration']})
        out_dir = Path(json.load(open(dp / 'setting.json'))['output']['dir'])
        start = datetime.strptime(RDA.resume_begin_time, '%Y-%m-%d')

        def states_ok(day):
            prev = (day - timedelta(days=1)).strftime('%Y%m%d')
            return all((rc.find_ens_dir(out_dir, 'state', RDA.case, k, stage='OL') / ('state.%s.h5' % prev)).is_file()
                       for k in range(RDA.ens + 1))

        if start in starts and states_ok(start):
            return
        valid = [d for d in starts if d >= start and states_ok(d)] or [d for d in starts if states_ok(d)]
        hint = valid[0].strftime('%Y-%m-%d') if valid else 'none (no window start has the OL states of the day before)'
        reason = ('is not the first day of an observation window' if start not in starts else
                  'has no OL state of the day before for every member')
        raise RuntimeError('PyGLDA v1: DA start %s %s. First valid DA start date: %s.'
                           % (RDA.resume_begin_time, reason, hint))

    @staticmethod
    def config_external_data():
        """rewrite every path that contains /w3ra/, /GRACE/ or /ERA5/ so that it starts with the data root
        (RDA.data_root, else the deprecated RDA.external_data_path). Only the run copy of the settings is written."""
        import json
        base = RDA.data_root or RDA.external_data_path
        if RDA.data_root is None:
            RDA.data_root = base
        run_settings = RDA._settings()

        def reformulate(st: str):
            new = None
            for keyword in ['/w3ra/', '/GRACE/', '/ERA5/']:
                if keyword in st:
                    new = base + keyword + st.split(keyword)[1]
                    break
            return new

        def func(dict_data):
            for key, vv in dict_data.items():
                if type(vv) == str:
                    new = reformulate(vv)
                    if new is not None:
                        dict_data[key] = new
                    pass

                elif type(vv) == dict:
                    func(vv)
                else:
                    continue

        for file_name in ['setting.json', 'DA_setting.json', 'pre_process.json', 'perturbation.json']:
            config = json.load(open(Path(run_settings) / file_name, 'r'))
            func(config)
            with open(Path(run_settings) / file_name, 'w') as f:
                json.dump(config, f, indent=4)
        '''later stages copy the settings again from setting_dir: keep using the rewritten run copy as the source'''
        RDA.setting_dir = run_settings

        pass

    @staticmethod
    def _prepare_grace(cls, dir_in, cov_dir_in, dir_out, is_diagonal):
        """GRACE preparation of the observation units (basin TWS, grids, covariance). Layout: masks in
        units/<unit_set>/mask, products in units/<unit_set>/grace/<grace_product> with made_with.json (reused when the
        inputs match and every month is there; missing months -> recomputed over the whole span of old + new months)."""
        RDA._settings()
        L = rc.get_layout()
        t1 = datetime.strptime(RDA.warm_begin_time, '%Y-%m-%d').strftime('%Y-%m')
        t2 = datetime.strptime(RDA.warm_end_time, '%Y-%m-%d').strftime('%Y-%m')
        if L is None:
            GR = cls(basin_name=RDA.basin, shp_path=RDA.shp_path, mask_dir=rc.legacy_mask_dir())
            GR.generate_mask()
            GR.basin_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
            GR.grid_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
            GR.basin_COV(month_begin=t1, month_end=t2, dir_in=cov_dir_in, dir_out=dir_out, is_diagonal=is_diagonal)
            return
        shp_file = rc.resolve_shp(RDA.shp_path)
        shp_md5 = rc.file_md5(shp_file)
        units_def = dict(shapefile=shp_file, shapefile_md5=shp_md5, basin=RDA.basin, res=[0.5, 1])
        units_state = rc.made_with_check(L.units_dir, 'unit set', units_def)[0]
        masks_ok = all((L.mask_dir / ('%s_res_%s.h5' % (RDA.basin, r))).is_file() for r in ('0.5', '1'))
        units_reuse = units_state == 'reuse' and masks_ok
        if str(Path(dir_out)) != str(L.grace_dir):
            rc.note('GRACE products go to %s (the layout), not to %s' % (L.grace_dir, dir_out))
        dir_out = str(L.grace_dir)
        product_def = dict(product=L.grace_product, preparation=cls.__name__, dir_in=str(dir_in),
                           cov_dir_in=str(cov_dir_in), is_diagonal=bool(is_diagonal), units_shapefile_md5=shp_md5)
        months = _months(RDA.warm_begin_time, RDA.warm_end_time)
        state, missing = rc.made_with_check(L.grace_dir, 'GRACE product', product_def, months)
        if state == 'reuse' and not missing and units_reuse:
            rc.note('reused GRACE product %s (months %s..%s present)' % (L.grace_dir, t1, t2))
            return
        GR = cls(basin_name=RDA.basin, shp_path=RDA.shp_path, mask_dir=L.mask_dir)
        if units_reuse:                          # shared masks are kept; only the sub-basin areas for basin_TWS
            GR.generate_mask(res_05=False, res_10=False)
            rc.note('reused unit masks %s' % L.mask_dir)
        else:
            GR.generate_mask()
            rc.made_with_write(L.units_dir, 'unit set', units_def)
        if state == 'reuse':
            import json
            have = json.loads((L.grace_dir / rc.MADE_WITH).read_text())['months']
            span = sorted(set(have) | set(months))
            t1, t2 = span[0], span[-1]
            rc.note('GRACE product %s: adding months %s (recomputing %s..%s)' % (L.grace_dir, missing, t1, t2))
            months = span
        GR.basin_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
        GR.grid_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
        GR.basin_COV(month_begin=t1, month_end=t2, dir_in=cov_dir_in, dir_out=dir_out, is_diagonal=is_diagonal)
        rc.made_with_write(L.grace_dir, 'GRACE product', product_def, months)

    @staticmethod
    @rc.stage_guard('prepare_GRACE', _log_dir, _ctx)
    def prepare_GRACE(dir_in='/media/user/My Book/Fan/GRACE/ewh',
                      cov_dir_in='/media/user/My Book/Fan/GRACE/ewh',
                      dir_out='/media/user/My Book/Fan/GRACE/output', **kwargs):
        """GRACE preparation from SH grids (SH_monthly)"""
        from src_OBS.prepare_GRACE import GRACE_preparation
        RDA._prepare_grace(GRACE_preparation, dir_in, cov_dir_in, dir_out, kwargs['is_diagonal'])

    @staticmethod
    @rc.stage_guard('prepare_GRACE_Mascon', _log_dir, _ctx)
    def prepare_GRACE_Mascon(dir_in='/media/user/My Book/Fan/GRACE/ewh',
                             cov_dir_in='/media/user/My Book/Fan/GRACE/ewh',
                             dir_out='/media/user/My Book/Fan/GRACE/output', **kwargs):
        """GRACE preparation from CSR mascons (Mascon_monthly)"""
        from src_OBS.prepare_GRACE_mascon import GRACE_CSR_mascon
        RDA._prepare_grace(GRACE_CSR_mascon, dir_in, cov_dir_in, dir_out, kwargs['is_diagonal'])

    @staticmethod
    @rc.stage_guard('forcing', _log_dir, _ctx)
    def prepare_Forcing():
        from src_FlowControl.SingleModel import SingleModel
        from src_GHM.EnumType import init_mode
        '''pre-process input forcing field'''
        # GDA.setting_dir = '../settings/single_run'

        '''configuration'''

        RDA.set_box(create_folder=False)

        demo = RDA._configure(SingleModel(case=RDA.case, setting_dir=RDA._settings()))

        demo.configure_time(begin_time=RDA.cold_begin_time, end_time=RDA.resume_end_time)

        demo.configure_area(box=RDA.box, basin=RDA.basin)

        '''change the sub-setting files according to the main setting'''
        demo.generate_settings(mode=init_mode.cold)

        '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
        L = rc.get_layout()
        if L is None:
            demo.preprocess()
            return
        '''layout: the domain input is shared; made_with = box, resolution and sources; only missing months are cropped'''
        import json
        sd = Path(RDA._run_settings)
        pp = json.loads((sd / 'pre_process.json').read_text())
        st = json.loads((sd / 'setting.json').read_text())
        domain_def = dict(box=[round(float(x), 6) for x in RDA.box], res=st['run']['res'],
                          exttype=st['run'].get('exttype'), forcing_dir=pp['forcing_dir'], mask_dir=pp['mask_dir'],
                          clim_dir=pp['clim_dir'], par_dir=pp['par_dir'])
        months = _months(RDA.cold_begin_time, RDA.resume_end_time, extra_month=True)
        state, missing = rc.made_with_check(L.domain_dir, 'domain', domain_def, months)
        if state == 'reuse' and not missing:
            rc.note('reused domain input %s (months %s..%s present)' % (L.input_dir, months[0], months[-1]))
            return
        if state == 'reuse':
            rc.note('domain input %s: cropping missing months %s' % (L.input_dir, missing))
        demo.preprocess()
        rc.made_with_write(L.domain_dir, 'domain', domain_def, months)

    @staticmethod
    def set_box(create_folder=True):
        import geopandas as gpd
        rc.set_data_root(rc.resolve_data_root(RDA.data_root))

        if RDA.isSet:
            return

        gf = gpd.read_file(rc.resolve_shp(RDA.shp_path))
        RDA.box = list(gf.total_bounds[[3, 1, 0, 2]])  # latmax, latmin, lonmin, lonmax

        '''slightly expand the box to ensure a complete coverage of specified area'''
        if RDA.box[0] + 0.2 > 90:
            pass
        else:
            RDA.box[0] += 0.2

        if RDA.box[1] - 0.2 < -90:
            pass
        else:
            RDA.box[1] -= 0.2

        if RDA.box[2] - 0.2 < -180:
            pass
        else:
            RDA.box[2] -= 0.2

        if RDA.box[3] + 0.2 > 180:
            pass
        else:
            RDA.box[3] += 0.2

        '''layout: <case>/res and <case>/figure. Legacy roots: RDA.res_output / RDA.figure_output if the driver set
        them, else the deprecated module variables if a driver changed them, else <data_root>/w3ra/{res,figure}; the
        case name is appended'''
        L = RDA._layout() if not RDA.legacy_layout else None
        if L is not None:
            RDA.figure_output, RDA.res_output = str(L.figure_dir), str(L.res_dir)
            RDA.isSet = True
            if create_folder:
                L.figure_dir.mkdir(parents=True, exist_ok=True)
                L.res_dir.mkdir(parents=True, exist_ok=True)
            return
        if RDA._res_root is None:
            if RDA.res_output is not None:
                RDA._res_root, RDA._fig_root = RDA.res_output, RDA.figure_output
            elif res_output != _DEFAULT_MODULE_RES or figure_output != _DEFAULT_MODULE_FIG:
                warnings.warn('Regional_DA.res_output/figure_output are deprecated; set RDA.res_output/figure_output',
                              DeprecationWarning)
                RDA._res_root, RDA._fig_root = res_output, figure_output
            else:
                root = rc.resolve_data_root(RDA.data_root)
                RDA._res_root, RDA._fig_root = str(Path(root) / 'w3ra' / 'res'), str(Path(root) / 'w3ra' / 'figure')
            if RDA._fig_root is None:
                RDA._fig_root = str(Path(rc.resolve_data_root(RDA.data_root)) / 'w3ra' / 'figure')
        dn1 = Path(RDA._fig_root) / RDA.case
        RDA.figure_output = str(dn1)
        dn2 = Path(RDA._res_root) / RDA.case
        RDA.res_output = str(dn2)

        RDA.isSet = True

        if create_folder:
            '''generate case-specified folder to store figures'''
            if not dn1.is_dir():
                dn1.mkdir(parents=True, exist_ok=True)
            '''generate case-specified folder to store res'''
            if not dn2.is_dir():
                dn2.mkdir(parents=True, exist_ok=True)

        pass

    @staticmethod
    @rc.stage_guard('spinup', _log_dir, _ctx)
    def single_run(skip_croping_data=True, skip_signal_extraction=True):
        from src_FlowControl.SingleModel import SingleModel
        from src_GHM.EnumType import init_mode

        '''configuration'''
        RDA.set_box()
        demo = RDA._configure(SingleModel(case=RDA.case, setting_dir=RDA._settings()))
        demo.configure_time(begin_time=RDA.cold_begin_time, end_time=RDA.cold_end_time)
        demo.configure_area(box=RDA.box, basin=RDA.basin)

        '''change the sub-setting files according to the main setting'''
        demo.generate_settings(mode=init_mode.cold)

        '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
        if not skip_croping_data:
            demo.preprocess()

        '''run the model (layout: the spin-up of the domain is shared; reused when the spin-up period matches)'''
        L = rc.get_layout()
        spin_def = dict(cold_begin=RDA.cold_begin_time, cold_end=RDA.cold_end_time)
        state = rc.made_with_check(L.spinup_dir, 'spin-up', spin_def)[0] if L else 'new'
        last = L.spinup_dir / 'state' / ('state.%s.h5' % RDA.cold_end_time.replace('-', '')) if L else None
        if state == 'reuse' and last.is_file():
            rc.note('reused spin-up %s (%s..%s)' % (L.spinup_dir, RDA.cold_begin_time, RDA.cold_end_time))
        else:
            demo.model_run()
            if L:
                rc.made_with_write(L.spinup_dir, 'spin-up', spin_def)

        '''create ini states (the last spin-up day, named for the day before the open loop)'''
        modifydate = (datetime.strptime(RDA.warm_begin_time, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y%m%d')
        ini = L.init_dir / ('%s_state.%s.h5' % (L.domain, modifydate)) if L else None
        if state == 'reuse' and ini is not None and ini.is_file():
            rc.note('reused initial state %s' % ini)          # shared: never rewritten
        else:
            demo.create_ini_states(mode=init_mode.cold, modifydate=modifydate)

        if not skip_signal_extraction:
            demo.extract_signal()

        pass

    @staticmethod
    def single_run_visualization():
        from src_FlowControl.SingleModel import SingleModel
        from src_GHM.EnumType import init_mode
        from datetime import datetime, timedelta

        '''configuration'''
        RDA.set_box()
        demo = SingleModel(case=RDA.case, setting_dir=RDA.activate())
        demo.configure_time(begin_time=RDA.cold_begin_time, end_time=RDA.cold_end_time)
        demo.configure_area(box=RDA.box, basin=RDA.basin)

        '''change the sub-setting files according to the main setting'''
        demo.generate_settings(mode=init_mode.cold)

        '''generate/save the figure'''
        fig_postfix = '0'
        # demo.visualize_signal(fig_path=GDA.figure_output, fig_postfix=fig_postfix)
        demo.visualize_comparison_GRACE(fig_path=RDA.figure_output, fig_postfix=fig_postfix)

        pass

    @staticmethod
    @rc.stage_guard('ol', _log_dir, _ctx)
    def OL_run(skip_perturbation=True):
        # from src_FlowControl.DA_GRACE import DA_GRACE
        from src_GHM.EnumType import init_mode
        from mpi4py import MPI
        import sys
        temp = sys.stdout

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (RDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (RDA.ens + 1)

        RDA.set_box()
        settings = RDA._settings(comm=comm)

        '''configuration'''
        demo = RDA._configure(DA_GRACE(case=RDA.case, setting_dir=settings, ens=RDA.ens))

        demo.configure_time(begin_time=RDA.warm_begin_time, end_time=RDA.warm_end_time)
        demo.configure_area(box=RDA.box, basin=RDA.basin)

        if rank == 0:
            print('\n=================Welcome to the OL experiment for: %s====================' % RDA.case)

            '''change the sub-setting files according to the main setting'''
            demo.generate_settings(mode=init_mode.warm)

            # '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
            # demo.preprocess()

            '''layout: ensemble and open loop are shared by every case on the domain; reused if made_with matches'''
            decision = RDA._ol_decision(settings) if rc.get_layout() else ('new', 'new', None, None)

            '''generate the perturbation for the data'''
            if not skip_perturbation and decision[0] == 'new':
                demo.perturbation()
        else:
            decision = None
        decision = _bcast(comm, decision)
        ens_state, ol_state, ens_def, ol_def = decision

        comm.barrier()

        '''run the model'''
        if ol_state == 'reuse':
            rc.note('reused open loop %s' % rc.get_layout().ol_dir) if rank == 0 else None
        else:
            demo.model_run()

        '''Analysis the model output/states'''
        if rank > 0:
            f = open(_rank_log('OL', 'signal_log', rank), 'w')
            sys.stdout = f
            sys.stdout.flush()

        if ol_state != 'reuse':
            print('\n\n==== Result analysis, post-processing, and collection =====')
            demo.extract_signal(postfix='OL')

        comm.barrier()
        if rank == 0 and rc.get_layout() is not None:
            L = rc.get_layout()
            if ens_state == 'new':
                rc.made_with_write(L.ensemble_dir, 'ensemble', ens_def, _months(RDA.warm_begin_time, RDA.warm_end_time))
            if ol_state == 'new':
                rc.made_with_write(L.ol_dir, 'open loop', ol_def)

        if rank == 0:
            print('\nPost-processing and collecting results ...')
            demo.post_processing(file_postfix='OL', save_dir=RDA.res_output)

            '''change the time to get prepared for DA experiment'''
            demo = RDA._configure(DA_GRACE(case=RDA.case, setting_dir=settings, ens=RDA.ens))

            demo.configure_time(begin_time=RDA.resume_begin_time, end_time=RDA.resume_end_time)
            demo.configure_area(box=RDA.box, basin=RDA.basin)
            demo.generate_settings(mode=init_mode.resume)

            '''gather the mean value of the open loop: ensemble mean of the temporal mean'''
            demo.gather_OLmean(post_fix='OL')

            '''get states samples for creating NaN mask (layout: shared with the open loop; kept when reused)'''
            if ol_state == 'reuse' and rc.get_layout().nan_sample.is_file():
                rc.note('reused NaN-mask sample %s' % rc.get_layout().nan_sample)
            else:
                demo.get_states_sample_for_mask()

        '''Job finished'''
        print('\njob finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.stdout = temp

        pass

    @staticmethod
    @rc.stage_guard('da', _log_dir, _ctx)
    def DA_run(skip_obs_perturbation=True):
        # from src_FlowControl.DA_GRACE import DA_GRACE
        from src_GHM.EnumType import init_mode
        from mpi4py import MPI
        import sys
        temp = sys.stdout

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (RDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (RDA.ens + 1)

        RDA.set_box()
        settings = RDA._settings(comm=comm)

        '''configuration'''
        demo = RDA._configure(DA_GRACE(case=RDA.case, setting_dir=settings, ens=RDA.ens))

        demo.configure_time(begin_time=RDA.resume_begin_time, end_time=RDA.resume_end_time)
        demo.configure_area(box=RDA.box, basin=RDA.basin)

        if rank == 0:
            print('\n\n=================Welcome to the DA experiment for: %s====================' % RDA.case)
            print('\n\n===================== DA preparation ==========================')
            '''change the sub-setting files according to the main setting'''
            demo.generate_settings(mode=init_mode.resume)

            '''issue 6: stop early with a clear message if the DA cannot start on resume_begin_time'''
            RDA._check_DA_start(demo)

            # '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
            # demo.preprocess()

            '''layout: the perturbed obs live in <case>/grace/obs; with skip_obs_perturbation they come from RDA.obs_input
            (copied once) or must already be there'''
            L = rc.get_layout()
            if skip_obs_perturbation and L is not None:
                obs_fn = L.obs_dir / ('%s_obs_GRACE.hdf5' % RDA.basin)
                if not obs_fn.is_file():
                    if RDA.obs_input is None:
                        raise RuntimeError('PyGLDA v1: skip_obs_perturbation=True but %s does not exist; set RDA.obs_input '
                                           'to the perturbed-obs file to use (e.g. ${DATA_ROOT}/GRACE/ESA_SING/'
                                           'Europe_obs_GRACE.hdf5) or run with skip_obs_perturbation=False' % obs_fn)
                    import shutil
                    src = rc.substitute(str(RDA.obs_input), L.data_root)
                    obs_fn.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, obs_fn)
                    rc.note('copied obs input %s (md5 %s) to %s' % (src, rc.file_md5(src), obs_fn))
                else:
                    rc.note('reused perturbed obs %s' % obs_fn)

            '''generate the perturbation for the data'''
            if not skip_obs_perturbation:
                '''prepare the GRACE observation over region of interest'''
                '''Calculation of the COV is time-consuming, so we do not suggest to repeatedly execute this command '''
                # demo.GRACE_obs_preprocess()

                '''perturb the GRACE obs for later use'''
                demo.generate_perturbed_GRACE_obs()

            '''prepare the design matrix (layout: shared per units, domain and ensemble; reused if made_with matches)'''
            L = rc.get_layout()
            if L is None:
                demo.prepare_design_matrix()
            else:
                import json
                layer = json.loads((Path(settings) / 'DA_setting.json').read_text())['basic']['layer']
                dm_def = dict(units_shapefile_md5=rc.file_md5(rc.resolve_shp(RDA.shp_path)), domain=L.domain, ensemble=L.ensemble,
                              layer=layer)
                if rc.made_with_check(L.dm_dir, 'design matrix', dm_def)[0] == 'reuse' and \
                        (L.dm_dir / 'DM.hdf5').is_file():
                    rc.note('reused design matrix %s' % L.dm_dir)
                else:
                    demo.prepare_design_matrix()
                    rc.made_with_write(L.dm_dir, 'design matrix', dm_def)

        comm.barrier()

        '''run the model'''
        demo.run_DA(rank=rank)

        '''Analysis the model output/states'''
        if rank > 0:
            f = open(_rank_log('DA', 'signal_log', rank), 'w')
            sys.stdout = f
            sys.stdout.flush()

        print('\n\n==== Result analysis, post-processing, and collection =====')
        demo.extract_signal(postfix='DA')

        comm.barrier()

        if rank == 0:
            print('\nPost-processing and collecting results ...')
            demo.post_processing(file_postfix='DA', save_dir=RDA.res_output, isGRACE=True)
            '''layout: the case tmp folder is emptied at the end of a successful run'''
            L = rc.get_layout()
            if L is not None and L.tmp_dir.is_dir():
                import shutil
                for x in L.tmp_dir.iterdir():
                    shutil.rmtree(x) if x.is_dir() else x.unlink()

        '''Job finished'''
        print('\njob finished: %s, %s' % (RDA.case, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        sys.stdout = temp

        pass

    @staticmethod
    def DA_visualization_basin_ensemble(allow_pop_up=False):
        # from src_FlowControl.DA_GRACE import DA_GRACE
        from src_GHM.EnumType import init_mode
        """
        this is to visualize assimilated spread for each vertical compartment
        """
        RDA.set_box()

        '''configuration'''
        demo = DA_GRACE(case=RDA.case, setting_dir=RDA.activate(), ens=RDA.ens)

        demo.configure_time(begin_time=RDA.resume_begin_time, end_time=RDA.resume_end_time)
        demo.configure_area(box=RDA.box, basin=RDA.basin)
        demo.generate_settings(mode=init_mode.resume)

        '''generate/save the figure'''
        fig_postfix = '0'
        demo.visualize_signal(fig_path=RDA.figure_output, fig_postfix=fig_postfix, file_postfix='DA',
                              data_dir=RDA.res_output, allow_pop_up=allow_pop_up)
        pass

    @staticmethod
    def OL_visualization_basin_ensemble():
        # from src_FlowControl.DA_GRACE import DA_GRACE
        from src_GHM.EnumType import init_mode
        """
        this is to visualize open loop model spread for each vertical compartment
        """
        RDA.set_box()

        '''configuration'''
        demo = DA_GRACE(case=RDA.case, setting_dir=RDA.activate(), ens=RDA.ens)

        demo.configure_time(begin_time=RDA.resume_begin_time, end_time=RDA.resume_end_time)
        demo.configure_area(box=RDA.box, basin=RDA.basin)
        demo.generate_settings(mode=init_mode.resume)

        '''generate/save the figure'''
        fig_postfix = '0'
        demo.visualize_signal(fig_path=RDA.figure_output, fig_postfix=fig_postfix, file_postfix='OL',
                              data_dir=RDA.res_output)
        pass

    @staticmethod
    def DA_visualization_basin_DA(signal='TWS', allow_pop_up=False):
        from src_DA.Analysis import Postprocessing_basin
        import pygmt
        import h5py
        from src_GHM.GeoMathKit import GeoMathKit
        import json
        from pathlib import Path
        import numpy as np
        from src_GHM.EnumType import init_mode
        # from src_FlowControl.DA_GRACE import DA_GRACE

        RDA.set_box()

        case = RDA.case
        setting_dir = RDA.activate()
        ens = RDA.ens
        box = RDA.box
        basin = RDA.basin
        data_dir = RDA.res_output
        figure_output = RDA.figure_output

        warm_begin_time = RDA.warm_begin_time
        warm_end_time = RDA.warm_end_time

        '''load OL result'''
        demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
        mode = init_mode.warm
        begin_time = warm_begin_time
        end_time = warm_end_time

        demo.configure_time(begin_time=begin_time, end_time=end_time)
        demo.configure_area(box=box, basin=basin)
        demo.generate_settings(mode=mode)

        file_postfix = 'OL'
        pp = Postprocessing_basin(ens=ens, case=case, basin=basin,
                                  date_begin=begin_time,
                                  date_end=end_time)
        # states_OL = pp.get_states(post_fix=file_postfix, dir=demo._outdir2)
        states_OL = pp.load_states(prefix=(file_postfix + '_' + basin), load_dir=data_dir)
        file_postfix = 'DA'
        states_DA = pp.load_states(prefix=(file_postfix + '_' + basin), load_dir=data_dir)

        '''load GRACE'''
        # dp_dir = Path(setting_dir) / 'DA_setting.json'
        # dp4 = json.load(open(dp_dir, 'r'))
        # GRACE = pp.get_GRACE(obs_dir=dp4['obs']['dir'])
        GRACE = pp.load_GRACE(prefix=(file_postfix + '_' + basin), load_dir=data_dir)

        OL_time = states_OL['time']
        DA_time = states_DA['time']
        GR_time = GRACE['time']

        basin_num = len(list(states_DA.keys())) - 1

        '''plot figure'''
        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET="0p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='10p,5',
                     FONT_LABEL='10p,5', MAP_TICK_LENGTH='7p')

        i = 0
        offset = 12
        height = 4.6

        keys = list(GRACE['ens_mean'].keys())

        for j in range(len(keys)):
            i += 1
            basin_id = 'basin_%s' % j
            GRACE_ens_mean = GRACE['ens_mean'][basin_id]
            GRACE_original = GRACE['original'][basin_id]
            OL = states_OL[basin_id][signal][0]
            OL_ens_mean = np.mean(np.array(list(states_OL[basin_id][signal].values()))[1:, ], axis=0)
            DA_ens_mean = np.mean(np.array(list(states_DA[basin_id][signal].values()))[1:, ], axis=0)

            values = [GRACE_ens_mean, OL, OL_ens_mean, DA_ens_mean]
            vvmin = []
            vvmax = []
            for vv in values:
                vvmin.append(np.min(vv[5:]))
                vvmax.append(np.max(vv[5:]))

            vmin, vmax = min(vvmin), min(vvmax)
            dmin = vmin - (vmax - vmin) * 0.1
            dmax = vmax + (vmax - vmin) * 0.15
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            if len(OL_time) > 365 * 4:
                fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa2f1g",
                                   'ya%df%dg+lwater [mm]' % (sp_2, sp_1)])
            else:
                fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa1f0.5g",
                                   'ya%df%dg+lwater [mm]' % (sp_2, sp_1)])

            # fig.plot(x=OL_time, y=OL, pen="0.5p,blue,-", label='%s' % ('OL_unperturbed'), transparency=30)
            fig.plot(x=OL_time, y=OL_ens_mean, pen="1.0p,blue,-", label='%s' % ('OL'), transparency=30)
            # fig.plot(x=OL_time, y=OL, pen="0.5p,grey,-", label='%s' % ('OL'), transparency=30)
            # fig.plot(x=OL_time, y=OL_ens_mean, pen="0.5p,red", label='%s' % ('OL_ens_mean'), transparency=30)
            fig.plot(x=DA_time, y=DA_ens_mean, pen="1.0p,green", label='%s' % ('DA'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_ens_mean, pen="0.5p,black", label='%s' % ('GRACE_ens_mean'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_original, pen="0.5p,purple,--.", label='%s' % ('GRACE_original'),
            #          transparency=30)
            fig.plot(x=GR_time, y=GRACE_original, style="c.2c", fill="black", label='%s' % ('GRACE'), transparency=30)

            # fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.legend(position='jBL')

            sf = '%sc' % ((offset - 1) * height)
            if i % offset == 0:
                fig.shift_origin(yshift=sf, xshift='14c')
                continue

            fig.shift_origin(yshift='-%sc' % height)

            pass

        fig_postfix = '0'
        fig.savefig(str(Path(figure_output) / ('DA_%s_%s_%s.pdf' % (basin, signal, fig_postfix))))
        fig.savefig(str(Path(figure_output) / ('DA_%s_%s_%s.png' % (basin, signal, fig_postfix))))

        if allow_pop_up:
            fig.show()
        pass

    @staticmethod
    def DA_visulization_2Dmap(signal='trend', allow_pop_up=False, GRACE_res=0.5, upscale_res=0.5):
        from src_auxiliary.ts import ts
        from src_auxiliary.upscaling import upscaling
        from datetime import datetime
        from src_GHM.GeoMathKit import GeoMathKit
        import geopandas as gpd
        import h5py
        import numpy as np
        import json
        import os
        import pygmt

        RDA.set_box()

        '''load shp'''
        gdf = gpd.read_file(rc.resolve_shp(RDA.shp_path))

        # signal = 'annual'

        '''load DA'''
        hf = h5py.File(Path(RDA.res_output) / ('monthly_mean_TWS_%s_DA.h5' % RDA.basin), 'r')

        time = []
        vv = []

        for x, v in hf.items():
            m = x.split('-')
            n = datetime(year=int(m[0]), month=int(m[1]), day=15)
            time.append(GeoMathKit.year_fraction(n))
            vv.append(v[:])
            pass

        tsd = ts().set_period(time=np.array(time)).setDecomposition()
        res_DA = tsd.getSignal(obs=vv)

        '''load OL'''
        hf = h5py.File(Path(RDA.res_output) / ('monthly_mean_TWS_%s_uOL.h5' % RDA.basin), 'r')

        time = []
        vv = []

        for x, v in hf.items():
            m = x.split('-')
            n = datetime(year=int(m[0]), month=int(m[1]), day=15)
            time.append(GeoMathKit.year_fraction(n))
            vv.append(v[:])
            pass

        tsd = ts().set_period(time=np.array(time)).setDecomposition()
        res_OL = tsd.getSignal(obs=vv)

        '''load GRACE'''
        dp_dir = Path(RDA.activate()) / 'DA_setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        hf = h5py.File(Path(dp4['obs']["GRACE"]['preprocess_res']) / ('%s_gridded_signal.hdf5' % RDA.basin), 'r')
        gr_time = hf['time_epoch'][:].astype(str)
        gr_data = hf['tws'][:]
        time = []

        for i in range(len(gr_time)):
            m = gr_time[i].split('-')
            if len(m) == 2: m.append('15')
            n = datetime(year=int(m[0]), month=int(m[1]), day=int(m[2]))
            time.append(GeoMathKit.year_fraction(n))

        tsd = ts().set_period(time=np.array(time)).setDecomposition()
        res_GRACE = tsd.getSignal(obs=gr_data)
        xx = res_GRACE[signal]
        if signal != 'trend':
            xx = xx[0]

        '''1d --> 2d map'''
        dp_dir = Path(RDA.activate()) / 'setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        sh = dp4['input']['mask_fn']
        us = upscaling(basin=RDA.basin)
        us.configure_model_state(globalmask_fn=rc.crop_dir(sh, RDA.case) / 'mask/mask_global.h5', upscaled_res=upscale_res,
                                 modelres=0.1)
        us.configure_GRACE(res=GRACE_res)
        # us.downscale_model(model_state=1, GRACEres=0.5, modelres=0.1)
        res_GRACE = us.get2D_GRACE(GRACE_obs=xx)

        xx = res_DA[signal]
        if signal != 'trend':
            xx = xx[0]
        res_DA = us.get2D_model_state(model_state=xx)
        '''mask'''
        res_DA[us.bshp_mask == False] = np.nan

        xx = res_OL[signal]
        if signal != 'trend':
            xx = xx[0]
        res_OL = us.get2D_model_state(model_state=xx)
        '''mask'''
        res_OL[us.bshp_mask == False] = np.nan

        '''plot figure'''
        fig = pygmt.Figure()
        pygmt.config(COLOR_NAN='white')
        pygmt.config(FONT_TITLE="19p,5", MAP_TITLE_OFFSET="-0.2p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                     FONT_LABEL='17p,5', MAP_TICK_LENGTH='7p')
        #
        nan_mask = (1 - np.isnan(res_GRACE)).astype(bool)
        vmax = np.nanmax(res_DA[nan_mask]) * 0.8
        vmin = np.nanmin(res_DA[nan_mask]) * 0.8

        if signal == 'trend':
            pygmt.makecpt(cmap='polar+h0', series=[vmin, vmax], background='o')
        else:
            pygmt.makecpt(cmap='jet', series=[0, vmax], background='o')

        '''for GRACE visualization'''
        res = GRACE_res
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        region = [min(lon), max(lon), min(lat), max(lat)]
        lon, lat = np.meshgrid(lon, lat)

        GRACE = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_GRACE.flatten(),
                              spacing=(res, res), region=region)

        '''for OL and DA visualization'''
        res = upscale_res
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        region = [min(lon), max(lon), min(lat), max(lat)]
        lon, lat = np.meshgrid(lon, lat)

        DA = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_DA.flatten(),
                           spacing=(res, res), region=region)

        OL = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_OL.flatten(),
                           spacing=(res, res), region=region)

        '''start drawing'''
        region = [RDA.box[2] - 2, RDA.box[3] + 2, RDA.box[1] - 2, RDA.box[0] + 2]
        ps = 'Q8c'
        offset = '-10c'
        fig.grdimage(
            grid=OL,
            cmap=True,
            frame=['xa5f5', 'ya5f5'] + ['+tOL'],
            dpi=100,
            projection=ps,
            region=region,
            interpolation='n'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")

        fig.shift_origin(xshift='10c')
        fig.grdimage(
            grid=DA,
            cmap=True,
            frame=['xa5f5', 'ya5f5'] + ['+tDA'],
            dpi=100,
            projection=ps,
            region=region,
            interpolation='n'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")
        # fig.colorbar()

        fig.shift_origin(xshift='10c')
        fig.grdimage(
            grid=GRACE,
            cmap=True,
            frame=['xa5f5', 'ya5f5'] + ['+tGRACE'],
            dpi=100,
            projection=ps,
            region=region,
            interpolation='n'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")

        '''===================================================='''
        '''filtered'''
        fig.shift_origin(xshift='-20c', yshift=offset)

        fig.grdimage(
            grid=OL,
            cmap=True,
            frame=['xa5f5', 'ya5f5'] + ['+tOL smoothed'],
            dpi=100,
            projection=ps,
            region=region,
            interpolation='l'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")

        if signal != 'trend':
            fig.colorbar(frame='af+l%s amplitude [mm]' % signal, position="JBC+w13c/0.45c+h+o10c/1.2c")
        else:
            fig.colorbar(frame='af+l%s [mm/yr]' % signal, position="JBC+w13c/0.45c+h+o10c/1.2c")
            pass

        fig.shift_origin(xshift='10c')
        fig.grdimage(
            grid=DA,
            cmap=True,
            frame=['xa5f5', 'ya5f5'] + ['+tDA, smoothed'],
            dpi=100,
            projection=ps,
            region=region,
            interpolation='l'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")
        # fig.colorbar()

        fig.shift_origin(xshift='10c')
        fig.grdimage(
            grid=GRACE,
            cmap=True,
            frame=['xa5f5', 'ya5f5'] + ['+tGRACE, smoothed'],
            dpi=100,
            projection=ps,
            region=region,
            interpolation='l'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")

        fig_postfix = '0'
        figure_output = RDA.figure_output
        fig.savefig(str(Path(figure_output) / ('map2D_%s_%s_%s.pdf' % (RDA.basin, signal, fig_postfix))))
        fig.savefig(str(Path(figure_output) / ('map2D_%s_%s_%s.png' % (RDA.basin, signal, fig_postfix))))

        if allow_pop_up:
            fig.show()

        pass


def demo_prepare_GRACE():
    RDA.prepare_GRACE()
    pass


def demo_prepare_forcing():
    RDA.prepare_Forcing()
    pass


def demo_global_run_complete(prepare=True):
    """
    This is a complete processing chain to deal with global data assimilation for each tile.
    """
    from mpi4py import MPI

    '''with MPI to parallelize the computation'''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    '''basic configuration of the tile of interest'''
    if rank == 0:
        RDA.set_box()  # to avoid conflict between threads
    comm.barrier()
    RDA.set_box()

    '''single_run (cold, warmup) to obtain reliable initial states, 10 years running for assuring accuracy'''
    if rank == 0:
        # RDA.single_run()
        pass

    comm.barrier()

    '''open loop running to obtain ensemble of initializations, and more importantly to obtain temporal mean'''
    RDA.OL_run(skip_perturbation=prepare)

    comm.barrier()

    '''carry out data assimilation'''
    RDA.DA_run(skip_obs_perturbation=prepare)

    pass


def demo_global_run_only_DA(prepare=False):
    """
    This is a complete processing chain to deal with global data assimilation for each tile.
    """
    # prepare = False

    '''basic configuration of the tile of interest'''
    RDA.set_box()

    '''carry out data assimilation'''
    RDA.DA_run(skip_obs_perturbation=prepare)

    pass


def demo_DA_visualization():
    RDA.DA_visualization_basin_ensemble()
    RDA.DA_visualization_basin_DA()
    RDA.DA_visulization_2Dmap(signal='trend')
    RDA.DA_visulization_2Dmap(signal='annual')
    pass


if __name__ == '__main__':
    '''single threads for preparing data'''
    RDA.prepare_Forcing()
    # RDA.single_run()

    '''multiple threads'''
    # demo_global_run_complete(prepare=False)
    # demo_global_run_only_DA(prepare=True)

    '''single thread for plotting'''
    # demo_DA_visualization()
