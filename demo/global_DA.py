import sys

sys.path.append('../')

from pathlib import Path

res_output = '/work/data_for_w3/w3ra/res'
# figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
figure_output = '/work/data_for_w3/w3ra/figure'


class GDA:
    setting_dir = '../settings/Ucloud_DA'
    ens = 30

    case = 'case_study_DA'
    box = [50.5, 42, 8.5, 29.5]
    basin = 'tile0'

    cold_begin_time = '2000-01-01'
    cold_end_time = '2010-01-31'

    warm_begin_time = '2000-01-01'
    warm_end_time = '2023-05-31'

    resume_begin_time = '2002-03-31'
    resume_end_time = '2023-05-31'

    pass

    @staticmethod
    def global_preparation_1():

        """preparation for GRACE and forcing fields for global tiles"""
        from src_GRACE.prepare_GRACE import GRACE_global_preparation
        gr = GRACE_global_preparation().configure_global_mask()
        gr.configure_global_shp()
        # gr.basin_TWS(month_begin='2002-04', month_end='2023-05')
        # gr.basin_COV(month_begin='2002-04', month_end='2023-05')
        gr.grid_TWS(month_begin='2002-04', month_end='2023-05')
        pass

    @staticmethod
    def global_preparation_2():
        from src_FlowControl.SingleModel import SingleModel
        from src_hydro.EnumType import init_mode
        '''pre-process input forcing field'''
        # GDA.setting_dir = '../settings/single_run'

        '''configuration'''
        for tile_ID in range(69, 71):

            try:
                GDA.set_tile(tile_ID=tile_ID, create_folder=False)
            except Exception:
                # print('Not exists')
                continue

            demo = SingleModel(case=GDA.case, setting_dir=GDA.setting_dir)

            demo.configure_time(begin_time=GDA.cold_begin_time, end_time=GDA.resume_end_time)

            demo.configure_area(box=GDA.box, basin=GDA.basin)

            '''change the sub-setting files according to the main setting'''
            demo.generate_settings(mode=init_mode.cold)

            '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
            demo.preprocess()

        pass

    @staticmethod
    def set_tile(tile_ID=1, create_folder=True):
        import geopandas as gpd

        if GDA.case == 'GDA_%s' % tile_ID:
            return

        GDA.case = 'GDA_%s' % tile_ID
        GDA.basin = 'Tile%s' % tile_ID

        gf = gpd.read_file('../data/basin/shp/global_shp_new/Tile%s_subbasins.shp' % tile_ID)
        GDA.box = list(gf.total_bounds[[3, 1, 0, 2]])  # latmax, latmin, lonmin, lonmax

        '''slightly expand the box area'''
        if GDA.box[0] + 0.2 > 90:
            pass
        else:
            GDA.box[0] += 0.2

        if GDA.box[1] - 0.2 < -90:
            pass
        else:
            GDA.box[1] -= 0.2

        if GDA.box[2] - 0.2 < -180:
            pass
        else:
            GDA.box[2] -= 0.2

        if GDA.box[3] + 0.2 > 180:
            pass
        else:
            GDA.box[3] += 0.2

        dn1 = Path(figure_output) / GDA.case
        GDA.figure_output = str(dn1)
        dn2 = Path(res_output) / GDA.case
        GDA.res_output = str(dn2)

        if create_folder:
            '''generate case-specified folder to store figures'''
            if not dn1.is_dir():
                dn1.mkdir()
            '''generate case-specified folder to store res'''
            if not dn2.is_dir():
                dn2.mkdir()

        pass

    @staticmethod
    def single_run(tile_ID):
        from src_FlowControl.SingleModel import SingleModel
        from src_hydro.EnumType import init_mode
        from datetime import datetime, timedelta

        '''configuration'''
        GDA.set_tile(tile_ID=tile_ID)
        demo = SingleModel(case=GDA.case, setting_dir=GDA.setting_dir)
        demo.configure_time(begin_time=GDA.cold_begin_time, end_time=GDA.cold_end_time)
        demo.configure_area(box=GDA.box, basin=GDA.basin)

        '''change the sub-setting files according to the main setting'''
        demo.generate_settings(mode=init_mode.cold)

        '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
        # demo.preprocess()

        # '''run the model'''
        demo.model_run()

        '''create ini states?'''
        modifydate = (datetime.strptime(GDA.warm_begin_time, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y%m%d')
        demo.create_ini_states(mode=init_mode.cold, modifydate=modifydate)

        demo.extract_signal()

        pass

    @staticmethod
    def single_run_visualization(tile_ID):
        from src_FlowControl.SingleModel import SingleModel
        from src_hydro.EnumType import init_mode
        from datetime import datetime, timedelta

        '''configuration'''
        GDA.set_tile(tile_ID=tile_ID)
        demo = SingleModel(case=GDA.case, setting_dir=GDA.setting_dir)
        demo.configure_time(begin_time=GDA.cold_begin_time, end_time=GDA.cold_end_time)
        demo.configure_area(box=GDA.box, basin=GDA.basin)

        '''change the sub-setting files according to the main setting'''
        demo.generate_settings(mode=init_mode.cold)

        '''generate/save the figure'''
        fig_postfix = '0'
        # demo.visualize_signal(fig_path=GDA.figure_output, fig_postfix=fig_postfix)
        demo.visualize_comparison_GRACE(fig_path=GDA.figure_output, fig_postfix=fig_postfix)

        pass

    @staticmethod
    def OL_run(tile_ID, prepare=True):
        from src_FlowControl.DA_GRACE import DA_GRACE
        from src_hydro.EnumType import init_mode
        from mpi4py import MPI
        import sys
        temp = sys.stdout

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (GDA.ens + 1), 'Not enough threads for parallelization!'

        GDA.set_tile(tile_ID=tile_ID)

        '''configuration'''
        demo = DA_GRACE(case=GDA.case, setting_dir=GDA.setting_dir, ens=GDA.ens)

        demo.configure_time(begin_time=GDA.warm_begin_time, end_time=GDA.warm_end_time)
        demo.configure_area(box=GDA.box, basin=GDA.basin)

        if rank == 0:
            '''change the sub-setting files according to the main setting'''
            demo.generate_settings(mode=init_mode.warm)

            # '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
            # demo.preprocess()

            '''generate the perturbation for the data'''
            if prepare:
                demo.perturbation()

        comm.barrier()

        '''run the model'''
        demo.model_run()

        '''Analysis the model output/states'''
        demo.extract_signal(postfix='OL')

        comm.barrier()

        if rank == 0:
            demo.post_processing(file_postfix='OL', save_dir=GDA.res_output)

            '''change the time to get prepared for DA experiment'''
            demo = DA_GRACE(case=GDA.case, setting_dir=GDA.setting_dir, ens=GDA.ens)

            demo.configure_time(begin_time=GDA.resume_begin_time, end_time=GDA.resume_end_time)
            demo.configure_area(box=GDA.box, basin=GDA.basin)
            demo.generate_settings(mode=init_mode.resume)

            '''gather the mean value of the open loop: ensemble mean of the temporal mean'''
            demo.gather_OLmean(post_fix='OL')

            '''get states samples for creating NaN mask'''
            demo.get_states_sample_for_mask()

        '''Job finished'''
        sys.stdout = temp
        print('job finished')

        pass

    @staticmethod
    def DA_run(tile_ID, prepare=True):
        from src_FlowControl.DA_GRACE import DA_GRACE
        from src_hydro.EnumType import init_mode
        from mpi4py import MPI
        import sys
        temp = sys.stdout

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (GDA.ens + 1), 'Not enough threads for parallelization!'

        GDA.set_tile(tile_ID=tile_ID)

        '''configuration'''
        demo = DA_GRACE(case=GDA.case, setting_dir=GDA.setting_dir, ens=GDA.ens)

        demo.configure_time(begin_time=GDA.resume_begin_time, end_time=GDA.resume_end_time)
        demo.configure_area(box=GDA.box, basin=GDA.basin)

        if rank == 0:
            '''change the sub-setting files according to the main setting'''
            demo.generate_settings(mode=init_mode.resume)

            # '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
            # demo.preprocess()

            '''generate the perturbation for the data'''
            if prepare:
                '''prepare the GRACE observation over region of interest'''
                '''Calculation of the COV is time-consuming, so we do not suggest to repeatedly execute this command '''
                # demo.GRACE_obs_preprocess()

                '''perturb the GRACE obs for later use'''
                demo.generate_perturbed_GRACE_obs()

            '''prepare the design matrix'''
            demo.prepare_design_matrix()

        comm.barrier()

        '''run the model'''
        demo.run_DA(rank=rank)

        '''Analysis the model output/states'''
        demo.extract_signal(postfix='DA')

        comm.barrier()

        if rank == 0:
            demo.post_processing(file_postfix='DA', save_dir=GDA.res_output, isGRACE=True)

        '''Job finished'''
        sys.stdout = temp
        print('job finished')

        pass

    @staticmethod
    def DA_visualization_basin_ensemble(tile_ID):
        from src_FlowControl.DA_GRACE import DA_GRACE
        from src_hydro.EnumType import init_mode

        GDA.set_tile(tile_ID=tile_ID)

        '''configuration'''
        demo = DA_GRACE(case=GDA.case, setting_dir=GDA.setting_dir, ens=GDA.ens)

        demo.configure_time(begin_time=GDA.resume_begin_time, end_time=GDA.resume_end_time)
        demo.configure_area(box=GDA.box, basin=GDA.basin)
        demo.generate_settings(mode=init_mode.resume)

        '''generate/save the figure'''
        fig_postfix = '0'
        demo.visualize_signal(fig_path=GDA.figure_output, fig_postfix=fig_postfix, file_postfix='DA',
                              data_dir=GDA.res_output)
        pass

    @staticmethod
    def DA_visualization_basin_DA(tile_ID, signal='TWS'):
        from src_DA.Analysis import Postprocessing_basin
        import pygmt
        import h5py
        from src_hydro.GeoMathKit import GeoMathKit
        import json
        from pathlib import Path
        import numpy as np
        from src_hydro.EnumType import init_mode
        from src_FlowControl.DA_GRACE import DA_GRACE

        GDA.set_tile(tile_ID=tile_ID)

        case = GDA.case
        setting_dir = GDA.setting_dir
        ens = GDA.ens
        box = GDA.box
        basin = GDA.basin
        data_dir = GDA.res_output
        figure_output = GDA.figure_output

        warm_begin_time = GDA.warm_begin_time
        warm_end_time = GDA.warm_end_time

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

        i = 0
        offset = 12
        height = 4.6

        keys = list(GRACE['ens_mean'].keys())
        keys

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
            dmax = vmax + (vmax - vmin) * 0.1
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                        frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa2f1",
                               'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

            # fig.plot(x=OL_time, y=OL, pen="0.5p,blue,-", label='%s' % ('OL_unperturbed'), transparency=30)
            fig.plot(x=OL_time, y=OL, pen="0.5p,grey,-", label='%s' % ('OL'), transparency=30)
            # fig.plot(x=OL_time, y=OL_ens_mean, pen="0.5p,red", label='%s' % ('OL_ens_mean'), transparency=30)
            fig.plot(x=DA_time, y=DA_ens_mean, pen="0.5p,green", label='%s' % ('DA'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_ens_mean, pen="0.5p,black", label='%s' % ('GRACE_ens_mean'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_original, pen="0.5p,purple,--.", label='%s' % ('GRACE_original'),
            #          transparency=30)
            fig.plot(x=GR_time, y=GRACE_original, style="c.1c", fill="black", label='%s' % ('GRACE'), transparency=30)

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
        # fig.show()
        pass

    @staticmethod
    def DA_visulization_2Dmap(tile_ID, signal='trend'):
        from src_auxiliary.ts import ts
        from src_auxiliary.upscaling import upscaling
        from datetime import datetime
        from src_hydro.GeoMathKit import GeoMathKit
        import geopandas as gpd
        import h5py
        import numpy as np
        import json
        import os
        import pygmt

        GDA.set_tile(tile_ID=tile_ID)

        '''load shp'''
        '''search for the shape file'''
        pp = Path('../data/basin/shp/')
        target = []
        for root, dirs, files in os.walk(pp):
            for file_name in files:
                if (str(file_name).split('_')[0] == GDA.basin) and file_name.endswith('.shp') and (
                        'subbasins' in file_name):
                    # if (GDA.basin in file_name) and file_name.endswith('.shp') and ('subbasins' in file_name):
                    target.append(os.path.join(root, file_name))
        assert len(target) == 1, target

        gdf = gpd.read_file(str(target[0]))

        # signal = 'annual'

        '''load DA'''
        hf = h5py.File(Path(GDA.res_output) / ('monthly_mean_TWS_%s_DA.h5' % GDA.basin), 'r')

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
        hf = h5py.File(Path(GDA.res_output) / ('monthly_mean_TWS_%s_uOL.h5' % GDA.basin), 'r')

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
        dp_dir = Path(GDA.setting_dir) / 'DA_setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        hf = h5py.File(Path(dp4['obs']["GRACE"]['preprocess_res']) / ('%s_gridded_signal.hdf5' % GDA.basin), 'r')
        gr_time = hf['time_epoch'][:].astype(str)
        gr_data = hf['tws'][:]
        time = []

        for i in range(len(gr_time)):
            m = gr_time[i].split('-')
            n = datetime(year=int(m[0]), month=int(m[1]), day=int(m[2]))
            time.append(GeoMathKit.year_fraction(n))

        tsd = ts().set_period(time=np.array(time)).setDecomposition()
        res_GRACE = tsd.getSignal(obs=gr_data)
        xx = res_GRACE[signal]
        if signal != 'trend':
            xx = xx[0]

        '''1d --> 2d map'''
        dp_dir = Path(GDA.setting_dir) / 'setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        sh = dp4['input']['mask_fn']
        us = upscaling(basin=GDA.basin)
        us.configure_model_state(globalmask_fn=Path(sh) / GDA.case / 'mask/mask_global.h5')
        us.configure_GRACE()
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
        pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
        pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
        #
        nan_mask = (1 - np.isnan(res_GRACE)).astype(bool)
        vmax = np.nanmax(res_DA[nan_mask]) * 0.8
        vmin = np.nanmin(res_DA[nan_mask]) * 0.8

        if signal == 'trend':
            pygmt.makecpt(cmap='polar+h0', series=[vmin, vmax], background='o')
        else:
            pygmt.makecpt(cmap='jet', series=[0, vmax], background='o')

        res = 0.5
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        region = [min(lon), max(lon), min(lat), max(lat)]
        lon, lat = np.meshgrid(lon, lat)

        DA = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_DA.flatten(),
                           spacing=(res, res), region=region)

        OL = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_OL.flatten(),
                           spacing=(res, res), region=region)

        GRACE = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_GRACE.flatten(),
                              spacing=(res, res), region=region)

        region = [GDA.box[2] - 2, GDA.box[3] + 2, GDA.box[1] - 2, GDA.box[0] + 2]
        ps = 'Q8c'
        offset = '-8.5c'
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
            # interpolation='n'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")

        if signal != 'trend':
            fig.colorbar(frame='a20f10+l%s amplitude [mm]' % signal, position="JBC+w13c/0.45c+h+o10c/1.2c")
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
            # interpolation='n'
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
            # interpolation='n'
        )

        fig.coast(shorelines="1/0.2p", region=region, projection=ps)

        fig.plot(data=gdf.boundary, pen="1p,black")

        fig_postfix = '0'
        figure_output = GDA.figure_output
        fig.savefig(str(Path(figure_output) / ('map2D_%s_%s_%s.pdf' % (GDA.basin, signal, fig_postfix))))
        fig.savefig(str(Path(figure_output) / ('map2D_%s_%s_%s.png' % (GDA.basin, signal, fig_postfix))))

        pass


def demo1():
    GDA.global_preparation_1()
    pass


def demo2():
    GDA.global_preparation_2()
    pass


def demo3():
    # GDA.setting_dir = '../settings/single_run'
    GDA.single_run(tile_ID=81)
    pass


def demo4():
    # GDA.setting_dir = '../settings/single_run'
    GDA.single_run_visualization(tile_ID=81)
    pass


def demo5():
    GDA.OL_run(tile_ID=81, prepare=True)
    pass


def demo6():
    GDA.DA_run(tile_ID=81, prepare=False)
    pass


def demo_global_run_complete(tile_ID=80):
    """
    This is a complete processing chain to deal with global data assimilation for each tile.
    """
    from mpi4py import MPI

    prepare = True

    '''with MPI to parallelize the computation'''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    '''basic configuration of the tile of interest'''
    if rank == 0:
        GDA.set_tile(tile_ID=tile_ID)  # to avoid conflict between threads
    comm.barrier()
    GDA.set_tile(tile_ID=tile_ID)

    '''single_run (cold, warmup) to obtain reliable initial states, 10 years running for assuring accuracy'''
    if rank == 0:
        GDA.single_run(tile_ID=tile_ID)
        pass

    comm.barrier()

    '''open loop running to obtain ensemble of initializations, and more importantly to obtain temporal mean'''
    GDA.OL_run(tile_ID=tile_ID, prepare=prepare)

    comm.barrier()

    '''carry out data assimilation'''
    GDA.DA_run(tile_ID=tile_ID, prepare=prepare)

    pass


def demo_global_run_only_DA(tile_ID=80, prepare=False):
    """
    This is a complete processing chain to deal with global data assimilation for each tile.
    """
    # prepare = False

    '''basic configuration of the tile of interest'''
    GDA.set_tile(
        tile_ID=tile_ID
    )

    '''carry out data assimilation'''
    GDA.DA_run(tile_ID=tile_ID, prepare=prepare)

    pass


def demo_DA_visualization(tile_ID=80):
    GDA.DA_visualization_basin_ensemble(tile_ID=tile_ID)
    GDA.DA_visualization_basin_DA(tile_ID=tile_ID)
    GDA.DA_visulization_2Dmap(tile_ID=tile_ID, signal='trend')
    GDA.DA_visulization_2Dmap(tile_ID=tile_ID, signal='annual')
    pass


if __name__ == '__main__':
    # demo2()
    # for tile_ID in [33]:
    #     demo_global_run_only_DA(prepare=True, tile_ID=tile_ID)
    for tile_ID in [3, 4, 5, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]:
        demo_global_run_complete(tile_ID=tile_ID)
    # demo_DA_visualization(tile_ID=tile_ID)

    # for tile_ID in [20,23,24,25,26,27,28,29,30,33,34,35]:
    #     demo_global_run_complete(tile_ID=tile_ID)

    # for tile_ID in [36, 38, 39, 40, 41, 42,43,44, 45,48,49, 50]:
    #     demo_global_run_complete(tile_ID=tile_ID)

    # for tile_ID in [52, 53,54,55,56,57,58,59,64,65,66,67,68,69]:
    #     demo_global_run_complete(tile_ID=tile_ID)

    # for tile_ID in [70,71,72,73,80,81,82,83,84,85,87,88,89,95]:
    #     demo_global_run_complete(tile_ID=tile_ID)

    # for tile_ID in [96,99,100,103,104,110,111,119,120]:
    # demo_global_run_complete(tile_ID=tile_ID)