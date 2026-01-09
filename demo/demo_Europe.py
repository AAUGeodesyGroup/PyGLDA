import os
import sys

sys.path.append('../')
from pathlib import Path
from datetime import datetime, timedelta
from src_FlowControl.Regional_DA import RDA
import src_FlowControl.Regional_DA as mm

'''This is a demo to show the complete process of DA. In this demo, we select three examples as below: 1. Regional 
basin-scale (three major sub-basins are defined) DA for Danube river basin. More configuration on this experiment 
refers to our paper. 2. Regional grid-scale DA for Danube river basin. More configuration on this experiment refers 
to our paper. 3. Global grid-scale DA, but only one box is shown here as example.'''

'''To run DA, a parallel call of this script has to be implemented, which is therefore meant to run on the cluster. 
We make use of MPI to do the parallelization, so please type the command to execute the code in the terminal as below 
(where the number 31= ensemble size+1)'''
'''mpiexec -n 31 python -u demo_3.py'''

'''This path is set for Ucloud'''
'''Define where to store the key results of DA'''
mm.res_output = '/work/data_for_w3/w3ra/res'
mm.figure_output = '/work/data_for_w3/w3ra/figure'
'''Define the main path of external data'''
RDA.external_data_path = '/work/data_for_w3'

'''This path is set for running locally'''
'''Define where to store the key results of DA'''
# mm.res_output = '/work/data_for_w3/w3ra/res'
# mm.figure_output = '/work/data_for_w3/w3ra/figure'
'''Define the main path of external data'''
# RDA.external_data_path = '/work/data_for_w3'

'''Define where to load the necessary setting files'''
RDA.setting_dir = '../settings/Europe/GRACE_mascon_3deg'

'''Define the size of ensemble to run for DA'''
RDA.ens = 30

'''Define the name of your case study'''
RDA.case = 'GRACEM3D'

'''Define the shape file of basin and its sub-basin to be assimilated with GRACE'''
RDA.basin = 'Europe'
RDA.shp_path = '../data/basin/shp/Europe/Grid_3/Europe_subbasins.shp'

'''this is useless, as the area will be automatically calculated from the shape file'''
RDA.box = [50.5, 42, 8.5, 29.5]

'''for spin-up'''
RDA.cold_begin_time = '2000-01-01'
RDA.cold_end_time = '2001-12-31'

'''for open-loop'''
RDA.warm_begin_time = '2002-01-01'
RDA.warm_end_time = '2017-12-31'

'''for data assimilation'''
RDA.resume_begin_time = '2002-03-01'
RDA.resume_end_time = '2017-12-31'

RDA.isSet = False


def demo_complete_DA(skipModelPerturb=False, skipObsPerturb=False, skipSR=False):
    """
    This is a complete processing chain to deal with global data assimilation for each tile.
    """
    from mpi4py import MPI

    '''with MPI to parallelize the computation'''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    '''basic configuration of the tile of interest'''
    comm.barrier()
    if rank == 0:
        RDA.set_box()  # to avoid conflict between threads
    comm.barrier()
    RDA.set_box()

    '''single_run (cold, warmup) to obtain reliable initial states, 10 years running for assuring accuracy'''
    if rank == 0:
        if not skipSR:
            RDA.single_run()
            pass

    comm.barrier()

    '''open loop running to obtain ensemble of initializations, and more importantly to obtain temporal mean to be 
    removed from the GRACE observations'''
    RDA.OL_run(skip_perturbation=skipModelPerturb)

    comm.barrier()

    '''carry out data assimilation'''
    RDA.DA_run(skip_obs_perturbation=skipObsPerturb)

    comm.barrier()

    pass

def demo_OL(skipModelPerturb=False, skipSR=False):
    """
    This is a complete processing chain to deal with global data assimilation for each tile.
    """
    from mpi4py import MPI

    '''with MPI to parallelize the computation'''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    '''basic configuration of the tile of interest'''
    comm.barrier()
    if rank == 0:
        RDA.set_box()  # to avoid conflict between threads
    comm.barrier()
    RDA.set_box()

    '''single_run (cold, warmup) to obtain reliable initial states, 10 years running for assuring accuracy'''
    if rank == 0:
        if not skipSR:
            RDA.single_run()
            pass

    comm.barrier()

    '''open loop running to obtain ensemble of initializations, and more importantly to obtain temporal mean to be 
    removed from the GRACE observations'''
    RDA.OL_run(skip_perturbation=skipModelPerturb)

    pass

def demo_only_DA(skip_obs_perturbation=True):
    """
    Only DA is performed, which assumes that SR and OL have been done already.
    SKip the GRACE observation and perturbation ?
    """
    # prepare = False

    '''basic configuration of the tile of interest'''
    RDA.set_box()

    '''carry out data assimilation'''
    RDA.DA_run(skip_obs_perturbation=skip_obs_perturbation)

    pass


def demo_prepare_GRACE(isDiagonal=False):
    dir_in = Path(RDA.external_data_path) / 'GRACE' / 'SaGEA' / 'signal_DDK3'
    cov_dir_in = Path(RDA.external_data_path) / 'GRACE' / 'SaGEA' / 'sample_DDK3'
    dir_out = Path(RDA.external_data_path) / 'GRACE' / 'output'
    RDA.prepare_GRACE(dir_in=dir_in, dir_out=dir_out, cov_dir_in=cov_dir_in, is_diagonal=isDiagonal)

    pass


def demo_prepare_GRACE_Mascon(isDiagonal=False):
    dir_in = Path(RDA.external_data_path) / 'GRACE' / 'SaGEA' / 'signal_Mascon'
    cov_dir_in = Path(RDA.external_data_path) / 'GRACE' / 'SaGEA' / 'sample_DDK3'
    dir_out = Path(RDA.external_data_path) / 'GRACE' / 'output'
    RDA.prepare_GRACE_Mascon(dir_in=dir_in, dir_out=dir_out, cov_dir_in=cov_dir_in, is_diagonal=isDiagonal)
    pass

def demo_prepare_GRACE_Mascon_locally(isDiagonal=False):
    # dir_in = Path(RDA.external_data_path) / 'GRACE' / 'SaGEA' / 'signal_Mascon'
    # cov_dir_in = Path(RDA.external_data_path) / 'GRACE' / 'SaGEA' / 'sample_DDK3'
    # dir_out = Path(RDA.external_data_path) / 'GRACE' / 'output'

    dir_in = '/media/user/My Book/Fan/GRACE/signal_Mascon'
    cov_dir_in = '/media/user/My Book/Fan/GRACE/DDK3'
    dir_out = '/media/user/My Book/Fan/GRACE/output/Europe'

    RDA.prepare_GRACE_Mascon(dir_in=dir_in, dir_out=dir_out, cov_dir_in=cov_dir_in, is_diagonal=isDiagonal)
    pass

def demo_prepare_ESA_SING_5daily(isDiagonal=False, **kwargs):
    """preparation for GRACE and forcing fields for global tiles"""
    from src_OBS.prepare_ESA_SING import ESA_SING_5daily
    ''''''

    es = ESA_SING_5daily(basin_name=RDA.basin,
                         shp_path=RDA.shp_path)

    # ['GRACE-C-like', 'NGGM', 'MAGIC']
    # ['1e+14', '1e+15', '1e+16', '1e+17', '1e+18', '1e+19']
    # ['Grid_1', 'Grid_2', 'Grid_3', 'Grid_4', 'subbasins_3']

    filter, mission, grid = kwargs['filter'], kwargs['mission'], kwargs['grid']
    es.set_extra_info(filter=filter, mission=mission, grid_def=grid)

    # es.set_extra_info(filter='1e+16', mission='GRACE-C-like')
    # es.set_extra_info(filter='1e+16', mission='MAGIC')

    # es.set_extra_info(filter='1e+16', mission='NGGM', grid_def='Grid_1')
    # es.set_extra_info(filter='1e+16', mission='NGGM', grid_def='Grid_2')
    # es.set_extra_info(filter='1e+16', mission='NGGM', grid_def='Grid_3')
    # es.set_extra_info(filter='1e+16', mission='NGGM', grid_def='Grid_4')
    # es.set_extra_info(filter='1e+16', mission='NGGM', grid_def='subbasins_3')

    # es.set_extra_info(filter='1e+16', mission='MAGIC', grid_def='Grid_1')
    # es.set_extra_info(filter='1e+16', mission='MAGIC', grid_def='Grid_2')
    # es.set_extra_info(filter='1e+16', mission='MAGIC', grid_def='Grid_3')
    # es.set_extra_info(filter='1e+16', mission='MAGIC', grid_def='Grid_4')
    # es.set_extra_info(filter='1e+16', mission='MAGIC', grid_def='subbasins_3')

    # es.set_extra_info(filter='1e+16', mission='GRACE-C-like', grid_def='Grid_1')
    # es.set_extra_info(filter='1e+16', mission='GRACE-C-like', grid_def='Grid_2')
    # es.set_extra_info(filter='1e+16', mission='GRACE-C-like', grid_def='Grid_3')
    # es.set_extra_info(filter='1e+16', mission='GRACE-C-like', grid_def='Grid_4')
    # es.set_extra_info(filter='1e+16', mission='GRACE-C-like', grid_def='subbasins_3')

    es.generate_mask()
    t1 = RDA.warm_begin_time
    t2 = RDA.warm_end_time

    # es.basin_TWS(day_begin=t1, day_end=t2, dir_in='/media/user/My Book/Fan/ESA_SING/Brahmaputra',
    #              dir_out='/media/user/My Book/Fan/ESA_SING/TestRes')
    # es.basin_COV(day_begin=t1, day_end=t2, isDiagonal=isDiagonal, dir_in='/media/user/My Book/Fan/ESA_SING/Brahmaputra',
    #              dir_out='/media/user/My Book/Fan/ESA_SING/TestRes')
    # es.grid_TWS(day_begin=t1, day_end=t2, dir_in='/media/user/My Book/Fan/ESA_SING/globe/Global_EWHA_all',
    #             dir_out='/media/user/My Book/Fan/ESA_SING/TestRes')

    es.basin_TWS(day_begin=t1, day_end=t2, dir_in='/media/user/My Book/Fan/ESA_SING/Danube',
                 dir_out='/media/user/My Book/Fan/ESA_SING/TestRes')
    es.basin_COV(day_begin=t1, day_end=t2, isDiagonal=isDiagonal, dir_in='/media/user/My Book/Fan/ESA_SING/Danube',
                 dir_out='/media/user/My Book/Fan/ESA_SING/TestRes')
    es.grid_TWS(day_begin=t1, day_end=t2, dir_in='/media/user/My Book/Fan/ESA_SING/globe/Global_EWHA_all',
                dir_out='/media/user/My Book/Fan/ESA_SING/TestRes')

    pass


def demo_DA_visualization():
    RDA.DA_visualization_basin_ensemble()
    RDA.DA_visualization_basin_DA()
    RDA.DA_visulization_2Dmap(signal='trend')
    RDA.DA_visulization_2Dmap(signal='annual')
    pass


def save_data():
    from src_postprocessing.DataManager import dataManager_ensemble_statistic, dataManager_ensemble_member, data_var, \
        data_dim
    from src_DA.shp2mask import basin_shp_process

    basin = RDA.basin
    shp_path = RDA.shp_path
    settings = Path(RDA.setting_dir) / 'setting.json'
    global_basin_mask = basin_shp_process(res=0.1, basin_name=basin, save_dir='../data/basin/mask').shp_to_mask(
        shp_path=shp_path).mask[0]

    '''DA results'''
    dsr = dataManager_ensemble_statistic().configure(setting_fn=str(settings),
                                                     out_dir_mean='/work/data_for_w3/w3ra/save_data/DA_mean',
                                                     out_dir_std='/work/data_for_w3/w3ra/save_data/DA_std',
                                                     variable=data_var.TotalWater, dims=data_dim.one_dimension)
    dsr.reduce_datasize(global_basin_mask=global_basin_mask, save_mask='/work/data_for_w3/w3ra/save_data/Mask')

    '''====================================================================='''
    dsr.aggregation_daily(date_begin=RDA.resume_begin_time, date_end=RDA.resume_end_time)

    '''OpenLoop results'''
    dsr = dataManager_ensemble_member(ens=0).configure(setting_fn=str(settings),
                                                       out_dir='/work/data_for_w3/w3ra/save_data/OL_mean',
                                                       variable=data_var.TotalWater, dims=data_dim.one_dimension)
    dsr.reduce_datasize(global_basin_mask=global_basin_mask)

    '''====================================================================='''
    dsr.aggregation_daily(date_begin=RDA.resume_begin_time, date_end=RDA.resume_end_time)

    pass


def customize(mission: str, filter: str, grid: str, region: str):
    assert mission in ['GRACEC', 'NGGM', 'MAGIC']
    assert filter in ['1e14', '1e15', '1e16', '1e17', '1e18', '1e19']
    assert grid in ['3subbasins', '1degree', '2degree', '3degree', '4degree']
    assert region in ['Brahmaputra', 'Danube']

    RDA.setting_dir = '../settings/'
    RDA.setting_dir = str(Path(RDA.setting_dir) / mission / filter)
    RDA.case = 'flood_'+ mission + '_' + filter + '_' + grid + '_' + region

    RDA.shp_path = Path('../data/basin/shp/ESA_SING/')
    RDA.basin = region + grid
    if grid == '3subbasins':
        RDA.shp_path = RDA.shp_path / 'subbasins' / ('%s%s_subbasins.shp' % (region, grid))
    else:
        RDA.shp_path = RDA.shp_path / ('Grid_%s' % grid[0]) / ('%s%s_subbasins.shp' % (region, grid))

    pass


def demo_prepare_ESA_SING_5daily_loop():
    mission = 'NGGM'
    filter = '1e19'
    region = 'Danube'

    mission_ref = {
        'GRACEC': 'GRACE-C-like',
        'NGGM': 'NGGM',
        'MAGIC': 'MAGIC'
    }

    filter_ref = {
        '1e14': '1e+14',
        '1e15': '1e+15',
        '1e16': '1e+16',
        '1e17': '1e+17',
        '1e18': '1e+18',
        '1e19': '1e+19'
    }

    grid_ref = {
        '1degree': 'Grid_1',
        '2degree': 'Grid_2',
        '3degree': 'Grid_3',
        '4degree': 'Grid_4',
        '3subbasins': 'subbasins_3'
    }

    # for grid in ['3subbasins', '1degree', '2degree', '3degree', '4degree']:
    for grid in ['4degree']:
        customize(mission=mission, filter=filter, grid=grid, region=region)
        config = {
            'filter': filter_ref[filter],
            'mission': mission_ref[mission],
            'grid': grid_ref[grid]
        }
        if grid == '1degree':
            isDiagonal = True
        else:
            isDiagonal = False

        demo_prepare_ESA_SING_5daily(isDiagonal=isDiagonal, **config)

    pass


def demo_batch_run():
    from mpi4py import MPI

    '''with MPI to parallelize the computation'''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    missions = ['GRACEC', 'NGGM', 'MAGIC']
    filters = ['1e14', '1e15', '1e16', '1e17', '1e18', '1e19']
    grids = ['3subbasins', '1degree', '2degree', '3degree', '4degree']
    regions = ['Brahmaputra', 'Danube']

    region = 'Brahmaputra'
    # region = 'Danube'
    grids = ['1degree']
    missions=['MAGIC']
    filters =['1e17']

    customize(mission='MAGIC', filter='1e19', grid='4degree', region=region)

    for mission in missions:
        for filter in filters:
            for grid in grids:
                if rank == 0:
                    print('========= ESA-SING experiment: %s, %s, %s, %s =========' % (mission, filter, grid, region))
                    old = RDA.case
                customize(mission=mission, filter=filter, grid=grid, region=region)

                '''make sure all cases share the same open-loop perturbations'''
                comm.barrier()
                if rank == 0:
                    a = os.getcwd()
                    os.chdir(r"/work/data_for_w3/w3ra/crop_input")
                    os.rename(old, RDA.case)
                    os.chdir(a)

                comm.barrier()
                demo_complete_DA(skipModelPerturb=True, skipObsPerturb=False, skipSR=False)  # OL and DA
                RDA.isSet = False # to make a new folder for the analysis results
    pass

def demo_batch_visualization():

    missions = ['GRACEC', 'NGGM', 'MAGIC']
    filters = ['1e14', '1e15', '1e16', '1e17', '1e18', '1e19']
    grids = ['3subbasins', '1degree', '2degree', '3degree', '4degree']
    regions = ['Brahmaputra', 'Danube']

    region = 'Brahmaputra'
    # region = 'Danube'
    # missions = ['MAGIC']
    # filters = ['1e19']
    grids = ['1degree']
    customize(mission='MAGIC', filter='1e19', grid='1degree', region=region)

    for mission in missions:
        for filter in filters:
            for grid in grids:
                print('========= ESA-SING experiment: %s, %s, %s, %s =========' % (mission, filter, grid, region))

                old = RDA.case
                customize(mission=mission, filter=filter, grid=grid, region=region)
                '''make sure all cases share the same open-loop perturbations'''
                a = os.getcwd()
                os.chdir(r"/work/data_for_w3/w3ra/crop_input")
                os.rename(old, RDA.case)
                os.chdir(a)
                '''run'''
                demo_DA_visualization()
                save_data()
                '''reset'''
                RDA.isSet = False # to make a new folder for the analysis results
    pass


if __name__ == '__main__':
    '''single threads for preparing GRACE data: local'''
    # demo_prepare_ESA_SING_5daily_loop()
    # demo_prepare_GRACE_Mascon_locally(isDiagonal=True)

    '''link the setting file to the external data path: switch between local and cluster'''
    # RDA.config_external_data()

    '''preparing model data: cluster; single thread is required'''
    # RDA.prepare_Forcing()
    # RDA.single_run(skip_croping_data=True, skip_signal_extraction=True)  # only SR

    '''multiple threads'''
    # demo_OL(skipModelPerturb=False, skipSR=True)
    # demo_OL(skipModelPerturb=True, skipSR=True)
    demo_only_DA(skip_obs_perturbation=False)

    # demo_complete_DA(skipModelPerturb=True, skipObsPerturb=False, skipSR=True)  # OL and DA
    # demo_only_DA(skip_obs_perturbation=False)  # only DA

    '''batch run'''
    # demo_batch_run()
    # demo_batch_visualization()

    '''single thread for plotting'''
    # demo_DA_visualization()
    # save_data()
