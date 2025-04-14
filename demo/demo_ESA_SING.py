import sys

sys.path.append('../')
from pathlib import Path
from datetime import datetime, timedelta
from src_FlowControl.Regional_DA import RDA, res_output, figure_output

'''This is a demo to show the complete process of DA. In this demo, we select three examples as below: 1. Regional 
basin-scale (three major sub-basins are defined) DA for Danube river basin. More configuration on this experiment 
refers to our paper. 2. Regional grid-scale DA for Danube river basin. More configuration on this experiment refers 
to our paper. 3. Global grid-scale DA, but only one box is shown here as example.'''

'''To run DA, a parallel call of this script has to be implemented, which is therefore meant to run on the cluster. 
We make use of MPI to do the parallelization, so please type the command to execute the code in the terminal as below 
(where the number 31= ensemble size+1)'''
'''mpiexec -n 31 python demo_3.py'''

'''Define where to store the key results of DA'''
res_output = '/work/data_for_w3/w3ra/res'
figure_output = '/work/data_for_w3/w3ra/figure'

'''Define where to load the necessary setting files'''
# RDA.setting_dir = '../settings/MAGIC'
# RDA.setting_dir = '../settings/GRACEC'
RDA.setting_dir = '../settings/'

'''Define the size of ensemble to run for DA'''
RDA.ens = 30

'''Define the name of your case study'''
RDA.case = 'NGGM1E16_B'
# RDA.case = 'NGGM1E16_1degree'
# RDA.case = 'NGGM1E16_2degree'
# RDA.case = 'NGGM1E16_3degree'
# RDA.case = 'NGGM1E16_4degree'

# RDA.case = 'MAGIC1E16'
# RDA.case = 'MAGIC1E16_1degree'
# RDA.case = 'MAGIC1E16_2degree'
# RDA.case = 'MAGIC1E16_3degree'
# RDA.case = 'MAGIC1E16_4degree'

# RDA.case = 'GRACEC1E16'
# RDA.case = 'GRACEC1E16_1degree'
# RDA.case = 'GRACEC1E16_2degree'
# RDA.case = 'GRACEC1E16_3degree'
# RDA.case = 'GRACEC1E16_4degree'

'''Define the shape file of basin and its sub-basin to be assimilated with GRACE'''

RDA.basin = 'Brahmaputra3subbasins'
# RDA.basin = 'Brahmaputra1degree'
# RDA.basin = 'Brahmaputra2degree'
# RDA.basin = 'Brahmaputra3degree'
# RDA.basin = 'Brahmaputra4degree'

RDA.shp_path = '../data/basin/shp/ESA_SING/subbasins/Brahmaputra3subbasins_subbasins.shp'
# RDA.shp_path = '../data/basin/shp/ESA_SING/Grid_1/Brahmaputra1degree_subbasins.shp'
# RDA.shp_path = '../data/basin/shp/ESA_SING/Grid_2/Brahmaputra2degree_subbasins.shp'
# RDA.shp_path = '../data/basin/shp/ESA_SING/Grid_3/Brahmaputra3degree_subbasins.shp'
# RDA.shp_path = '../data/basin/shp/ESA_SING/Grid_4/Brahmaputra4degree_subbasins.shp'

'''this is useless, as the area will be automatically calculated from the shape file'''
RDA.box = [50.5, 42, 8.5, 29.5]

'''Because the size limit of repository, a limited sample data is available to allow only one-year DA'''

'''for spin-up'''
RDA.cold_begin_time = '2000-01-01'
RDA.cold_end_time = '2002-08-31'

'''for open-loop'''
RDA.warm_begin_time = '2002-09-01'
RDA.warm_end_time = '2006-12-31'

'''for data assimilation'''
RDA.resume_begin_time = '2003-09-01'
RDA.resume_end_time = '2006-12-31'

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

    es.basin_TWS(day_begin=t1, day_end=t2, dir_in='/media/user/My Book/Fan/ESA_SING/Brahmaputra',
                 dir_out='/media/user/My Book/Fan/ESA_SING/TestRes')
    es.basin_COV(day_begin=t1, day_end=t2, isDiagonal=isDiagonal, dir_in='/media/user/My Book/Fan/ESA_SING/Brahmaputra',
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

    RDA.setting_dir = str(Path(RDA.setting_dir) / mission / filter)
    RDA.case = mission + '_' + filter + '_' + grid + '_' + region

    RDA.shp_path = Path('../data/basin/shp/ESA_SING/')
    RDA.basin = region + grid
    if grid == '3subbasins':
        RDA.shp_path = RDA.shp_path / 'subbasins' / ('%s%s_subbasins.shp' % (region, grid))
    else:
        RDA.shp_path = RDA.shp_path / ('Grid_%s' % grid[0]) / ('%s%s_subbasins.shp' % (region, grid))

    pass


def demo_prepare_ESA_SING_5daily_loop():
    mission = 'MAGIC'
    filter = '1e19'
    region = 'Brahmaputra'

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

    for grid in ['3subbasins', '1degree', '2degree', '3degree', '4degree']:
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


if __name__ == '__main__':
    '''single threads for preparing GRACE data: local'''
    # demo_prepare_ESA_SING_5daily_loop()

    '''single thread for preparing model data: cluster'''
    # customize(mission='NGGM', filter='1e14', grid='3subbasins', region='Brahmaputra')
    # RDA.prepare_Forcing()
    # RDA.single_run(skip_croping_data=True, skip_signal_extraction=True)  # only SR

    '''multiple threads'''
    customize(mission='NGGM', filter='1e14', grid='3subbasins', region='Brahmaputra')
    demo_complete_DA(skipModelPerturb=True, skipObsPerturb=False, skipSR=False)  # OL and DA
    # demo_only_DA(skip_obs_perturbation=False)  # only DA

    '''single thread for plotting'''
    # demo_DA_visualization()
    # save_data()
