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

'''Define the name of your case study'''
# RDA.case = 'GRACEM3D'
# RDA.case = 'GRACE_C_3deg_5daily'
RDA.case = 'NGGM_3deg_5daily'
# RDA.case = 'MAGIC_3deg_5daily'

'''Define where to load the necessary setting files'''
# RDA.setting_dir = '../settings/Europe/GRACE_mascon_3deg'
# RDA.setting_dir = '../settings/EuropeESM3/GRACE_C_3deg_5daily'
RDA.setting_dir = '../settings/EuropeESM3/NGGM_3deg_5daily'
# RDA.setting_dir = '../settings/EuropeESM3/MAGIC_3deg_5daily'

'''Define the size of ensemble to run for DA'''
RDA.ens = 30

'''Define the shape file of basin and its sub-basin to be assimilated with GRACE'''
RDA.basin = 'Europe'
RDA.shp_path = '../data/basin/shp/Europe/Grid_3/Europe_subbasins.shp'

'''this is useless, as the area will be automatically calculated from the shape file'''
RDA.box = [50.5, 42, 8.5, 29.5]

'''for spin-up'''
# RDA.cold_begin_time = '2000-01-01'
# RDA.cold_end_time = '2001-12-31'

RDA.cold_begin_time = '2011-01-01'
RDA.cold_end_time = '2011-12-31'

'''for open-loop'''
# RDA.warm_begin_time = '2002-01-01'
# RDA.warm_end_time = '2017-12-31'

# RDA.warm_begin_time = '2002-01-01'
# RDA.warm_end_time = '2006-12-31'

RDA.warm_begin_time = '2012-01-01'
RDA.warm_end_time = '2018-12-31'

'''for data assimilation'''
# RDA.resume_begin_time = '2002-03-01'
# RDA.resume_end_time = '2017-12-31'

# RDA.resume_begin_time = '2002-03-01'
# RDA.resume_end_time = '2006-12-31'

RDA.resume_begin_time = '2012-03-01'
RDA.resume_end_time = '2018-12-31'

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


def demo_DA_visualization(GRACE_res=0.5, upscale_res=0.5):
    RDA.DA_visualization_basin_ensemble()
    RDA.DA_visualization_basin_DA()
    RDA.DA_visulization_2Dmap(signal='trend', GRACE_res=GRACE_res, upscale_res=upscale_res)
    RDA.DA_visulization_2Dmap(signal='annual', GRACE_res=GRACE_res, upscale_res=upscale_res)
    pass


from src_postprocessing.DataManager import data_var


def save_data(data_type=data_var.TotalWater):
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
                                                     variable=data_type, dims=data_dim.one_dimension)
    # dsr.reduce_datasize(global_basin_mask=global_basin_mask, save_mask='/work/data_for_w3/w3ra/save_data/Mask')

    '''====================================================================='''
    dsr.aggregation_daily(date_begin=RDA.resume_begin_time, date_end=RDA.resume_end_time)

    '''OpenLoop results'''
    dsr = dataManager_ensemble_member(ens=0).configure(setting_fn=str(settings),
                                                       out_dir='/work/data_for_w3/w3ra/save_data/OL_mean',
                                                       variable=data_type, dims=data_dim.one_dimension)
    # dsr.reduce_datasize(global_basin_mask=global_basin_mask)

    '''====================================================================='''
    dsr.aggregation_daily(date_begin=RDA.resume_begin_time, date_end=RDA.resume_end_time)

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
    # demo_only_DA(skip_obs_perturbation=False)

    # demo_complete_DA(skipModelPerturb=True, skipObsPerturb=False, skipSR=True)  # OL and DA
    # demo_only_DA(skip_obs_perturbation=False)  # only DA

    '''single thread for plotting'''
    # demo_DA_visualization(GRACE_res=1, upscale_res=1)
    save_data(data_type=data_var.TotalWater)
    save_data(data_type=data_var.SoilWater)
    save_data(data_type=data_var.GroundWater)
    save_data(data_type=data_var.SurfaceWater)
    save_data(data_type=data_var.TopSoil)

