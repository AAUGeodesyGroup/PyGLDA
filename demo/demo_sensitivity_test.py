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
RDA.setting_dir = '../settings/sensitivity_test'

'''Define the size of ensemble to run for DA'''
RDA.ens = 30

'''Define the name of your case study'''
# RDA.case = 'NGGM1E16'
# RDA.case = 'NGGM1E16_1degree'
# RDA.case = 'NGGM1E16_2degree'
# RDA.case = 'NGGM1E16_3degree'
# RDA.case = 'NGGM1E16_4degree'

# RDA.case = 'MAGIC1E16'
RDA.case = 'sensitivity'
# RDA.case = 'MAGIC1E16_2degree'
# RDA.case = 'MAGIC1E16_3degree'
# RDA.case = 'MAGIC1E16_4degree'

# RDA.case = 'GRACEC1E16'
# RDA.case = 'GRACEC1E16_1degree'
# RDA.case = 'GRACEC1E16_2degree'
# RDA.case = 'GRACEC1E16_3degree'
# RDA.case = 'GRACEC1E16_4degree'

'''Define the shape file of basin and its sub-basin to be assimilated with GRACE'''

# RDA.basin = 'Brahmaputra3subbasins'
RDA.basin = 'Brahmaputra1degree'
# RDA.basin = 'Brahmaputra2degree'
# RDA.basin = 'Brahmaputra3degree'
# RDA.basin = 'Brahmaputra4degree'

# RDA.shp_path = '../data/basin/shp/ESA_SING/subbasins_3/Brahmaputra3subbasins_3_subbasins.shp'
RDA.shp_path = '../data/basin/shp/ESA_SING/Grid_1/Brahmaputra1degree_subbasins.shp'
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
RDA.warm_end_time = '2004-12-31'

'''for data assimilation'''
RDA.resume_begin_time = '2003-09-01'
RDA.resume_end_time = '2004-12-31'

RDA.isSet = False


def demo_complete_OL(skip_perturbation=False, skipSR=False):
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
        print('===================sensitivity test============================')
        if not skipSR:
            RDA.single_run()
            pass

    comm.barrier()

    '''open loop running to obtain ensemble of initializations, and more importantly to obtain temporal mean to be 
    removed from the GRACE observations'''
    RDA.OL_run(skip_perturbation=skip_perturbation)

    comm.barrier()

    pass


def demo_OL_visualization():
    RDA.OL_visualization_basin_ensemble()
    pass


if __name__ == '__main__':
    '''single threads for preparing data'''
    # RDA.prepare_Forcing()
    # RDA.single_run(skip_croping_data=True, skip_signal_extraction=True) # only SR

    '''multiple threads'''
    # demo_complete_OL(skip_perturbation=False, skipSR=True)  # OL and DA

    '''single thread for plotting'''
    demo_OL_visualization()
