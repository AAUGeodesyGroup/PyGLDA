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
res_output='../../../External Data/w3ra/res'
figure_output='../../../External Data/w3ra/figure'

'''Define where to load the necessary setting files'''
RDA.setting_dir = '../settings/demo_3'

'''Define the size of ensemble to run for DA'''
RDA.ens = 2

'''Define the name of your case study'''
RDA.case = 'case_study_demo3'

'''Define the shape file of basin and its sub-basin to be assimilated with GRACE'''
'''Example-I: Regional basin-scale DA for Danube'''
RDA.basin = 'DRB'
RDA.shp_path = '../data/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp'
'''Example-II: Regional grid-scale DA for Danube, please uncomment below to activate this example'''
# RDA.basin = 'GDRB'
# RDA.shp_path = '../data/basin/shp/GDRB_shapefiles/GDRB_subbasins.shp'
'''Example-III: Global (one tile) grid-scale DA for Danube, please uncomment below to activate this example'''
# RDA.basin = 'Tile85'
# RDA.shp_path = '../data/basin/shp/global_shp_new/Tile85_subbasins.shp'

'''this is useless, as the area will be automatically calculated from the shape file'''
RDA.box = [50.5, 42, 8.5, 29.5]

'''Because the size limit of repository, a limited sample data is available to allow only one-year DA'''

'''for spin-up'''
RDA.cold_begin_time = '2005-01-01'
RDA.cold_end_time = '2005-12-31'

'''for open-loop'''
RDA.warm_begin_time = '2005-01-02'
RDA.warm_end_time = '2005-12-31'
# warm_begin_time = '2000-01-01'
# warm_end_time = '2010-01-31'

'''for data assimilation'''
RDA.resume_begin_time = '2005-04-01'
RDA.resume_end_time = '2005-12-31'

# resume_begin_time = '2002-03-31'
# resume_end_time = '2010-01-31'
RDA.isSet = False


def demo_global_run_complete(prepare=True):
    """,
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
        RDA.single_run()
        pass

    comm.barrier()

    '''open loop running to obtain ensemble of initializations, and more importantly to obtain temporal mean to be 
    removed from the GRACE observations'''
    RDA.OL_run(skip=prepare)

    comm.barrier()

    '''carry out data assimilation'''
    RDA.DA_run(skip=prepare)

    pass


if __name__ == '__main__':
    '''multiple threads'''
    demo_global_run_complete(prepare=True)