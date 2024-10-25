import sys

sys.path.append('../')

from src_FlowControl.OpenLoop import OpenLoop
from src_hydro.EnumType import init_mode
from datetime import datetime

case = 'OL_test'
setting_dir = '../settings/OL_run'
box = [-9.9, -43.8, 112.4, 154.3]
basin = 'MDB'
ens = 2

mode = init_mode.resume

cold_begin_time = '2000-01-01'
cold_end_time = '2000-01-31'

warm_begin_time = '2000-01-01'
warm_end_time = '2003-01-31'

resume_begin_time = '2002-10-31'
resume_end_time = '2003-01-31'

figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
fig_postfix = '0'


def demo_fc_OL_step_1():
    """
    A demo for single running of the model: step-1, no parallelization
    """

    '''configuration'''
    # demo = OpenLoop(case=case, setting_dir=setting_dir, ens=ens)
    demo = OpenLoop(case=case, setting_dir=setting_dir, ens=ens)

    if mode == init_mode.cold:
        begin_time = cold_begin_time
        end_time = cold_end_time
    elif mode == init_mode.warm:
        begin_time = warm_begin_time
        end_time = warm_end_time
    else:
        begin_time = resume_begin_time
        end_time = resume_end_time

    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)

    '''change the sub-setting files according to the main setting'''
    demo.generate_settings(mode=mode)

    '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
    demo.preprocess()

    '''generate the perturbation for the data'''
    demo.perturbation()

    # demo.visualize_signal()
    pass


def demo_fc_OL_step_2():
    """
    A demo for open-loop running of the model: step-2, with parallelization
    """
    import sys
    temp = sys.stdout

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == (ens+1), 'Not enough threads for parallelization!'

    '''configuration'''
    demo = OpenLoop(case=case, setting_dir=setting_dir, ens=ens)
    if mode == init_mode.cold:
        begin_time = cold_begin_time
        end_time = cold_end_time
    elif mode == init_mode.warm:
        begin_time = warm_begin_time
        end_time = warm_end_time
    else:
        begin_time = resume_begin_time
        end_time = resume_end_time

    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)

    '''change the sub-setting files according to the main setting'''
    if rank==0:
        demo.generate_settings(mode=mode)

    comm.barrier()

    '''run the model'''
    demo.model_run()

    '''Analysis the model output/states'''
    demo.extract_signal()

    '''Job finished'''
    sys.stdout = temp
    print('job finished: %s'% datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    pass


def demo_visualization():
    """
    A demo for visualization of the time-series of model states
    """

    '''keep the same as the running'''
    demo = OpenLoop(case=case, setting_dir=setting_dir, ens=ens)
    if mode == init_mode.cold:
        begin_time = cold_begin_time
        end_time = cold_end_time
    elif mode == init_mode.warm:
        begin_time = warm_begin_time
        end_time = warm_end_time
    else:
        begin_time = resume_begin_time
        end_time = resume_end_time
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)

    demo.generate_settings(mode=mode)

    '''generate/save the figure'''
    demo.visualize_signal(fig_path=figure_output, fig_postfix=fig_postfix)
    pass


if __name__ == '__main__':
    # demo_fc_OL_step_1()
    demo_fc_OL_step_2()
    # demo_visualization()