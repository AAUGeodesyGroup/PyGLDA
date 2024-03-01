import sys

sys.path.append('../')

from src_FlowControl.DA_GRACE import DA_GRACE
from src_hydro.EnumType import init_mode

case = 'DA_test'
setting_dir = '../settings/DA_local'
box = [-9.9, -43.8, 112.4, 154.3]
basin = 'MDB'
ens = 2

mode = init_mode.resume

cold_begin_time = '2000-01-01'
cold_end_time = '2010-01-31'

warm_begin_time = '2000-01-01'
warm_end_time = '2003-05-31'

resume_begin_time = '2002-03-31'
resume_end_time = '2003-05-31'

figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
fig_postfix = '33'


def demo_fc_DA_step_1():
    """
    preparation for open-loop running: single run
    """

    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
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

    pass


def demo_fc_DA_step_2():
    """
    A demo for open-loop running of the model: step-2, with parallelization
    """
    import sys
    temp = sys.stdout

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == (ens + 1), 'Not enough threads for parallelization!'

    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)

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
    if rank == 0:
        demo.generate_settings(mode=mode)

    comm.barrier()

    '''run the model'''
    demo.model_run()

    '''Analysis the model output/states'''
    demo.extract_signal(postfix='OL')

    '''Job finished'''
    sys.stdout = temp
    print('job finished')
    pass


def demo_fc_DA_step_3():
    """
    post-processing of open-loop running: single run
    """

    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)

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

    '''gather the mean value of the open loop: ensemble mean of the temporal mean'''
    demo.gather_OLmean()

    '''get states samples for creating NaN mask'''
    demo.get_states_sample_for_mask()

    '''above must be done based on the open-loop running.'''
    pass


def demo_fc_DA_step4():
    """
    preparation for DA running: single run
    """
    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)

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

    '''prepare the GRACE observation over region of interest'''
    '''Calculation of the COV is time-consuming, so we do not suggest to repeatedly execute this command '''
    demo.GRACE_obs_preprocess()

    '''perturb the GRACE obs for later use'''
    # demo.generate_perturbed_GRACE_obs()

    '''prepare the design matrix'''
    # demo.prepare_design_matrix()

    pass


def demo_fc_DA_step5():
    """
    DA experiment: parallel computing with MPI
    """
    import sys
    temp = sys.stdout

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == (ens + 1), 'Not enough threads for parallelization!'

    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)

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
    if rank == 0:
        demo.generate_settings(mode=mode)

    comm.barrier()

    '''run the model'''
    demo.run_DA(rank=rank)

    '''Analysis the model output/states'''
    demo.extract_signal(postfix='DA')

    '''Job finished'''
    sys.stdout = temp
    print('job finished')

    pass


def demo_visualization_OL():
    """
    A demo for visualization of the time-series of model states
    """

    '''keep the same as the running'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)

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
    demo.visualize_signal(fig_path=figure_output, fig_postfix=fig_postfix, file_postfix='OL')
    pass


def demo_visualization_DA():
    """
    A demo for visualization of the time-series of model states
    """

    '''keep the same as the running'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)

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
    demo.visualize_signal(fig_path=figure_output, fig_postfix=fig_postfix, file_postfix='DA')
    pass


if __name__ == '__main__':
    # demo_fc_DA_step_1()
    # demo_fc_DA_step_2()
    # demo_fc_DA_step_3()
    # demo_fc_DA_step4()
    # demo_fc_DA_step5()
    # demo_visualization_OL()
    demo_visualization_DA()

