import sys

sys.path.append('../')

from FlowControl.DA_GRACE import DA_GRACE

case = 'DA_test'
setting_dir = '../settings/DA_local'
begin_time = '2005-01-01'
end_time = '2005-01-31'
box = [-9.9, -43.8, 112.4, 154.3]
basin = 'MDB'
ens = 2

figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
fig_postfix = '0'


def demo_fc_DA_step_1():
    """
    preparation for open-loop running
    """

    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)

    '''change the sub-setting files according to the main setting'''
    demo.generate_settings()

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

    assert size == (ens+1), 'Not enough threads for parallelization!'

    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)

    '''run the model'''
    demo.model_run()

    '''Analysis the model output/states'''
    demo.extract_signal()

    '''Job finished'''
    sys.stdout = temp
    print('job finished')
    pass


def demo_fc_DA_step_3():
    """
    preparation for DA running
    """

    '''configuration'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)
    demo.generate_settings()

    '''gather the mean value of the open loop: ensemble mean of the temporal mean'''
    demo.gather_OLmean()

    '''prepare the GRACE observation'''
    # demo.GRACE_obs_preprocess()

    '''perturb the GRACE obs for later use'''
    # demo.generate_perturbed_GRACE_obs()

    '''prepare the design matrix'''
    # demo.prepare_design_matrix()
    # demo.visualize_signal()

    pass

def demo_visualization():
    """
    A demo for visualization of the time-series of model states
    """

    '''keep the same as the running'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)
    demo.generate_settings()

    '''generate/save the figure'''
    demo.visualize_signal(fig_path=figure_output, postfix=fig_postfix)
    pass


if __name__ == '__main__':
    # demo_fc_DA_step_1()
    # demo_fc_DA_step_2()
    demo_fc_DA_step_3()
    # demo_visualization()