import sys

sys.path.append('../')

from FlowControl.OpenLoop import OpenLoop

case = 'OL_test'
setting_dir = '../settings/OL_run'
begin_time = '2000-01-01'
end_time = '2000-01-31'
box = [-9.9, -43.8, 112.4, 154.3]
basin = 'MDB'
ens = 2

figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
fig_postfix = '0'


def demo_fc_OL_step_1():
    """
    A demo for single running of the model: step-1, no parallelization
    """

    '''configuration'''
    # demo = OpenLoop(case=case, setting_dir=setting_dir, ens=ens)
    demo = OpenLoop(case=case, setting_dir=setting_dir, ens=ens)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)

    '''change the sub-setting files according to the main setting'''
    demo.generate_settings()

    '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
    demo.preprocess()

    '''generate the perturbation for the data'''
    demo.perturbation()

    # demo.visualize_signal()
    pass

def demo_fc_OL_step_2():
    """
    A demo for single running of the model: step-2, with parallelization
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


def demo_visualization():
    """
    A demo for visualization of the time-series of model states
    """

    '''keep the same as the running'''
    demo = OpenLoop(case=case, setting_dir=setting_dir, ens=ens)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)
    demo.generate_settings()

    '''generate/save the figure'''
    demo.visualize_signal(fig_path=figure_output, postfix=fig_postfix)
    pass


if __name__ == '__main__':
    # demo_fc_OL_step_1()
    # demo_fc_OL_step_2()
    demo_visualization()