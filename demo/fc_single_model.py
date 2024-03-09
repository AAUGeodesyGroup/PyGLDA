import sys

sys.path.append('../')

from src_FlowControl.SingleModel import SingleModel
from src_hydro.EnumType import init_mode
from datetime import datetime,timedelta

# case = 'single_run_test'
# setting_dir = '../settings/single_run'
# box = [-9.9, -43.8, 112.4, 154.3]
# basin = 'MDB'

case = 'case_study_SR'
setting_dir = '../settings/single_run'
box = [50.5, 42, 8.5, 29.5]
basin = 'DRB'

mode = init_mode.cold

cold_begin_time = '2000-01-01'
cold_end_time = '2010-01-31'

# cold_begin_time = '2010-01-31'
# cold_end_time = '2023-05-31'

warm_begin_time = '2000-01-01'
warm_end_time = '2000-01-31'

resume_begin_time = '2005-01-31'
resume_end_time = '2005-02-20'

figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
fig_postfix = '0'


def demo_fc_single_model():
    """
    A demo for single running of the model
    """

    '''configuration'''
    demo = SingleModel(case=case, setting_dir=setting_dir)
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

    # '''run the model'''
    # demo.model_run()
    #
    # '''create ini states?'''
    # if mode == init_mode.cold:
    #     modifydate = (datetime.strptime(warm_begin_time, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y%m%d')
    #     demo.create_ini_states(mode=mode, modifydate=modifydate)
    #
    # '''Analysis the model output/states'''
    # demo.extract_signal()

    # demo.visualize_signal()
    pass


def demo_visualization():
    """
    A demo for visualization of the time-series of model states
    """

    '''keep the same as the running'''
    demo = SingleModel(case=case, setting_dir=setting_dir)
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
    # demo.visualize_signal(fig_path=figure_output, fig_postfix=fig_postfix)
    demo.visualize_comparison_GRACE(fig_path=figure_output, fig_postfix=fig_postfix)
    pass


if __name__ == '__main__':
    # demo_fc_single_model()
    demo_visualization()
