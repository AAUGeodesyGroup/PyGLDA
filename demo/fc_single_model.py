from FlowControl.SingleModel import SingleModel

case = 'single_run_test'
setting_dir = '../settings/single_run'
begin_time = '2000-01-01'
end_time = '2000-01-31'
box = [-9.9, -43.8, 112.4, 154.3]
basin = 'MDB'

figure_output = '/media/user/My Book/Fan/W3RA_data/figure'
fig_postfix = '0'


def demo_fc_single_model():
    """
    A demo for single running of the model
    """

    '''configuration'''
    demo = SingleModel(case=case, setting_dir=setting_dir)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)

    '''change the sub-setting files according to the main setting'''
    demo.generate_settings()

    '''crop the data (forcing field, climatologies, parameters and land mask) at regions of interest'''
    demo.preprocess()

    '''run the model'''
    demo.model_run()

    '''Analysis the model output/states'''
    demo.extract_signal()

    # demo.visualize_signal()
    pass


def demo_visualization():
    """
    A demo for visualization of the time-series of model states
    """

    '''keep the same as the running'''
    demo = SingleModel(case=case, setting_dir=setting_dir)
    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)
    demo.generate_settings()

    '''generate/save the figure'''
    demo.visualize_signal(fig_path=figure_output, postfix=fig_postfix)
    pass


if __name__ == '__main__':
    # demo_fc_single_model()
    demo_visualization()
