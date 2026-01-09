import sys
sys.path.append('../')
from src_FlowControl.SingleModel import SingleModel
from src_GHM.EnumType import init_mode
from datetime import datetime, timedelta

'''This is a demo to show how to run W3RA model. Be aware that the script is not able to work unless the sample data 
has been downloaded and placed at the correct folder to be recognized by the code. Nevertheless, the sample data 
indeed can be placed anywhere only if its full path is configured well in the setting file.'''

'''define a name for your case study. Here, as an example, we run W3RA for the whole Globe'''
case = 'Globe'

'''introduce the setting file (json) for your case study. In those setting file, the model can be flexibly configured 
as one wishes it to be. For example, the path of input forcing field, parameters and mask files etc.'''
setting_dir = '../settings/demo_global_run_model_locally'

'''the box coordinate of the desired area where W3RA is expected to run'''
box = [89.975, -89.975, -179.975, 179.975]

'''arbitrary name could be assigned since this is not used for this case study'''
basin = 'Globe'

'''three modes are available for W3RA: (1) cold, i.e., spin-up, where initialization is empirically defined, 
(2) warm, where initialization is obtained from pre-saved record. (3) resume, which can resume running at any place where 
it stopped last time'''
mode = init_mode.cold

'''Time period set for each mode. Here, because of the limited sample data, only 1 year is defined here as an example'''
cold_begin_time = '2005-01-01'
cold_end_time = '2005-02-01'

warm_begin_time = '2005-01-02'
warm_end_time = '2005-12-31'

resume_begin_time = '2005-04-01'
resume_end_time = '2005-12-31'


def demo_fc_single_model():
    """
    A demo for running W3RA model on a single thread.
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
    '''this is time-consuming but good news is that it is only required to run for once. please comment it after a 
    'cold' running'''
    # demo.preprocess()

    '''run the model'''
    demo.model_run()

    pass


if __name__ == '__main__':
    demo_fc_single_model()

