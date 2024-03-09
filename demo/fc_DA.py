import sys

sys.path.append('../')

from src_FlowControl.DA_GRACE import DA_GRACE
from src_hydro.EnumType import init_mode

# case = 'DA_test'
# # setting_dir = '../settings/DA_local'
# setting_dir = '../settings/Ucloud_DA'
# box = [-9.9, -43.8, 112.4, 154.3]
# basin = 'MDB'
# ens= 30

setting_dir = '../settings/Ucloud_DA'
ens = 30
mode = init_mode.warm

case = 'case_study_DA'
box = [50.5, 42, 8.5, 29.5]
basin = 'DRB'

cold_begin_time = '2000-01-01'
cold_end_time = '2010-01-31'

warm_begin_time = '2000-01-01'
warm_end_time = '2023-05-31'

resume_begin_time = '2002-03-31'
resume_end_time = '2023-05-31'

figure_output = '/work/data_for_w3/w3ra/figure'
fig_postfix = 'DA2'


def demo_fc_DA_step_1(mode=init_mode.warm):
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
    # demo.preprocess()

    '''generate the perturbation for the data'''
    demo.perturbation()

    pass


def demo_fc_DA_step_2(mode=init_mode.warm):
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


def demo_fc_DA_step_3(mode=init_mode.warm):
    """
    post-processing
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

    '''post_processing'''
    demo.post_processing(file_postfix='OL')

    pass


def demo_fc_DA_step_4(mode=init_mode.resume):
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
    demo.gather_OLmean(post_fix='OL')

    '''get states samples for creating NaN mask'''
    demo.get_states_sample_for_mask()

    '''above must be done based on the open-loop running.'''
    pass


def demo_fc_DA_step_5(mode=init_mode.resume):
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
    # demo.GRACE_obs_preprocess()

    '''perturb the GRACE obs for later use'''
    demo.generate_perturbed_GRACE_obs()

    '''prepare the design matrix'''
    demo.prepare_design_matrix()

    pass


def demo_fc_DA_step_6(mode=init_mode.resume):
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


def demo_fc_DA_step_7(mode=init_mode.resume):
    """
    post-processing
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

    '''post_processing'''
    demo.post_processing(file_postfix='DA')

    pass


def demo_visualization_OL(mode=init_mode.warm):
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


def demo_visualization_DA(mode=init_mode.resume):
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


def demo_visualization_DA_2(signal='TWS'):
    """
    A demo for visualization of the time-series of model states
    """
    from src_DA.Analysis import Postprocessing_basin
    import pygmt
    import h5py
    from src_hydro.GeoMathKit import GeoMathKit
    import json

    '''load OL result'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
    mode = init_mode.warm
    begin_time = warm_begin_time
    end_time = warm_end_time

    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)
    demo.generate_settings(mode=mode)

    file_postfix = 'OL'
    pp = Postprocessing_basin(ens=ens, case=case,
                              date_begin=begin_time,
                              date_end=end_time)
    states_OL = pp.get_states(post_fix=file_postfix, dir=demo._outdir2)

    '''load DA result'''
    begin_time = resume_begin_time
    end_time = resume_end_time
    file_postfix = 'DA'
    pp = Postprocessing_basin(ens=ens, case=case,
                              date_begin=begin_time,
                              date_end=end_time)

    states_DA = pp.get_states(post_fix=file_postfix, dir=demo._outdir2)

    '''load GRACE'''
    dp_dir = setting_dir / 'DA_setting.json'
    dp4 = json.load(open(dp_dir, 'r'))
    GRACE = pp.get_GRACE(obs_dir=dp4['obs']['dir'])

    OL_time = states_OL['time']
    DA_time = states_DA['time']

    basin_num = len(list(states_DA.keys())) - 1

    pass


def demo_visualization_DA_2(signal='TWS'):
    """
    A demo for visualization of the time-series of model states
    """
    from src_DA.Analysis import Postprocessing_basin
    import pygmt
    import h5py
    from src_hydro.GeoMathKit import GeoMathKit
    import json
    from pathlib import Path
    import numpy as np

    '''load OL result'''
    demo = DA_GRACE(case=case, setting_dir=setting_dir, ens=ens)
    mode = init_mode.warm
    begin_time = warm_begin_time
    end_time = warm_end_time

    demo.configure_time(begin_time=begin_time, end_time=end_time)
    demo.configure_area(box=box, basin=basin)
    demo.generate_settings(mode=mode)

    file_postfix = 'OL'
    pp = Postprocessing_basin(ens=ens, case=case, basin=basin,
                              date_begin=begin_time,
                              date_end=end_time)
    # states_OL = pp.get_states(post_fix=file_postfix, dir=demo._outdir2)
    states_OL = pp.load_states(prefix=(file_postfix + '_' + basin))
    file_postfix = 'DA'
    states_DA = pp.load_states(prefix=(file_postfix + '_' + basin))

    '''load GRACE'''
    dp_dir = Path(setting_dir) / 'DA_setting.json'
    dp4 = json.load(open(dp_dir, 'r'))
    GRACE = pp.get_GRACE(obs_dir=dp4['obs']['dir'])

    pp.save_GRACE(prefix=basin)

    OL_time = states_OL['time']
    DA_time = states_DA['time']
    GR_time = GRACE['time']

    basin_num = len(list(states_DA.keys())) - 1

    '''plot figure'''
    fig = pygmt.Figure()

    i = 0
    offset = 3
    height = 4.6
    for basin_id in GRACE['ens_mean'].keys():
        i += 1
        GRACE_ens_mean = GRACE['ens_mean'][basin_id]
        GRACE_original = GRACE['original'][basin_id]
        OL = states_OL[basin_id][signal][0]
        OL_ens_mean = np.mean(np.array(list(states_OL[basin_id][signal].values()))[1:, ], axis=0)
        DA_ens_mean = np.mean(np.array(list(states_DA[basin_id][signal].values()))[1:, ], axis=0)

        values = [GRACE_ens_mean, OL, OL_ens_mean, DA_ens_mean]
        vvmin = []
        vvmax = []
        for vv in values:
            vvmin.append(np.min(vv[5:]))
            vvmax.append(np.max(vv[5:]))

        vmin, vmax = min(vvmin), min(vvmax)
        dmin = vmin - (vmax - vmin) * 0.1
        dmax = vmax + (vmax - vmin) * 0.1
        sp_1 = int(np.round((vmax - vmin) / 10))
        if sp_1 == 0:
            sp_1 = 0.5
        sp_2 = sp_1 * 2

        sf = '%sc' % (offset * height)
        if i % (offset + 1) == 0:
            fig.shift_origin(yshift=sf, xshift='14c')
            pass

        fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                    frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa2f1",
                           'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

        fig.plot(x=OL_time, y=OL, pen="0.5p,blue,-", label='%s' % ('OL_unperturbed'), transparency=30)
        fig.plot(x=OL_time, y=OL_ens_mean, pen="0.5p,red", label='%s' % ('OL_ens_mean'), transparency=30)
        fig.plot(x=DA_time, y=DA_ens_mean, pen="0.5p,green", label='%s' % ('DA'), transparency=30)
        fig.plot(x=GR_time, y=GRACE_ens_mean, pen="0.5p,black", label='%s' % ('GRACE_ens_mean'), transparency=30)
        fig.plot(x=GR_time, y=GRACE_original, pen="0.5p,purple,--.", label='%s' % ('GRACE_original'), transparency=30)

        # fig.legend(position='jTR', box='+gwhite+p0.5p')
        fig.legend(position='jBL')
        fig.shift_origin(yshift='-%sc' % height)

        fig.savefig(str(Path(figure_output) / ('Exp2_%s_%s_%s.pdf' % (case, signal, fig_postfix))))
        fig.savefig(str(Path(figure_output) / ('Exp2_%s_%s_%s.png' % (case, signal, fig_postfix))))

        # fig.show()
        pass

    pass


if __name__ == '__main__':
    '''OL'''
    # demo_fc_DA_step_1()
    # demo_fc_DA_step_2()
    # demo_fc_DA_step_3()

    '''DA'''
    # demo_fc_DA_step_4()
    # demo_fc_DA_step_5()
    # demo_fc_DA_step_6()
    # demo_fc_DA_step_7()


    '''visualization'''
    # demo_visualization_OL()
    # demo_visualization_DA()
    demo_visualization_DA_2()
