import sys

sys.path.append('../')

import numpy as np
from enum import Enum
import h5py
import json
from pathlib import Path
from datetime import datetime
from src_GHM.GeoMathKit import GeoMathKit


class perturbe_method(Enum):
    addictive = 1
    multiplicative = 2
    pass


class error_distribution(Enum):
    triangle = 1
    normal = 2
    pass


class perturbation:

    def __init__(self, dp='../settings/perturbation.json'):
        self.setting = json.load(open(dp, 'r'))
        self.ens = self.setting['ensemble']
        pass

    def setDate(self, month_begin='2002-04', month_end='2002-04'):
        self.monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        return self

    def perturb_forcing(self):

        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['forcing']

        dir_out = dir_out / 'ens_forcing'

        if dir_out.exists():
            pass
        else:
            dir_out.mkdir()
            pass

        for month in self.monthlist:
            print(month.strftime('%Y-%m'))

            '''load data'''
            fn_1 = dir_in / 'forcing' / ('%s.h5' % (month.strftime('%Y-%m')))
            ff_1 = h5py.File(fn_1, 'r')

            '''save data'''
            fn_2 = dir_out / ('%s.h5' % (month.strftime('%Y-%m')))
            ff_2 = h5py.File(fn_2, 'w')

            outdata = {}
            for key in forcing:
                dict_group_load = ff_1['data'][key][:]

                if forcing[key]['is_perturbed']:

                    pp = self.__generate_perturbation(config=forcing[key], mean=dict_group_load)

                    outdata[key] = pp

                else:

                    outdata[key] = dict_group_load
                    pass

            for ens in np.arange(self.ens):
                dict_group = ff_2.create_group('ens_%s' % (ens + 1))
                for key in outdata.keys():
                    if np.ndim(outdata[key]) == 2:
                        dict_group[key] = outdata[key]
                    else:
                        dict_group[key] = outdata[key][ens]

            '''add the unperturbed ens: defined as ens 0'''
            dict_group = ff_2.create_group('ens_%s' % 0)
            for key in outdata.keys():
                dict_group[key] = ff_1['data'][key][:]

            ff_1.close()
            ff_2.close()

            '''generate perturbation'''

        pass

    def perturb_par(self):

        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['par']

        dir_out = dir_out / 'ens_par'

        if dir_out.exists():
            pass
        else:
            dir_out.mkdir()
            pass

        '''load data'''
        fn_1 = dir_in / 'par' / 'par.h5'
        ff_1 = h5py.File(fn_1, 'r')

        '''save data'''
        fn_2 = dir_out / 'par.h5'
        ff_2 = h5py.File(fn_2, 'w')

        outdata = {}
        for key in forcing:
            dict_group_load = ff_1['data'][key][:]

            if forcing[key]['is_perturbed']:

                pp = self.__generate_perturbation(config=forcing[key], mean=dict_group_load)

                outdata[key] = pp

            else:

                outdata[key] = dict_group_load
                pass

        for ens in np.arange(self.ens):
            dict_group = ff_2.create_group('ens_%s' % (ens + 1))
            for key in outdata.keys():
                if np.ndim(outdata[key]) == 2:
                    dict_group[key] = outdata[key]
                else:
                    dict_group[key] = outdata[key][ens]

        '''add the unpurturbed ens: defined as ens 0'''
        dict_group = ff_2.create_group('ens_%s' % 0)
        for key in outdata.keys():
            dict_group[key] = ff_1['data'][key][:]

        ff_1.close()
        ff_2.close()

        pass

    def perturb_forcing_spatial_coherence(self):

        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['forcing']

        dir_out = dir_out / 'ens_forcing'

        if dir_out.exists():
            pass
        else:
            dir_out.mkdir()
            pass

        for month in self.monthlist:
            print(month.strftime('%Y-%m'))

            '''load data'''
            fn_1 = dir_in / 'forcing' / ('%s.h5' % (month.strftime('%Y-%m')))
            ff_1 = h5py.File(fn_1, 'r')

            '''save data'''
            fn_2 = dir_out / ('%s.h5' % (month.strftime('%Y-%m')))
            ff_2 = h5py.File(fn_2, 'w')

            outdata = {}
            for key in forcing:
                dict_group_load = ff_1['data'][key][:]

                if forcing[key]['is_perturbed']:

                    s_shape = tuple([self.ens, np.shape(dict_group_load)[0]])

                    samples = self.Gaussian_perturbation(mean=0, std=1, size=s_shape)

                    percentage = forcing[key]['coefficients'][0]

                    dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))

                    std = dict_group_load * percentage

                    hh = dd * dict_group_load[None, :, :] + samples[:, :, None] * dd * std[None, :, :]

                    outdata[key] = hh

                else:

                    outdata[key] = dict_group_load
                    pass

            for ens in np.arange(self.ens):
                dict_group = ff_2.create_group('ens_%s' % (ens + 1))
                for key in outdata.keys():
                    if np.ndim(outdata[key]) == 2:
                        dict_group[key] = outdata[key]
                    else:
                        dict_group[key] = outdata[key][ens]

            '''add the unpurturbed ens: defined as ens 0'''
            dict_group = ff_2.create_group('ens_%s' % 0)
            for key in outdata.keys():
                dict_group[key] = ff_1['data'][key][:]

            ff_1.close()
            ff_2.close()

            '''generate perturbation'''

        pass

    def perturb_par_spatial_coherence(self):

        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['par']

        dir_out = dir_out / 'ens_par'

        if dir_out.exists():
            pass
        else:
            dir_out.mkdir()
            pass

        '''load data'''
        fn_1 = dir_in / 'par' / 'par.h5'
        ff_1 = h5py.File(fn_1, 'r')

        '''save data'''
        fn_2 = dir_out / 'par.h5'
        ff_2 = h5py.File(fn_2, 'w')

        outdata = {}
        for key in forcing:
            dict_group_load = ff_1['data'][key][:]

            if forcing[key]['is_perturbed']:

                s_shape = tuple([self.ens, np.shape(dict_group_load)[0]])

                percentage = forcing[key]['coefficients'][0]

                if forcing[key]['error_distribution'] == error_distribution.triangle.name:
                    samples = self.triangle_perturbation(mean=1, error_percentage=percentage, size=s_shape)
                    dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))

                    hh = samples[:, :, None] * dd* dict_group_load[None, :, :]

                else:
                    samples = self.Gaussian_perturbation(mean=0, std=1, size=s_shape)
                    dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))

                    std = dict_group_load * percentage

                    hh = dd * dict_group_load[None, :, :] + samples[:, :, None] * dd * std[None, :, :]

                outdata[key] = hh

            else:

                outdata[key] = dict_group_load
                pass

        for ens in np.arange(self.ens):
            dict_group = ff_2.create_group('ens_%s' % (ens + 1))
            for key in outdata.keys():
                if np.ndim(outdata[key]) == 2:
                    dict_group[key] = outdata[key]
                else:
                    dict_group[key] = outdata[key][ens]

        '''add the unpurturbed ens: defined as ens 0'''
        dict_group = ff_2.create_group('ens_%s' % 0)
        for key in outdata.keys():
            dict_group[key] = ff_1['data'][key][:]

        ff_1.close()
        ff_2.close()

        pass

    def perturb_coherent_par(self, percentage):

        samples = self.Gaussian_perturbation(mean=0, std=1, size=self.ens)

        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['par']

        dir_out = dir_out / 'ens_par'

        if dir_out.exists():
            pass
        else:
            dir_out.mkdir()
            pass

        '''load data'''
        fn_1 = dir_in / 'par' / 'par.h5'
        ff_1 = h5py.File(fn_1, 'r')

        '''save data'''
        fn_2 = dir_out / 'par.h5'
        ff_2 = h5py.File(fn_2, 'w')

        outdata = {}
        for key in forcing:
            dict_group_load = ff_1['data'][key][:]

            if forcing[key]['is_perturbed']:
                dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))

                std = dict_group_load * percentage

                hh = dd * dict_group_load[None, :, :] + samples[:, None, None] * dd * std[None, :, :]

                outdata[key] = hh
            else:

                outdata[key] = dict_group_load
                pass

        for ens in np.arange(self.ens):
            dict_group = ff_2.create_group('ens_%s' % (ens + 1))
            for key in outdata.keys():
                if np.ndim(outdata[key]) == 2:
                    dict_group[key] = outdata[key]
                else:
                    dict_group[key] = outdata[key][ens]

        '''add the unpurturbed ens: defined as ens 0'''
        dict_group = ff_2.create_group('ens_%s' % 0)
        for key in outdata.keys():
            dict_group[key] = ff_1['data'][key][:]

        ff_1.close()
        ff_2.close()

        pass

    def perturb_coherent_forcing(self, percentage):

        samples = self.Gaussian_perturbation(mean=0, std=1, size=self.ens)

        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['forcing']

        dir_out = dir_out / 'ens_forcing'

        if dir_out.exists():
            pass
        else:
            dir_out.mkdir()
            pass

        for month in self.monthlist:
            print(month.strftime('%Y-%m'))

            '''load data'''
            fn_1 = dir_in / 'forcing' / ('%s.h5' % (month.strftime('%Y-%m')))
            ff_1 = h5py.File(fn_1, 'r')

            '''save data'''
            fn_2 = dir_out / ('%s.h5' % (month.strftime('%Y-%m')))
            ff_2 = h5py.File(fn_2, 'w')

            outdata = {}
            for key in forcing:
                dict_group_load = ff_1['data'][key][:]

                if forcing[key]['is_perturbed']:

                    dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))

                    std = dict_group_load * percentage

                    hh = dd * dict_group_load[None, :, :] + samples[:, None, None] * dd * std[None, :, :]

                    outdata[key] = hh


                else:

                    outdata[key] = dict_group_load
                    pass

            for ens in np.arange(self.ens):
                dict_group = ff_2.create_group('ens_%s' % (ens + 1))
                for key in outdata.keys():
                    if np.ndim(outdata[key]) == 2:
                        dict_group[key] = outdata[key]
                    else:
                        dict_group[key] = outdata[key][ens]

            '''add the unpurturbed ens: defined as ens 0'''
            dict_group = ff_2.create_group('ens_%s' % 0)
            for key in outdata.keys():
                dict_group[key] = ff_1['data'][key][:]

            ff_1.close()
            ff_2.close()

            '''generate perturbation'''

        pass

    def __generate_perturbation(self, config: dict, mean=None):

        co = config['coefficients']

        dd = np.ones(tuple([self.ens] + list(np.shape(mean))))

        if config['perturbe_method'] == perturbe_method.addictive:
            if config['error_distribution'] == error_distribution.triangle.name:
                pp = self.triangle_perturbation(mean=co[0] * dd, error_percentage=co[1] * dd)
            else:
                pp = self.Gaussian_perturbation(mean=co[0] * dd, std=co[1] * co[0] * dd)

            mean = mean[None, :, :] + pp

        else:

            if config['error_distribution'] == error_distribution.triangle.name:
                pp = self.triangle_perturbation(mean=dd, error_percentage=co[0])
            else:
                pp = self.Gaussian_perturbation(mean=dd, std=dd * co[0])

            mean = mean[None, :, :] * pp

            pass

        return mean

    def triangle_perturbation(self, mean, error_percentage, size =None):
        return np.random.triangular(left=mean - error_percentage * mean, mode=mean,
                                    right=mean + error_percentage * mean, size=size)

    def Gaussian_perturbation(self, mean, std, size=None):
        return np.random.normal(loc=mean, scale=std, size=size)

    @staticmethod
    def save_default_json(sample_dir='/media/user/My Book/Fan/W3RA_data/perturbation_sample'):
        ff = h5py.File(Path(sample_dir) / 'forcing.h5', 'r')
        forcing = list(ff['data'].keys())
        ff.close()

        ff = h5py.File(Path(sample_dir) / 'par.h5', 'r')
        par = list(ff['data'].keys())
        ff.close()

        perturbation_dict = {
            'ensemble': 30,
            'dir': {
                'in': '/media/user/My Book/Fan/W3RA_data/crop_input/single_run_test',
                'out': '/media/user/My Book/Fan/W3RA_data/crop_input/ens'
            },
            'forcing': {},
            'par': {}
        }

        template = {
            'is_perturbed': False,
            'perturbe_method': perturbe_method.multiplicative.name,
            'error_distribution': error_distribution.triangle.name,
            'coefficients': [0.3]
        }

        for key in forcing:
            perturbation_dict['forcing'][key] = template

        for key in par:
            perturbation_dict['par'][key] = template

        default_path = Path.cwd().parent / 'settings' / 'perturbation.json'
        with open(default_path, 'w') as f:
            json.dump(perturbation_dict, f, indent=4)

        return str(default_path)


def demo1():
    dp = perturbation.save_default_json()
    # dp = '../settings/perturbation_2.json'
    # pp = perturbation(dp=dp).setDate(month_begin='1999-12', month_end='2023-05')
    #
    # pp.perturbe_forcing()
    # pp.perturbe_par()

    pass


if __name__ == '__main__':
    demo1()
