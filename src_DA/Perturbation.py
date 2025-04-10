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
    additive = 1
    multiplicative = 2
    pass


class error_distribution(Enum):
    triangle = 1
    gaussian = 2
    pass


class perturbation:
    """
    #TODO: A sensitivity test is necessary for further analysis!
    Notice: Take care of the unit of the perturbation! see ext_forcing.py
    """

    def __init__(self, dp='../settings/perturbation.json'):
        self.setting = json.load(open(dp, 'r'))
        self.ens = self.setting['ensemble']
        pass

    def setDate(self, month_begin='2002-04', month_end='2002-04'):
        self.monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        return self

    def perturb_forcing(self):
        forcing = self.setting['correlation']['forcing']

        if forcing['customize']:
            self._perturb_forcing_customize()
            return

        """using the built-in functions"""
        if forcing["isSpatialCorrelated"] and forcing["isTemporalCorrelated"]:
            self._forcing_spatiotemporal_correlated()
        elif forcing["isSpatialCorrelated"] and (not forcing["isTemporalCorrelated"]):
            self._forcing_spatial_correlated_temporal_noncorrelated()
        elif (not forcing["isSpatialCorrelated"]) and forcing["isTemporalCorrelated"]:
            self._forcing_spatial_noncorrelated_temporal_correlated()
        else:
            self._forcing_spatiotemporal_noncorrelated()

        pass

    def _perturb_forcing_customize(self):
        """
        This has to be customized by user
        """
        # todo: considering partial spatial-temporal correlation
        pass

    def perturb_par(self):

        par = self.setting['correlation']['par']

        if par['customize']:
            self._perturb_par_customize()
            return

        """using the built-in functions"""
        if par["isSpatialCorrelated"]:
            self._par_spatial_correlated()
        else:
            self._par_spatial_noncorrelated()

        pass

    def _perturb_par_customize(self):
        """
        This has to be customized by user
        """

        pass

    def _forcing_spatiotemporal_noncorrelated(self):
        """
        This means the forcing fields are perturbed for each individual grid and time epoch
        """

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

    def _forcing_spatiotemporal_correlated(self):
        """
        this means that all perturbation in both space and time domain follow the same pattern.
        """

        forcing = self.setting['forcing']

        '''pre-generation of the sample as templates'''
        samples_mul, samples_add = {}, {}
        for key in forcing:
            if not forcing[key]['is_perturbed']:
                continue
            if forcing[key]['perturbe_method'] == perturbe_method.multiplicative.name:
                samples_mul[key] = self.__generate_perturbation(config=forcing[key],
                                                                mean=np.ones(1))
                pass
            else:
                samples_add[key] = self.__generate_perturbation(config=forcing[key],
                                                                mean=np.zeros(1))

        ''''''
        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

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

                if not forcing[key]['is_perturbed']:
                    outdata[key] = dict_group_load
                    continue

                dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))

                if forcing[key]['perturbe_method'] == perturbe_method.multiplicative.name:
                    hh = samples_mul[key][:, :, None] * dd * dict_group_load[None, :, :]
                    outdata[key] = hh
                else:
                    hh = dd * dict_group_load[None, :, :] + samples_add[key][:, :, None] * dd
                    outdata[key] = hh

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

    def _forcing_spatial_correlated_temporal_noncorrelated(self):
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

                if not forcing[key]['is_perturbed']:
                    outdata[key] = dict_group_load
                    continue

                dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))
                s_shape = tuple([np.shape(dict_group_load)[0]])
                if forcing[key]['perturbe_method'] == perturbe_method.multiplicative.name:
                    samples = self.__generate_perturbation(config=forcing[key], mean=np.ones(s_shape))
                    hh = samples[:, :, None] * dd * dict_group_load[None, :, :]
                    outdata[key] = hh
                    pass
                else:
                    samples = self.__generate_perturbation(config=forcing[key], mean=np.zeros(s_shape))
                    hh = dd * dict_group_load[None, :, :] + samples[:, :, None] * dd
                    outdata[key] = hh

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

    def _forcing_spatial_noncorrelated_temporal_correlated(self):

        forcing = self.setting['forcing']

        ''''''
        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

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

            '''generate template for later use'''
            if month == self.monthlist[0]:
                samples_add = {}
                samples_mul = {}
                for key in forcing:
                    if not forcing[key]['is_perturbed']:
                        continue
                    s_shape = tuple([np.shape(ff_1['data'][key][:])[1]])
                    if forcing[key]['perturbe_method'] == perturbe_method.multiplicative.name:
                        samples_mul[key] = self.__generate_perturbation(config=forcing[key],
                                                                        mean=np.ones(s_shape))
                        pass
                    else:
                        samples_add[key] = self.__generate_perturbation(config=forcing[key],
                                                                        mean=np.zeros(s_shape))

            '''generate perturbation'''
            outdata = {}
            for key in forcing:
                dict_group_load = ff_1['data'][key][:]

                if not forcing[key]['is_perturbed']:
                    outdata[key] = dict_group_load
                    continue

                dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))

                if forcing[key]['perturbe_method'] == perturbe_method.multiplicative.name:
                    hh = samples_mul[key][:, None, :] * dd * dict_group_load[None, :, :]
                    outdata[key] = hh
                else:
                    hh = dd * dict_group_load[None, :, :] + samples_add[key][:, None, :] * dd
                    outdata[key] = hh

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

    def _par_spatial_noncorrelated(self):
        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        par = self.setting['par']

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
        for key in par:
            dict_group_load = ff_1['data'][key][:]

            if par[key]['is_perturbed']:

                pp = self.__generate_perturbation(config=par[key], mean=dict_group_load)

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

    def _par_spatial_correlated(self):
        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        pars = self.setting['par']

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
        for key in pars:
            dict_group_load = ff_1['data'][key][:]

            if not pars[key]['is_perturbed']:
                outdata[key] = dict_group_load
                continue

            ''''''
            dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))
            s_shape = tuple([np.shape(dict_group_load)[0]])
            if pars[key]['perturbe_method'] == perturbe_method.multiplicative.name:
                samples = self.__generate_perturbation(config=pars[key], mean=np.ones(s_shape))
                hh = samples[:, :, None] * dd * dict_group_load[None, :, :]
                outdata[key] = hh
                pass
            else:
                samples = self.__generate_perturbation(config=pars[key], mean=np.zeros(s_shape))
                hh = dd * dict_group_load[None, :, :] + samples[:, :, None] * dd
                outdata[key] = hh

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

        pass

    def __generate_perturbation(self, config: dict, mean=None):

        co = config['coefficients']

        dd = np.ones(tuple([self.ens] + list(np.shape(mean))))

        if config['perturbe_method'] == perturbe_method.additive.name:
            if config['error_distribution'] == error_distribution.triangle.name:
                """ co[0]: limit"""
                pp = self.triangle_perturbation(mean=0 * dd, limit=dd * co[0])
            else:
                """ co[0]: std"""
                pp = self.Gaussian_perturbation(mean=0 * dd, std=co[0] * dd)
                pass

            mean = np.expand_dims(mean, axis=0) + pp

        else:
            """multiplicative error"""
            if config['error_distribution'] == error_distribution.triangle.name:
                """co[0]: error percentage"""
                pp = self.triangle_perturbation(mean=dd, limit=dd * co[0])
                pass
            else:
                """co[0]: error percentage"""
                pp = self.Gaussian_perturbation(mean=dd, std=dd * co[0])

            mean = np.expand_dims(mean, axis=0) * pp

            pass

        return mean

    def triangle_perturbation(self, mean, limit, size=None):
        return np.random.triangular(left=mean - limit, mode=mean,
                                    right=mean + limit, size=size)

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
            "correlation": {
                "forcing": {
                    "isSpatialCorrelated": True,
                    "isTemporalCorrelated": False,
                    "customize": False
                },
                "par": {
                    "isSpatialCorrelated": True,
                    "customize": False
                }
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


@DeprecationWarning
class perturbation_old:
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

                    hh = samples[:, :, None] * dd * dict_group_load[None, :, :]

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

        '''add the unperturbed ens: defined as ens 0'''
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

        if config['perturbe_method'] == perturbe_method.additive:
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

    def triangle_perturbation(self, mean, error_percentage, size=None):
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
            "correlation": {
                "forcing": {
                    "isSpatialCorrelated": True,
                    "isTemporalCorrelated": False,
                    "customize": False
                },
                "par": {
                    "isSpatialCorrelated": True,
                    "isTemporalCorrelated": False,
                    "customize": False
                }
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
    # dp = perturbation.save_default_json()
    dp = '../settings/perturbation.json'
    pp = perturbation(dp=dp).setDate(month_begin='2000-01', month_end='2000-02')
    # pp.perturb_forcing()
    pp.perturb_par()

    pass


if __name__ == '__main__':
    demo1()
