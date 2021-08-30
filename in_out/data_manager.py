import numpy as np
import configparser as cp
import pandas as pd
import os
import json


class DataHandling:
    """
    Class that handles all kinds of data manipulation (saving/loading/combination)
    """
    def __init__(self, file=None):
        """
        Initiation of Data class
        :param file: str; path to input file
        """
        if not file:
            self.file = './demo_input.ini'
        else:
            self.file = file

        # config['DATA']['fp'] has to end with /
        self.dir, self.input_config = self.load_input()

        # check if the log directory already exists in the saving directory:
        if not os.path.exists(self.dir + 'log/'):
            os.makedirs('log/')
            with open(self.dir + 'log/info.txt', 'w') as file:
                json.dump({'collisions': [], 'errors': []}, file)

        self.collisions, self.errors = self._read_from_json(self.dir + 'log/info.txt')

    def check_for_duplicates(self, r0, t0, p0):
        tag = '{}_{}_{}'.format(str(r0)[:7], str(t0)[:7], str(p0)[:7])

        return os.path.isfile(self.dir + tag + '.ini') or tag in self.collisions

    def generate_result_file(self, sigma, data):
        """
        Method to generate the data file (.csv format) from the result of the solver
        :param sigma: np.array; array which contains the affine parameter
        :param data: np.array (8, ); array which contains the result of the ODESolver
        :return: pd.DataFrame; a pandas DataFrame that contains all data
        """
        df = pd.DataFrame({
            'sigma': sigma,
            't': data[:, 0],
            'dt': data[:, 1],
            'r': data[:, 2],
            'dr': data[:, 3],
            'theta': data[:, 4],
            'dtheta': data[:, 5],
            'phi': data[:, 6],
            'dphi': data[:, 7],
        })

        r = data[:, 2][0]
        theta = data[:, 4][0]
        phi = data[:, 6][0]

        tag = self.dir + '{}_{}_{}'.format(str(r)[:7], str(theta)[:7], str(phi)[:7])
        df.to_csv(tag + '.csv', index=False)

        return df

    def generate_data_config(self, emitter, photon):
        """
        Method to generate the config file (.ini format) from the emitter and photon object, as well as
        meta data from the ODESolver
        :param emitter: emitter.Emitter object; emitter object that has all emitter meta data that produced the ray data
        :param photon: light.Photon object; photon object that has all photon meta data that produced the ray data
        :return: config; full config object (input + meta data)
        """
        chi, iota, eta = photon.get_angles()
        dt, dr, dtheta, dphi = photon.get_ic()
        E, L, Q = photon.get_com()

        vr, vphi, gamma = emitter.get_velocities()
        u1, u3, gamma2 = emitter.get_rotation_velocities()
        math_v, gamma3 = emitter.get_momentum_velocity()
        r, theta, phi = emitter.get_position()

        config = self.input_config

        config['CENTER_POSITION']['vr'] = str(vr)
        config['CENTER_POSITION']['vphi'] = str(vphi)
        config['CENTER_POSITION']['gamma'] = str(gamma)

        config['PHOTON']['chi'] = str(chi)
        config['PHOTON']['iota'] = str(iota)
        config['PHOTON']['eta'] = str(eta)
        config['PHOTON']['E'] = str(E)
        config['PHOTON']['Q'] = str(Q)
        config['PHOTON']['L'] = str(L)

        config['INITIAL_CONDITIONS'] = {}
        config['INITIAL_CONDITIONS']['t0'] = '0.'
        config['INITIAL_CONDITIONS']['r0'] = str(r)
        config['INITIAL_CONDITIONS']['theta0'] = str(theta)
        config['INITIAL_CONDITIONS']['phi0'] = str(phi)
        config['INITIAL_CONDITIONS']['dt'] = str(dt)
        config['INITIAL_CONDITIONS']['dr'] = str(dr)
        config['INITIAL_CONDITIONS']['dtheta'] = str(dtheta)
        config['INITIAL_CONDITIONS']['dphi'] = str(dphi)

        config['SPHERE']['theta'] = str(emitter.T)
        config['SPHERE']['phi'] = str(emitter.P)
        config['SPHERE']['u1'] = str(u1)
        config['SPHERE']['u3'] = str(u3)
        config['SPHERE']['gamma'] = str(gamma2)
        config['SPHERE']['mom_v'] = str(math_v)
        config['SPHERE']['gamma3'] = str(gamma3)

        with open(self.dir + '{}_{}_{}.ini'.format(str(r)[:7], str(theta)[:7], str(phi)[:7]), 'w') as file:
            config.write(file)

        return config

    def get_input_center_config(self):
        """
        Simple getter config that returns the center of the ball config.
        :return: [s, r0, theta0, phi0]; spin and position of the origin of the ball
        """
        cf = self.input_config['CENTER_POSITION']

        return float(cf['s']), float(cf['r0']), float(cf['theta0']), float(cf['phi0'])

    def get_input_sphere_config(self):
        """
        Simple getter config that returns the sphere properties config.
        :return: [rho, T, P, omega]; radius of the sphere, T and P coordinates of the surface, and omega
        """
        cf = self.input_config['SPHERE']

        return float(cf['rho']), float(cf['theta']), float(cf['phi']), float(cf['omega'])

    def get_input_photon_config(self):
        """
        Simple getter config that returns the photon config.
        :return: [chi, iota, eta, rotation]; chi, the emission angles and sense of rotation
        """
        cf = self.input_config['PHOTON']

        return float(cf['chi']), float(cf['iota']), float(cf['eta']), cf['rotation']

    def get_input_numeric_config(self):
        """
        Simple getter config that returns the numerical config.
        :return: [start, stop, num, abserr, relerr]: start, end and number of steps for numerics, as well as the
                                                     absolute and relative error margins for the ODESOlver
        """
        cf = self.input_config['NUMERICS']

        return float(cf['start']), float(cf['stop']), int(cf['num']), float(cf['abserr']), float(cf['relerr'])

    def get_input_observer_config(self):
        """
        Simple getter config that returns the observer config.
        :return: [r_obs, theta_obs, phi_obs]: position of the observer
        """
        cf = self.input_config['OBSERVER']

        return float(cf['ro']), float(cf['thetao']), float(cf['phio'])

    def write_collision_entry(self, sigma, data):
        r = data[:, 2][0]
        theta = data[:, 4][0]
        phi = data[:, 6][0]

        tag = '{}_{}_{}'.format(str(r)[:7], str(theta)[:7], str(phi)[:7])

        self.collisions.append(tag)

        with open(self.dir + 'log/info.txt', 'w') as file:
            json.dump({'collisions': self.collisions, 'errors': self.errors}, file, indent=4)

    def write_error_entry(self, errors):
        self.errors += errors

        with open(self.dir + 'log/info.txt', 'w') as file:
            json.dump({'collisions': self.collisions, 'errors': self.errors}, file, indent=4)

    def load_input(self):
        """
        Reads the input file specified in the initialization.
        :return: [fp, config]; returns the path where the data should be stored, as well as the config itself
        """
        config = cp.ConfigParser()
        config.read(self.file)
        return config['DATA']['fp'], config

    def _write_input(self):
        """
        (unused)(private)
        Create a sample input file.
        """
        config = cp.ConfigParser()

        config['DATA'] = {}
        config['DATA']['fp'] = './'

        config['CENTER_POSITION'] = {}
        config['CENTER_POSITION']['s'] = '0.'
        config['CENTER_POSITION']['r0'] = '8.'
        config['CENTER_POSITION']['theta0'] = str(np.pi / 2)
        config['CENTER_POSITION']['phi0'] = '0.'

        config['SPHERE'] = {}
        config['SPHERE']['rho'] = '1.'
        config['SPHERE']['theta'] = str(np.pi / 2)
        config['SPHERE']['phi'] = '0.'
        config['SPHERE']['omega'] = '0.0075'

        config['PHOTON'] = {}
        config['PHOTON']['chi'] = '1.'
        config['PHOTON']['iota'] = '0.'
        config['PHOTON']['eta'] = '0.'
        config['PHOTON']['rotation'] = 'positive'

        config['NUMERICS'] = {}
        config['NUMERICS']['start'] = '0.'
        config['NUMERICS']['stop'] = '15.'
        config['NUMERICS']['num'] = '5000'
        config['NUMERICS']['abserr'] = '1e-7'
        config['NUMERICS']['relerr'] = '1e-7'

        with open(self.dir+'demo_input.ini', 'w') as file:
            config.write(file)

    def _read_from_json(self, fp):
        with open(fp, 'r') as file:
            data = json.load(file)

        return data['collisions'], data['errors']
