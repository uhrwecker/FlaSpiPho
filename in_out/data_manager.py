import numpy as np
import configparser as cp
import pandas as pd


class DataHandling:
    def __init__(self, dir_path='./', file=None):
        self.dir = dir_path
        if not file:
            self.file = './demo_input.ini'
        else:
            self.file = file

        self.input_config = self.load_input()

    def generate_result_file(self, sigma, data):
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
        chi, iota, eta = photon.get_angles()
        dt, dr, dtheta, dphi = photon.get_ic()
        E, L, Q = photon.get_com()

        vr, vphi, gamma = emitter.get_velocities()
        u1, u3, gamma2 = emitter.get_rotation_velocities()
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

        config['SPHERE']['u1'] = str(u1)
        config['SPHERE']['u3'] = str(u3)
        config['SPHERE']['gamma'] = str(gamma2)

        with open(self.dir + '{}_{}_{}.ini'.format(str(r)[:7], str(theta)[:7], str(phi)[:7]), 'w') as file:
            config.write(file)

        return config

    def get_input_center_config(self):
        cf = self.input_config['CENTER_POSITION']

        return float(cf['s']), float(cf['r0']), float(cf['theta0']), float(cf['phi0'])

    def get_input_sphere_config(self):
        cf = self.input_config['SPHERE']

        return float(cf['rho']), float(cf['theta']), float(cf['phi']), float(cf['omega'])

    def get_input_photon_config(self):
        cf = self.input_config['PHOTON']

        return float(cf['chi']), float(cf['iota']), float(cf['eta']), cf['rotation']

    def get_input_numeric_config(self):
        cf = self.input_config['NUMERICS']

        return float(cf['start']), float(cf['stop']), int(cf['num']), float(cf['abserr']), float(cf['relerr'])

    def load_input(self):
        config = cp.ConfigParser()
        config.read(self.dir + self.file)
        return config

    def _write_input(self):
        config = cp.ConfigParser()

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
