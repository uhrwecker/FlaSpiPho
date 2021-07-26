import configparser as cp

class DataPoint:
    def __init__(self, fp='./demo.ini'):
        self.config = cp.ConfigParser()
        self.config.read(fp)

        self.s, self.rc, self.tc, self.pc, self.vr, self.vphi, self.gamma = self.get_from_center_config()
        self.rho, self.T, self.P, self.omega, self.u1, self.u3, self.gamma2 = self.get_from_sphere_config()
        self.chi, self.iota, self.eta, self.rotation, self.e, self.q, self.l = self.get_from_photon_config()
        self.robs, self.tobs, self.pobs = self.get_from_observer_config()
        self.start, self.stop, self.num, self.abserr, self.relerr = self.get_from_numerics_config()
        self.t0, self.r0, self.theta0, self.phi0, self.dt, self.dr, self.dtheta, self.dphi = self.get_from_ic_config()

    def get_from_center_config(self):
        cf = self.config['CENTER_POSITION']
        return float(cf['s']), float(cf['r0']), float(cf['theta0']), float(cf['phi0']), float(cf['vr']), \
               float(cf['vphi']), float(cf['gamma'])

    def get_from_sphere_config(self):
        cf = self.config['SPHERE']
        return float(cf['rho']), float(cf['theta']), float(cf['phi']), float(cf['omega']), float(cf['u1']), \
               float(cf['u3']), float(cf['gamma'])

    def get_from_photon_config(self):
        cf = self.config['PHOTON']
        return float(cf['chi']), float(cf['iota']), float(cf['eta']), cf['rotation'], float(cf['e']), \
               float(cf['q']), float(cf['l'])

    def get_from_observer_config(self):
        cf = self.config['OBSERVER']
        return float(cf['ro']), float(cf['thetao']), float(cf['phio'])

    def get_from_numerics_config(self):
        cf = self.config['NUMERICS']
        return float(cf['start']), float(cf['stop']), float(cf['num']), float(cf['abserr']), float(cf['relerr'])

    def get_from_ic_config(self):
        cf = self.config['INITIAL_CONDITIONS']
        return float(cf['t0']), float(cf['r0']), float(cf['theta0']), float(cf['phi0']), \
               float(cf['dt']), float(cf['dr']), float(cf['dtheta']), float(cf['dphi'])