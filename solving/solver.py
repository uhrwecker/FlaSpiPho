import numpy as np
import emitter.emitter as emitter
from light import photon as light, geodesics
from in_out import data_manager as dm

from scipy.integrate import odeint


class ODESolver:
    """
        Wrapper class for solving the differential equations for the motion of timelike geodesics.
    """
    def __init__(self, directory='./', file=None):

        self.dm = dm.DataHandling(directory, file)
        s, r, _, phi0 = self.dm.get_input_center_config()
        rho, T, P, omega = self.dm.get_input_sphere_config()
        chi, iota, eta, rotation = self.dm.get_input_photon_config()
        start, stop, num, abserr, relerr = self.dm.get_input_numeric_config()

        self.emitter = emitter.EmitterProperties(s, r0=r, phi0=phi0, omega=omega, P=P, T=T, rho=rho, rotation=rotation)
        self.photon = light.PhotonProperties(self.emitter, chi, iota, eta)

        self.start = start
        self.stop = stop
        self.num = num

        self.sigma = np.linspace(start, stop, num=num)

        self.abserr = abserr
        self.relerr = relerr

        self.t0 = 0
        self.r0, self.theta0, self.phi0 = self.emitter.get_position()

        self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def solve(self):
        """
            Main routine for solving with the previously specified initial conditions
            :return: iter; [sigma, result] where sigma is the array of affine parameter, and result includes all [x, x']
        """
        psi = np.array([self.t0, self.dt, self.r0, self.dr, self.theta0, self.dtheta, self.phi0, self.dphi])

        result = odeint(geodesics.geod, psi, self.sigma, atol=self.abserr, rtol=self.relerr)

        return self.sigma, result

    def set_iota(self, iota, recalc=True):
        """
            Set (and possibly recalc) the emission angle alpha.
            :param iota: phi-like emission angle
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.photon.set_iota(iota, recalc)
        if recalc:
            self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def set_eta(self, eta, recalc=True):
        """
            Set (and possibly recalc) the emission angle beta.
            :param eta: theta-like emission angle
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.photon.set_eta(eta, recalc)
        if recalc:
            self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def set_P_and_T(self, P, T):
        chi, iota, eta, rotation = self.dm.get_input_photon_config()

        self.emitter.set_P(P)
        self.emitter.set_T(T)

        self.photon = light.PhotonProperties(self.emitter, chi, iota, eta)

        self.r0, self.theta0, self.phi0 = self.emitter.get_position()

        self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def save(self, save_colliding_light_ray=True):
        sigma, data = self.solve()
        collision = self.check_for_collision(data, 1)

        if not collision:
            self.dm.generate_data_config(self.emitter, self.photon)
            self.dm.generate_result_file(*self.solve())
        elif collision and save_colliding_light_ray:
            self.dm.generate_data_config(self.emitter, self.photon)
            self.dm.generate_result_file(*self.solve())
        else:
            print('Did not save the colliding light ray.')

        return collision

    def check_for_collision(self, data, start=5):
        r = data[:, 2]
        theta = data[:, 4]
        phi = data[:, 6]

        r0 = self.emitter.r0
        phi0 = self.emitter.phi0
        theta0 = np.pi / 2

        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)

        dx = r0 * np.cos(phi0) * np.sin(theta0)
        dy = r0 * np.sin(phi0) * np.sin(theta0)
        dz = r0 * np.cos(theta0)

        dist = np.sqrt((x - dx)**2 + (y - dy)**2 + (z - dz)**2)[start:]
        tol = 1e-10
        flag = bool(dist[dist <= self.emitter.rho+tol].size)

        return flag
