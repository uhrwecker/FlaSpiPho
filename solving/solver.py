import numpy as np
import emitter.emitter as emitter
from light import photon as light, geodesics
from in_out import data_manager as dm

from scipy.integrate import odeint


class ODESolver:
    """
        Wrapper class for solving the differential equations for the motion of timelike geodesics.
    """
    def __init__(self, file=None):
        """
        :param file: str; path to the input .ini file
        """
        # setup the initial parameters from the input file:
        self.dm = dm.DataHandling(file)
        s, r, _, phi0 = self.dm.get_input_center_config()
        rho, T, P = self.dm.get_input_sphere_config()
        chi, iota, eta, rotation = self.dm.get_input_photon_config()
        start, stop, num, abserr, relerr = self.dm.get_input_numeric_config()

        # setup the emitter and photon object:
        self.emitter = emitter.EmitterProperties(s, r0=r, phi0=phi0, P=P, T=T, rho=rho, rotation=rotation)
        self.photon = light.PhotonProperties(self.emitter, chi, iota, eta)

        self.start = start
        self.stop = stop
        self.num = num

        # array of the affine parameter:
        self.sigma = np.linspace(start, stop, num=num)

        self.abserr = abserr
        self.relerr = relerr

        # setup initial position:
        self.t0 = 0
        self.r0, self.theta0, self.phi0 = self.emitter.get_position()

        # setup initial velocity:
        self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def solve(self):
        """
            Main routine for solving with the previously specified initial conditions
            :return: iter; [sigma, result] where sigma is the array of affine parameter, and result includes all [x, x']
        """
        psi = np.array([self.t0, self.dt, self.r0, self.dr, self.theta0, self.dtheta, self.phi0, self.dphi])

        result = odeint(geodesics.geod, psi, self.sigma, atol=self.abserr, rtol=self.relerr)

        return self.sigma, result

    def check_duplicate_saving(self):
        return self.dm.check_for_duplicates(self.r0, self.theta0, self.phi0)

    def set_iota(self, iota, recalc=True):
        """
            Set (and possibly recalculate) the emission angle iota.
            :param iota: phi-like emission angle
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.photon.set_iota(iota, recalc)
        if recalc:
            self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def set_eta(self, eta, recalc=True):
        """
            Set (and possibly recalculate) the emission angle eta.
            :param eta: theta-like emission angle
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.photon.set_eta(eta, recalc)
        if recalc:
            self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def set_T_and_P(self, T, P):
        """
        Set the local spherical coordinates P and T that point to a point on the surface of the ball.
        :param T: float; local theta-like coordinate in the rest frame of the ball
        :param P: float; local phi-like coordinate in the rest frame of the ball
        """
        chi, iota, eta, rotation = self.dm.get_input_photon_config()

        self.emitter.set_P(P, recalc=False)
        self.emitter.set_T(T)

        self.photon = light.PhotonProperties(self.emitter, chi, iota, eta)

        self.r0, self.theta0, self.phi0 = self.emitter.get_position()

        self.dt, self.dr, self.dtheta, self.dphi = self.photon.get_ic()

    def save(self, save_colliding_light_ray=True):
        """
        Method to save the light ray via the data manager.
        :param save_colliding_light_ray: bool; handles whether a light ray that collides with the emitter should be
                                               saved
        :return: collision; bool that measures if the light ray collided with the emitter
        """
        sigma, data = self.solve()
        collision = self.check_for_collision(data, 1)

        if not collision:
            self.dm.generate_data_config(self.emitter, self.photon)
            self.dm.generate_result_file(*self.solve())
        elif collision and save_colliding_light_ray:
            self.dm.generate_data_config(self.emitter, self.photon)
            self.dm.generate_result_file(*self.solve())
            self.dm.write_collision_entry(*self.solve())
        else:
            self.dm.write_collision_entry(*self.solve())
            print('Did not save the colliding light ray.\n')

        return collision

    def check_for_collision(self, data, start=5):
        """
        Routine that checks if a data set from self.solve intersects with the extended position of the ball.
        :param data: np.ndarray (8, ); array that contains all the light ray data.
        :param start: int; starting index at which the array should be checked for collision. Set it to start >= 1.
        :return: bool; flag that describes if the emitter was hit.
        """
        # tolerance when calculating the collision:
        tol = 1e-10

        # light ray array:
        r = data[:, 2]
        theta = data[:, 4]
        phi = data[:, 6]

        # position of the emitter:
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

        flag = bool(dist[dist <= self.emitter.rho+tol].size)

        return flag
