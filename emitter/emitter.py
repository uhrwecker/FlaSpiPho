import numpy as np

from utility.util import convert_position

class EmitterProperties:
    """
        Class containing all emitter properties.
    """
    def __init__(self, s, r0=8, phi0=0, omega=1, P=0, T=np.pi/2, rho=0.2, rotation='positive'):
        """
        :param s: float; spin of the object, perpendicular to the angular momentum of the orbit
        :param r0: float; radial position of the center of the sphere
        :param phi0: float; phi-position of the center of the sphere
        :param omega: float; coordinate dot(phi) constant - used to describe the orbital velocity
        :param P: float; local spherical phi-like coordinate on the spheres' surface
        :param T: float; local spherical theta-like coordinate on the spheres' surface
        :param rho: float; local radius of the sphere, as measured from the rest system of the sphere
        :param rotation: str ['positive', 'negative']; sense of rotation of the emission object around the origin
        """
        # spin of the emitter:
        self.s = s

        # position on sphere:
        self.rho = rho
        self.P = P
        self.T = T

        # check for rho and s:
        if 2 * rho * 5/2 * s / rho**2 > 1:
            raise ValueError('The radius of the sphere is too high.')

        # sense of rotation:
        self.rotation = rotation

        # position of the center:
        self.r0 = r0
        self.phi0 = phi0

        # coordinate position on the sphere:
        self.r = None
        self.theta = None
        self.phi = None

        # measure for the orbital velocity:
        self.omega = omega

        # orbit velocities:
        self.vr = None
        self.vphi = None
        self.gamma = None

        # linear velocity on the surface of the sphere:
        self.u1 = None
        self.u3 = None
        self.gamma2 = None

        # velocity between tangent and momentum:
        self.math_v = None
        self.gamma3 = None

        self.setup()

    def setup(self):
        """
            Basic setup function.
        """
        self.r, self.theta, self.phi = convert_position(self.r0, np.pi / 2, self.phi0, self.rho, self.T, self.P)
        self.vr, self.vphi, self.gamma = self.calculate_vel()
        self.u1, self.u3, self.gamma2 = self.calculate_rotation()
        self.math_v, self.gamma3 = self.calculate_mom()

    def calculate_vel(self):
        """
            Routine to encapsule the calculation of the physical velocities.
            :return: [vr, vphi, gamma]; return the radial and orbit velocities, as well as common gamma.
        """
        vphi = self._vphi()
        vr = 0  # self._vr(); only circular orbits considered, thus vr = 0
        if vphi > 1:
            raise ValueError(f'The orbit velocity {vphi} is too high.')

        return vr, vphi, 1/np.sqrt(1 - vr**2 - vphi**2)

    def calculate_rotation(self):
        """
        Method to calculate the linear velocity on the surface, result of the angular velocity
        :return: [u1, u3, gamma(2)]; return the linear velocity on the surface, as well as gamma.
        """
        u1 = self._u1()
        u3 = self._u3()

        return u1, u3, 1/np.sqrt(1 - u1**2 - u3**2)

    def calculate_mom(self):
        math_v = self._math_v()

        return math_v, 1/np.sqrt(1 - math_v**2)

    def set_s(self, s, recalc=True):
        """
            Set (and possibly recalc) the spin parameter
            :param s: float; spin of the timelike object
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.s = s
        if recalc:
            self.setup()

    def set_r0(self, r0, recalc=True):
        """
            Set (and possibly recalc) the orbit radius parameter.
            :param r0: float; orbit of the timelike object
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.r0 = r0
        if recalc:
            self.setup()

    def set_P(self, P, recalc=True):
        """
        Set (and possibly recalc) the local Phi-like coordinate.
        :param P: float; local spherical phi-like coordinate on the spheres' surface
        :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.P = P
        if recalc:
            self.setup()

    def set_T(self, T, recalc=True):
        """

        :param T: float; local spherical theta-like coordinate on the spheres' surface
        :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.T = T
        if recalc:
            self.setup()

    def get_r0(self):
        """
            Getter for the orbit radius parameter.
            :return: float; orbit of the timelike object.
        """
        return self.r0

    def get_velocities(self):
        """
            Getter for the physical velocities of the timelike object.
            :return: [vr, vphi, gamma]; physical velocities of the emitter, as well as well known gamma.
        """
        return self.vr, self.vphi, self.gamma

    def get_rotation_velocities(self):
        """
            Getter for the physical velocities of the timelike object.
            :return: [u1, u3, gamma2]; linear velocities of the surface, as well as well known gamma.
        """
        return self.u1, self.u3, self.gamma2

    def get_momentum_velocity(self):
        return self.math_v, self.gamma3

    def get_sense_of_rotation(self):
        """
            Getter for the sense of rotation.
            :return: ['positive', 'negative]; determines the sign of the angular momentum of the orbit.
        """
        return self.rotation

    def get_s(self):
        """
            Getter for the spin of the timelike object
            :return: float; spin of the timelike object.
        """
        return self.s

    def get_position(self):
        """
        Getter for the coordinate position of the surface point.
        :return: [r, theta, phi]; coordinate position of the point on the surface of the sphere
        """
        return self.r, self.theta, self.phi

    def _u1(self):
        """
        Private method to calculate the [1] component of the linear velocity
        :return: u1; coordinate velocity in [1] direction
        """
        o = 5/2 * self.s / self.rho**2
        u1 = self.rho * o * np.sin(self.P) * np.sin(self.T)
        return u1

    def _u3(self):
        """
        Private method to calculate the [3] component of the linear velocity
        :return: u3; coordinate velocity in [3] direction
        """
        o = 5/2 * self.s / self.rho**2
        return self.rho * o * np.cos(self.P) * np.sin(self.T)

    def _vphi(self):
        """
            (private)
            Calculate the orbital velocity. This subroutine includes all equations necessary.
            :return: float; orbital velocity of the timelike object.
        """
        return self.r0**2 * self.omega / np.sqrt(1 + self.r0**2 * self.omega**2)

    def _vr(self):
        """
            (private)
            Calculate the radial velocity. This subroutine includes all equations necessary.
            Note that this is 0 (most of the times), as only circular motion is considered.
            :return: float; radial velocity of the timelike object.
        """
        root = (self.r0 ** 2 * self.E - self.s / self.r0 * self.L) ** 2 - \
               self.r0 * (self.r0 - 2) * ((self.r0 ** 3 - self.s ** 2) ** 2 / self.r0 ** 4 +
                                          (self.L - self.s * self.E) ** 2)

        if root < 0:
            print(f'warning: root smaller 0; might be ok, root is {root}')  #

        return 0#self.r0 / (self.r0 ** 3 * self.E - self.s * self.L) * np.sqrt(np.abs(root))

    def _math_v(self):
        xv = - 3 * self.s + np.sqrt(4*self.r**3 + 13 * self.s**2 - 8 * self.s**4 / self.r**3)
        xv /= 2 * np.sqrt(self.r**2 - 2*self.r) * (self.r - self.s**2 / self.r**2)

        xu = (self.r - self.s**2 / self.r**2) / (self.r + 2 * self.s**2 / self.r**2) * xv

        return np.sqrt(1 - (1 - xv**2) * (1 - xu**2) / (1 - xv * xu)**2)
