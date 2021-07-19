import numpy as np

from utility.util import convert_position

class EmitterProperties:
    """
        Class containing all emitter properties.
    """
    def __init__(self, s, r0=8, phi0=0, omega=1, P=0, T=np.pi/2, rho=0.2, rotation='positive'):
        """
            :param s: float; spin of the timelike object
            :param r0: float; orbit of the timelike object
            :param rotation: rotation: ['positive', 'negative']; determining equations are oblivious to the sign of L;
            this is determined by the rotation parameter.
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

        self.r0 = r0
        self.phi0 = phi0
        self.omega = omega
        self.E = None
        self.L = None

        # orbit velocities:
        self.vr = None
        self.vphi = None
        self.gamma = None

        self.u1 = None
        self.u3 = None
        self.gamma2 = None

        # real position:
        self.r = None
        self.theta = None
        self.phi = None

        self.setup()

    def setup(self):
        """
            Basic setup function.
        """
        self.r, self.theta, self.phi = convert_position(self.r0, np.pi / 2, self.phi0, self.rho, self.T, self.P)
        self.vr, self.vphi, self.gamma = self.calculate_vel()
        self.u1, self.u3, self.gamma2 = self.calculate_rotation()

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

        :return:
        """
        u1 = self._u1()
        u3 = self._u3()

        return u1, u3, 1/np.sqrt(1 - u1**2 - u3**2)

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
        self.P = P
        if recalc:
            self.setup()

    def set_T(self, T, recalc=True):
        self.T = T
        if recalc:
            self.setup()

    def get_com(self):
        """
            Getter for the constants of motion.
            :return: [E, L]; energy and angular momentum of the orbit.
        """
        return self.E, self.L

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
        return self.u1, self.u3, self.gamma2

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
        return self.r, self.theta, self.phi

    def _u1(self):
        o = 5/2 * self.s / self.rho**2
        u1 = self.rho * o * np.sin(self.P) * np.sin(self.T)
        return u1

    def _u3(self):
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

        return self.r0 / (self.r0 ** 3 * self.E - self.s * self.L) * np.sqrt(np.abs(root))
