import numpy as np


class PhotonProperties:
    """
        Class for encapsuling the photon properties.
    """
    def __init__(self, emitter, chi=1, iota=0, eta=np.pi / 2):
        """
            :param emitter: object; emitter.EmitterProperties object
            :param chi: float; scaling parameter for the photon momentum, invariant for the application
            :param iota: float; phi-like emission angle
            :param eta: float; theta-like emission angle
        """
        self.emitter = emitter

        self.chi = chi
        self.iota = iota
        self.eta = eta

        # initial position:
        self.r = None
        self.theta = None
        self.phi = None

        # COM:
        self.E = None
        self.L = None
        self.Q = None

        # initial velocities:
        self.dt = None
        self.dr = None
        self.dtheta = None
        self.dphi = None

        self.alpha = 0.

        self.setup()

    def setup(self):
        """
            Simple setup routine.
            :return: position, constants of motion and intitial velocities of the photon
        """
        self.r, self.theta, self.phi = self.emitter.get_position()
        self.E, self.L, self.Q = self.calculate_com()
        self.dt, self.dr, self.dtheta, self.dphi = self.calculate_ic()

        return self.r, self.theta, self.phi, self.E, self.L, self.Q, self.dt, self.dr, self.dtheta, self.dphi

    def calculate_com(self):
        """
            Routine to calculate the constants of motion for the photon geodesic.
            :return: iterable; [E, L, Q]
        """
        vr, vphi, gamma = self.emitter.get_velocities()
        u1, u3, gamma2 = self.emitter.get_rotation_velocities()
        math_v, gamma3 = self.emitter.get_momentum_velocity()
        rho = self.emitter.rho

        alpha = 5/2 * self.emitter.get_s() / rho**2

        E = self._E(self.chi, self.eta, self.iota, gamma, vphi, gamma2, u1, u3, math_v, gamma3)
        L = self._L(self.chi, self.eta, self.iota, gamma, vphi, gamma2, u1, u3, math_v, gamma3)
        Q = self._Q(self.chi, self.eta, L)

        return E, L, Q

    def calculate_ic(self):
        """
            Routine to calculate the initial velocities for the photon geodesic.
            :return: iterable; [t', r', theta', phi']
        """
        # dt:
        dt = self.E / (1 - 2 / self.r)

        # dr:
        dr = np.sqrt(self.E ** 2 - (self.Q + self.L ** 2) / self.r ** 2 * (1 - 2 / self.r))
        #print(dr)
        if np.isnan(dr):
            dr = 0
        dr *= self._check_dr_sign()

        # dtheta:
        omega = self.Q - self.L ** 2 * (np.cos(self.theta) / np.sin(self.theta)) ** 2
        if omega < 0:
            omega = np.abs(omega)
        dtheta = np.sqrt(omega) / self.r**2
        if self.eta < np.pi / 2:
            dtheta *= -1

        # dphi:
        dphi = self.L / (self.r * np.sin(self.theta)) ** 2

        return dt, dr, dtheta, dphi

    def get_ic(self):
        """
            Getter for the initial velocities.
            :return: iterable; [t', r', theta', phi']
        """
        return self.dt, self.dr, self.dtheta, self.dphi

    def get_com(self):
        """
            Getter for the constants of motion.
            :return: iterable; [E, L, Q]
        """
        return self.E, self.L, self.Q

    def get_angles(self):
        """
            Getter for the emission angles.
            :return: iterable; [chi, alpha, beta]
        """
        return self.chi, self.iota, self.eta

    def set_iota(self, iota, recalc=True):
        """
            Set (and possibly recalc) the emission angle alpha.
            :param iota: phi-like emission angle
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.iota = iota
        if recalc:
            self.setup()

    def set_eta(self, eta, recalc=True):
        """
            Set (and possibly recalc) the emission angle beta.
            :param eta: theta-like emission angle
            :param recalc: bool; determines if the COM and physical velocities are recalculated for the given s.
        """
        self.eta = eta
        if recalc:
            self.setup()

    def _E(self, chi, eta, iota, gamma, vphi, gamma2, u1, u3, math_v, gamma3):
        """
        Private method to calculate the constant of motion E, as described by the theory.
        :param chi: float; scaling parameter for the photon momentum, invariant for the application
        :param iota: float; phi-like emission angle
        :param eta: float; theta-like emission angle
        :param gamma: float; 1 / sqrt(1 - (vphi)^2)
        :param vphi: float; orbital velocity of the emitter
        :param gamma2: float; 1 / sqrt(1 - (u1)^2 - (u3)^2)
        :param u1: float; linear velocity in [1] direction, resulting from angular velocity
        :param u3: float; linear velocity in [3] direction, resulting from angular velocity
        :param alpha: float; deprecated
        :return: E; constant of motion
        """
        c0 = gamma2 * (1 + u1 * np.cos(iota) * np.sin(eta) + u3 * np.sin(iota) * np.sin(eta))
        #c1 = -gamma2 * u1 + (1 + gamma2**2 * u1**2 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
        #     (gamma2 **2 * u1 * u3 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        c3 = -gamma2 * u1 + (gamma2 **2 * u1 * u3 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (1 + gamma2**2 * u3**2 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)

        b0 = gamma3 * (c0 + math_v * c3)
        b3 = gamma3 * math_v * c0 + (1 + gamma3**2 * math_v**2 / (1 + gamma3)) * c3

        a0 = gamma * (b0 + vphi * b3)

        return chi * a0 * np.sqrt(1 - 2 / self.r)

    def _L(self, chi, eta, iota, gamma, vphi, gamma2, u1, u3, math_v, gamma3):
        """
        Private method to calculate the constant of motion L, as described by the theory.
        :param chi: float; scaling parameter for the photon momentum, invariant for the application
        :param iota: float; phi-like emission angle
        :param eta: float; theta-like emission angle
        :param gamma: float; 1 / sqrt(1 - (vphi)^2)
        :param vphi: float; orbital velocity of the emitter
        :param gamma2: float; 1 / sqrt(1 - (u1)^2 - (u3)^2)
        :param u1: float; linear velocity in [1] direction, resulting from angular velocity
        :param u3: float; linear velocity in [3] direction, resulting from angular velocity
        :param alpha: float; deprecated
        :return: L; constant of motion
        """
        c0 = gamma2 * (1 + u1 * np.cos(iota) * np.sin(eta) + u3 * np.sin(iota) * np.sin(eta))
        # c1 = -gamma2 * u1 + (1 + gamma2**2 * u1**2 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
        #     (gamma2 **2 * u1 * u3 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        c3 = -gamma2 * u1 + (gamma2 ** 2 * u1 * u3 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (1 + gamma2 ** 2 * u3 ** 2 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)

        b0 = gamma3 * (c0 + math_v * c3)
        b3 = gamma3 * math_v * c0 + (1 + gamma3 ** 2 * math_v ** 2 / (1 + gamma3)) * c3

        a3 = gamma * vphi * b0 + (1 + gamma**2 * vphi**2 / (1 + gamma)) * b3

        return chi * a3 * self.r * np.sin(self.theta)

    def _Q(self, chi, eta, L):
        """
        Private method to calculate the constant of motion Q, as described by the theory.
        :param chi: float; scaling parameter for the photon momentum, invariant for the application
        :param eta: float; theta-like emission angle
        :param L: float; constant of motion
        :return: Q; constant of motion
        """
        return self.r**2 * chi**2 * np.cos(eta)**2 + L**2 / np.tan(self.theta)**2

    def _check_dr_sign(self, alpha=np.pi/2):
        """
            (private)(deprecated)
            Check the sign of r'.
            :param dr: float; derivative of r
            :return: float; r' with proper sign
        """
        u1, u3, gamma2 = self.emitter.get_rotation_velocities()
        iota, eta = self.iota, self.eta

        c1 = -gamma2 * u1 + (1 + gamma2 ** 2 * u1 ** 2 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (gamma2 **2 * u1 * u3 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)

        return np.sign(self.chi * c1 / np.sqrt(1 - 2/self.r))
