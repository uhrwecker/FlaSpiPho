import numpy as np


class PhotonProperties:
    """
        Class for encapsuling the photon properties.
    """
    def __init__(self, emitter, chi=1, iota=0, eta=np.pi / 2):
        """
            :param emitter: object; em.emitter.EmitterProperties object
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
        rho = self.emitter.rho

        alpha = 5/2 * self.emitter.get_s() / rho**2

        E = self._E(self.chi, self.eta, self.iota, gamma, vphi, gamma2, u1, u3, self.alpha)
        L = self._L(self.chi, self.eta, self.iota, gamma, vphi, gamma2, u1, u3, self.alpha)
        Q = self._Q(self.chi, self.eta, L)

        return E, L, Q

    def calculate_ic(self):
        """
            Routine to calculate the initial velocities for the photon geodesic.
            :return: iterable; [t', r', theta', phi']
        """
        r = self.r#self.emitter.get_r0()

        dt = self.E

        omega = self.Q - self.L ** 2 * (np.cos(self.theta) / np.sin(self.theta)) ** 2
        if omega < 0:
            omega = np.abs(omega)
        dtheta = np.sqrt(omega) / r**2
        if self.eta < np.pi / 2:
            dtheta *= -1
        #dtheta = self.chi / r * np.cos(self.beta)
        dphi = self.L / (r * np.sin(self.theta)) ** 2

        dr = -np.sqrt(self.E ** 2 - (self.Q + self.L ** 2) / r ** 2)
        #print(dr)
        if np.isnan(dr):
            dr = 0
        #dr = self._check_dr_sign(self.alpha)

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

    def _E(self, chi, eta, iota, gamma, vphi, gamma2, u1, u3, alpha):
        d0 = gamma2 * (1 + u1 * np.cos(iota) * np.sin(eta) + u3 * np.sin(iota) * np.sin(eta))
        d1 = -gamma2 * u1 + (1 + gamma2**2 * u1**2 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (gamma2 **2 * u1 * u3 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        d3 = -gamma2 * u1 + (gamma2 **2 * u1 * u3 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (1 + gamma2**2 * u3**2 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        c3 =- np.sin(alpha) * d1 + np.cos(alpha) * d3
        a0 = gamma * (d0 + vphi * c3)

        return chi * a0

    def _L(self, chi, eta, iota, gamma, vphi, gamma2, u1, u3, alpha):
        d0 = gamma2 * (1 + u1 * np.cos(iota) * np.sin(eta) + u3 * np.sin(iota) * np.sin(eta))
        d1 = -gamma2 * u1 + (1 + gamma2 ** 2 * u1 ** 2 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (gamma2 ** 2 * u1 * u3 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        d3 = -gamma2 * u1 + (gamma2 ** 2 * u1 * u3 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (1 + gamma2 ** 2 * u3 ** 2 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        c3 = - np.sin(alpha) * d1 + np.cos(alpha) * d3
        a3 = gamma * vphi * d0 + (1 + gamma**2 * vphi**2 / (1 + gamma)) * c3

        return chi * a3 * self.r * np.sin(self.theta)

    def _Q(self, chi, eta, L):
        return self.r**2 * chi**2 * np.cos(eta)**2 + L**2 / np.tan(self.theta)**2

    def _check_dr_sign(self, alpha=np.pi/2):
        """
            (private)
            Check the sign of r'.
            :param dr: float; derivative of r
            :return: float; r' with proper sign
        """
        u1, u3, gamma2 = self.emitter.get_rotation_velocities()
        iota, eta = self.iota, self.eta

        d1 = -gamma2 * u1 + (1 + gamma2 ** 2 * u1 ** 2 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (gamma2 ** 2 * u1 * u3 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        d3 = -gamma2 * u1 + (gamma2 ** 2 * u1 * u3 / (1 + gamma2)) * np.cos(iota) * np.sin(eta) + \
             (1 + gamma2 ** 2 * u3 ** 2 / (1 + gamma2)) * np.sin(iota) * np.sin(eta)
        c1 = np.cos(alpha) * d1 + np.sin(alpha) * d3

        return self.chi * c1
