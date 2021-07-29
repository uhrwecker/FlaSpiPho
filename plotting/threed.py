import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class ThreeDPlotmachine:
    """
    Class to plot the 3D geodesics, with come convenience.
    """
    def __init__(self, robs, tobs, pobs, rc, pc, rho):
        """
        Init of the class.
        :param robs: float; radial coordinate of the observer
        :param tobs: float; theta coordinate of the observer
        :param pobs: float; phi coordinate of the observer
        :param rc: float; radial coordinate of the center of the ball
        :param pc: float; phi coordinate of the center of the ball #no theta, as the ball should be in the eq. plane
        :param rho: float; radius of the ball
        """
        self.robs = robs
        self.tobs = tobs
        self.pobs = pobs

        self.rc = rc
        self.pc = pc
        self.rho = rho

        self.figure = pl.figure(figsize=(10, 10))
        self.ax = self.figure.add_subplot(111, projection='3d')

    def plot_observer(self):
        """
        Scatter the observer as a point in the figure at the position of the observer (duh)
        """
        self.ax.scatter(self.robs * np.cos(self.pobs) * np.sin(self.tobs),
                        self.robs * np.sin(self.pobs) * np.sin(self.tobs),
                        self.robs * np.cos(self.tobs))

    def plot_emitter(self):
        """
        Scatter the emitter as a wireframe sphere with the radius of the ball.
        """
        phis = np.linspace(0, 2*np.pi, num=1000)

        # get the local spherical coordinates transformed into local cartesian coordinates
        x0 = self.rc * np.cos(self.pc)
        y0 = self.rc * np.sin(self.pc)
        z0 = 0

        # create meshgrid to describe the phi and theta coordinates in the global coordinates
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]

        # calculate the real cartesian coordinates (r_local + r_0)
        x = x0 + self.rho * np.cos(u) * np.sin(v)
        y = y0 + self.rho * np.sin(u) * np.sin(v)
        z = z0 + self.rho * np.cos(v)

        # plot the emitter sphere
        self.ax.plot_wireframe(x, y, z)

        # plot the emitter orbit
        self.ax.plot(self.rc * np.cos(phis), self.rc * np.sin(phis), 0 * phis)

    def plot_test_ray(self, solver, iota, eta):
        """
        Plotting the light ray as describes the solver object and the emission angles.
        :param solver: solving.solver object; object that inherits all meta data to calculate the light ray
        :param iota: float; phi-like emission angle
        :param eta: float; theta-like emission angle
        """
        solver.set_iota(iota, False)
        solver.set_eta(eta, True)

        _, data = solver.solve()

        self.plot_from_data(data)

    def scatter_test_points(self, r, t, p):
        """
        Scatter test points onto the figure.
        :param r: float; radial coordinate of the test point
        :param t: float; theta coordinate of the test point
        :param p: float; phi coordinate of the test point
        """
        x = r * np.cos(p) * np.sin(t)
        y = r * np.sin(p) * np.sin(t)
        z = r * np.cos(t)

        self.ax.scatter(x, y, z)

    def plot_from_data(self, data):
        """
        Plotting the light ray as described by the data array from ODESolver
        :param data: np.ndarray (8, ); numpy array that contains all data of the light ray.
        """
        r = data[:, 2]
        t = data[:, 4]
        p = data[:, 6]

        x = r * np.cos(p) * np.sin(t)
        y = r * np.sin(p) * np.sin(t)
        z = r * np.cos(t)

        self.ax.plot(x, y, z)

    def plot_from_COM(self, e, l, q, r0, t0, p0, sr=1, st=1, sp=1):
        """
        Plotting the light ray as described by the constants of motion and initial position.
        :param e: float; E constant of motion
        :param l: float; L constant of motion
        :param q: float; Q constant of motion
        :param r0: float; initial radial position of the light ray
        :param t0: float; initial theta position of the light ray
        :param p0: float; initial phi position of the light ray
        :param sr: float; sign of the initial radial velocity
        :param st: float; sign of the initial theta velocity
        :param sp: float; sign of the initial phi velocity
        :return:
        """
        dt = e
        dr = sr * np.sqrt(e**2 - (q + l**2) / r0**2)
        dth = st * np.sqrt(q - l**2 / np.tan(t0)**2) / r0**2
        dp = sp * l / (r0**2 * np.sin(t0)**2)

        from light.geodesics import geod
        from scipy.integrate import odeint

        psi = np.array([0, dt, r0, dr, t0, dth, p0, dp])
        sigma = np.linspace(0, 40, num=1000)

        result = odeint(geod, psi, sigma, atol=1e-7, rtol=1e-7)

        self.plot_from_data(result)

    def adjust(self):
        """
        Semi-auto adjust the limits of the axes.
        """
        self.ax.set_xlim(-self.robs-1, self.robs+1)
        self.ax.set_ylim(-self.robs-1, self.robs+1)
        self.ax.set_zlim(-2/3*self.robs, 2/3*self.robs)

        self.ax.set_xlabel('x / M')
        self.ax.set_ylabel('y / M')
        self.ax.set_zlabel('z / M')

    def show(self):
        """
        (Simple) wrapper of the pyplot.show() feature for when you dont want to import pyplot in your main script.
        """
        pl.show()
