import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class ThreeDPlotmachine:
    def __init__(self, robs, tobs, pobs, rc, pc, rho):
        self.robs = robs
        self.tobs = tobs
        self.pobs = pobs

        self.rc = rc
        self.pc = pc
        self.rho = rho

        self.figure = pl.figure(figsize=(10, 10))
        self.ax = self.figure.add_subplot(111, projection='3d')

    def plot_observer(self):
        self.ax.scatter(self.robs * np.cos(self.pobs) * np.sin(self.tobs),
                        self.robs * np.sin(self.pobs) * np.sin(self.tobs),
                        self.robs * np.cos(self.tobs))

    def plot_emitter(self):
        phis = np.linspace(0, 2*np.pi, num=1000)

        x0 = self.rc * np.cos(self.pc)
        y0 = self.rc * np.sin(self.pc)
        z0 = 0

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = x0 + self.rho * np.cos(u) * np.sin(v)
        y = y0 + self.rho * np.sin(u) * np.sin(v)
        z = z0 + self.rho * np.cos(v)

        self.ax.plot_wireframe(x, y, z)

        self.ax.plot(self.rc * np.cos(phis), self.rc * np.sin(phis), 0 * phis)

    def plot_test_ray(self, solver, iota, eta):
        solver.set_iota(iota, False)
        solver.set_eta(eta, True)

        _, data = solver.solve()

        self.plot_from_data(data)

    def scatter_test_points(self, r, t, p):
        x = r * np.cos(p) * np.sin(t)
        y = r * np.sin(p) * np.sin(t)
        z = r * np.cos(t)

        self.ax.scatter(x, y, z)

    def plot_from_data(self, data):
        r = data[:, 2]
        t = data[:, 4]
        p = data[:, 6]

        x = r * np.cos(p) * np.sin(t)
        y = r * np.sin(p) * np.sin(t)
        z = r * np.cos(t)

        self.ax.plot(x, y, z)

    def plot_from_COM(self, e, l, q, r0, t0, p0, sr=1, st=1, sp=1):
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
        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-30, 30)
        self.ax.set_zlim(-20, 20)

    def show(self):
        pl.show()
