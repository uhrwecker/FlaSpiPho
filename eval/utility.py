import os
import numpy as np
from os.path import isfile, join

from eval.data_reading import DataPoint
from light.geodesics import geod
from scipy.integrate import odeint


def load_data_points(directory='./'):
    files = [join(directory, f) for f in os.listdir(directory) if isfile(join(directory, f))]
    return [DataPoint(join(directory, f)) for f in files if f.endswith('.ini')]


def get_coordinates_from_vars(q, l, robs, tobs):
    pr = np.sqrt(1 - (q + l**2) / robs)
    pphi = l / (robs * np.sin(tobs))
    ptheta = np.sqrt(q - l**2 / np.tan(tobs)**2) / robs

    alpha = robs * pphi / pr
    beta = robs * ptheta / pr

    return alpha, beta


def get_coordinates(dp):
    pr = np.sqrt(dp.e**2 - (dp.q + dp.l**2) / dp.robs**2)
    pphi = dp.l / (dp.robs * np.sin(dp.tobs))
    ptheta = np.sqrt(dp.q - dp.l**2 / np.tan(dp.tobs)**2) / dp.robs

    alpha = dp.robs * pphi / pr
    beta = dp.robs * ptheta / pr

    return alpha, beta


def get_redshift(dp):
    Eobs = dp.e
    a = -(dp.gamma * dp.gamma2 * (1 + dp.vphi * dp.u3)) * dp.dt
    b = dp.gamma2 * dp.u1 * dp.dr
    c = 1 / dp.r0 * (dp.gamma * dp.gamma2 * dp.vphi + dp.gamma2 * dp.u3 *
                     (1 + dp.gamma**2 * dp.vphi**2 / (1 + dp.gamma))) * dp.l

    Eem = - (a + b + c)

    return Eobs / Eem


def lamda(r, t, al, be):
    return al * np.sin(t) * np.sqrt(r**2 / (be**2 + al**2 + r**2))


def qu(r, t, al, be):
    l = al * np.sin(t) * np.sqrt(r**2 / (be**2 + al**2 + r**2))
    return l**2 * (be**2 / (al**2 * np.sin(t)**2) + 1 / np.tan(t)**2)


def get_proper_matrix(dp_list):
    ls = [dp.l / dp.e for dp in dp_list]
    qs = [dp.q / dp.e**2 for dp in dp_list]

    sr = -1
    st = 1
    sp = -1

    robs = dp_list[0].robs
    tobs = dp_list[0].tobs
    pobs = dp_list[0].pobs

    gg = []

    aleft, bleft = -1.4, 4#get_coordinates_from_vars(qs[0], ls[0], robs, tobs)
    aright, bright = 1.4, 6.8#get_coordinates_from_vars(qs[-1], ls[-1], robs, tobs)

    alphas = np.linspace(aleft, aright, num=100)
    betas = np.linspace(bleft, bright, num=100)

    for b in betas:
        row = []
        for a in alphas:
            l = lamda(robs, tobs, a, b)
            q = qu(robs, tobs, a, b)

            dt = 1
            dr = sr * np.sqrt(1 - (q + l ** 2) / robs ** 2)
            dth = st * np.sqrt(q - l ** 2 / np.tan(tobs) ** 2) / robs ** 2
            dp = sp * l / (robs ** 2 * np.sin(tobs) ** 2)

            psi = np.array([0, dt, robs, dr, tobs, dth, pobs, dp])
            sigma = np.linspace(0, 40, num=1000)

            data = odeint(geod, psi, sigma, atol=1e-7, rtol=1e-7)

            r = data[:, 2]
            t = data[:, 4]
            p = data[:, 6]

            x = r * np.cos(p) * np.sin(t)
            y = r * np.sin(p) * np.sin(t)
            z = r * np.cos(t)

            xc = dp_list[0].rc * np.cos(dp_list[0].pc)
            yc = dp_list[0].rc * np.sin(dp_list[0].pc)
            zc = 0

            dist = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
            tol = 1e-2
            flag = bool(dist[dist <= (dp_list[0].rho - tol)].size)

            if flag:
                dist = np.array([np.sqrt((dp.l / dp.e - l)**2 + (dp.q / dp.e**2 - q)**2) for dp in dp_list])
                idx = np.where(dist == np.amin(dist))[0][0]
                g = get_redshift(dp_list[idx])
            else:
                g = np.nan

            row.append(g)
        print(np.where(betas == b)[0][0])
        gg.append(row)

    return alphas, betas, np.array(gg)

        #import matplotlib.pyplot as pl

        #figure = pl.figure(figsize=(10, 10))
        #ax = figure.add_subplot(111, projection='3d')

        #ax.plot(x, y, z)
        #ax.scatter(x[0], y[0], z[0])
        #ax.scatter(dp_list[0].rc * np.cos(dp_list[0].pc),
        #           dp_list[0].rc * np.sin(dp_list[0].pc),
        #           0)
        #ax.set_xlim(-30, 30)
        #ax.set_ylim(-30, 30)
        #ax.set_zlim(-20, 20)

    #pl.show()


def interpolate(a, b, g, xlim=(-1.35, 1.35), ylim=(4, 6.7)):
    import scipy.interpolate as si
    f = si.interp2d(a, b, g, kind='linear')

    new_x = np.linspace(*xlim, num=1000)
    new_y = np.linspace(*ylim, num=1000)
    new_g = f(new_x, new_y)

    return new_x, new_y, new_g
