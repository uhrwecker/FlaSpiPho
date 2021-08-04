import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mp

from eval.utility import *


def get_coords(dp):
    dr = np.sqrt(dp.e**2 - (dp.q + dp.l**2)/dp.robs**2)
    dphi = dp.l / (dp.robs**2 * np.sin(dp.tobs)**2)
    theta = np.arccos(dr / dp.e)

    phi = np.arcsin(dp.robs * np.sin(dp.tobs) * dphi / dp.e)

    return theta, phi


fig = pl.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

dps = load_data_points('/home/jan-menno/Data/28_07_2021/s01/')

for dp in dps:
    t, p = get_coords(dp)
    ax.scatter(np.cos(p)*np.sin(t), np.sin(p)*np.sin(t), np.cos(t))

pl.show()