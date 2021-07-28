import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mp

from eval.utility import *


pl.figure(figsize=(10, 10))

dps = load_data_points('D:/2021_07_26/data/s01/')
alphas, betas, gs = get_proper_matrix(dps)

a = []
b = []
g = []

for dp in dps:
    alpha, beta = get_coordinates(dp)
    a.append(alpha)
    b.append(beta)
    g.append(get_redshift(dp))#

    pl.scatter(alpha, beta)

cmap = pl.cm.cool_r
norm = mp.colors.Normalize(np.nanmin(gs), np.nanmax(gs))
pl.imshow(gs, extent=(alphas[0], alphas[-1], betas[0], betas[-1]), cmap=cmap, norm=norm, interpolation='bilinear')

pl.xlim(alphas[0], alphas[-1])
pl.ylim(betas[0], betas[-1])

pl.show()
#gmin, gmax = np.amin(np.array(g)), np.amax(np.array(g))
#a, b, g = interpolate(a, b, g)
#tol = 0.03
#g[g < (gmin-tol)] = np.nan
#cmap = pl.cm.cool_r
#norg = g.flatten()
#norm = mp.colors.Normalize(np.amin(gmin), np.amax(gmax))
#pl.imshow(g, extent=(-1.35, 1.35, 4, 6.7), cmap=cmap, norm=norm)

#pl.xlim(-1.35, 1.35)
#pl.ylim(4, 6.7)
#pl.show()