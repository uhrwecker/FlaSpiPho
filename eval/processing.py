import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mp

from eval.utility import *


pl.figure(figsize=(10, 10))

dps = load_data_points('/home/jan-menno/Data/28_07_2021/s0/')
print(dps)
#alphas, betas, gs = get_proper_matrix(dps)

a = []
b = []
g = []

for dp in dps:
    alpha, beta = get_coordinates(dp)
    a.append(alpha)
    b.append(beta)
    g.append(get_redshift(dp))#

    #pl.scatter(alpha, beta)

cmap = pl.cm.cool_r
gmin, gmax = np.amin(np.array(g)), np.amax(np.array(g))
norm = mp.colors.Normalize(np.amin(gmin), np.amax(gmax))

for alpha, beta, gg in zip(a, b, g):
    pl.scatter(alpha, beta, color='black', s=2)
print(g)

a, b, g = interpolate(a, b, g)
tol = 0.0
#g[g < (gmin-tol)] = np.nan
mapp = pl.imshow(g, extent=(-1.35, 1.35, -6.7, -4), cmap=cmap, norm=norm)
pl.colorbar(mapp)
pl.xlim(-1.35, 1.35)
pl.xlabel(r'$\alpha$')
pl.ylabel(r'$\beta$')
pl.ylim(-6.7, -4)
#pl.show()
#pl.imshow(gs, extent=(alphas[0], alphas[-1], betas[0], betas[-1]), cmap=cmap, norm=norm, interpolation='bilinear')

#pl.xlim(alphas[0], alphas[-1])
#pl.ylim(betas[0], betas[-1])

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