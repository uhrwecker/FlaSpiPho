import numpy as np
import matplotlib.pyplot as pl

def vel_small_s(r, s):
    sigma = s / r
    return 6 * sigma / (r**2 - 2*r - 1)

r = np.linspace(2, 20, num=10000)

s = 0.1
pl.plot(r, vel_small_s(r, s))
pl.grid()
pl.xlim(2, 20)
pl.ylim(-0.1, 1.1)


pl.show()