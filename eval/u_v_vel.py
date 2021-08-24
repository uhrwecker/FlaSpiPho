import numpy as np
import matplotlib.pyplot as pl


def v(r, s):
    xv = (-3 * s + np.sqrt(4*r**3 + 13 * s**2 - 8*s**4 / r**3)) / (2 * np.sqrt(r**2 - 2*r) * (r - s**2 / r**2))
    xu = (r - s**2/r**2) / (r + 2*s**2/r**2) * xv

    return - (1 - xv**2) * (1 - xu**2) / (1 - xv * xu)**2


def v3(r, s):
    xv = (-3 * s + np.sqrt(4*r**3 + 13 * s**2 - 8*s**4 / r**3)) / (2 * np.sqrt(r**2 - 2*r) * (r - s**2 / r**2))

    return xv

fig = pl.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for s in [-1, -0.1, 0, 0.1, 1]:
    r = np.linspace(0, 20, num=10000)
    ax1.plot(r, 1+v(r, s), label=f's = {s}')
    ax2.plot(r, 1+v(r, s), label=f's = {s}')
    ax3.plot(r, 1+v(r, s), label=f's = {s}')
    ax4.plot(r, 1+v(r, s), label=f's = {s}')




ax1.legend()
ax1.set_xlim(2, 20)
ax1.grid()
ax1.set_xlabel('r / M')
ax1.set_ylabel('V^2')
ax1.axhline(1, c='gray', ls='--', alpha=0.8)

ax1.set_ylim(-0.1, 1.1)

ax2.legend()
ax2.set_xlim(2, 5)
ax2.grid()
ax2.set_xlabel('r / M')
ax2.set_ylabel('V^2')
ax2.axhline(1, c='gray', ls='--', alpha=0.8)

ax2.set_ylim(-0.1, 1.1)

ax3.legend()
ax3.set_xlim(5, 15)
ax3.grid()
ax3.set_xlabel('r / M')
ax3.set_ylabel('V^2')
ax3.axhline(1, c='gray', ls='--', alpha=0.8)

ax3.set_ylim(-0.000001, 0.00001)


ax4.legend()
ax4.set_xlim(5, 15)
ax4.grid()
ax4.set_xlabel('r / M')
ax4.set_ylabel('V^2')
ax4.axhline(1, c='gray', ls='--', alpha=0.8)

ax4.set_ylim(-0.0000000001, 0.000000001)


pl.show()