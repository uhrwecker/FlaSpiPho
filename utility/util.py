import numpy as np

def convert_position(r0, t0, p0, rho, T, P):
    # position to center: r0, t0, p0
    # local coords: rho, T, P

    x = r0 * np.cos(p0) * np.sin(t0) + rho * np.cos(P) * np.sin(T)
    y = r0 * np.sin(p0) * np.sin(t0) + rho * np.sin(P) * np.sin(T)
    z = r0 * np.cos(t0) + rho * np.cos(T)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = get_phi(x, y, z)

    return r, theta, phi

def get_phi(x, y, z, tol=1e-4):
    if x > tol:
        return np.arctan(y / x)
    elif -tol < x < tol:
        return np.sign(y) * np.pi / 2
    elif x < tol and y >= 0:
        return np.arctan(y / x) + np.pi
    elif x < tol and y < 0:
        return np.arctan(y / x) - np.pi

def get_spherical_grid(num_rings=5, max_n=10):
    theta = np.linspace(0, np.pi, num=num_rings, endpoint=False)

    vals = []
    for t in theta:
        if t == 0 or t == np.pi:
            vals.append((t, 0))
        else:
            max_phi = int(np.sin(t) * max_n)
            if max_phi >= 1:
                phi = np.linspace(0, 2*np.pi, num=max_phi, endpoint=False)
                vals += [(t, p) for p in phi]

    return vals