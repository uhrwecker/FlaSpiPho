import numpy as np


def convert_position(r0, t0, p0, rho, T, P):
    """
    Convert the local spherical coordinates into global spherical coordinates
    """
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
    """
    From cartesian x, y, z coordinates, get the angle phi.
    """
    if x > tol:
        return np.arctan(y / x)
    elif -tol < x < tol:
        return np.sign(y) * np.pi / 2
    elif x < tol and y >= 0:
        return np.arctan(y / x) + np.pi
    elif x < tol and y < 0:
        return np.arctan(y / x) - np.pi


def get_spherical_grid(num_rings=5, max_n=10):
    """
    Calculate a spherical grid from number of rings and points on the ring.
    """
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


def find_nearest(array, value):
    """
    (deprecated)
    Find the index where an array is closest to a specific *value*.
    :param array: np.array; array to be observed
    :param value: float; value which should be close(st).
    :return: int; index of original array that is the closest to the given value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_indices_of_k_smallest(arr, k):
    """
    (deprecated)
    Find the indices of the k smallest entries in an array.
    :param arr: np.array; array to be observed
    :param k: int; number of smallest entries to be retrieved
    :return: [ind]; returns a list of indices
    """
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])