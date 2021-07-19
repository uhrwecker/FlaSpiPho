import numpy as np
import time

from scipy.interpolate import interp1d


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])


class EmitterObserverProblem:
    def __init__(self, solver, r_obs, theta_obs, phi_obs):
        self.solver = solver
        self.robs = r_obs
        self.thetaobs = theta_obs
        self.phiobs = phi_obs

    def find_critical_angles(self, imin=None, imax=None, emin=None, emax=None, n=15, max_step=40):
        start = time.time()

        #tqdm.write(f'Algo for phi0 = {self.solver.phi0} and theta0 = {self.solver.theta0}')
        print(f'Algo for phi0 = {self.solver.phi0} and theta0 = {self.solver.theta0}')

        if emin and emax:
            iota, eta, flag = self.bad_montecarlo(imin, imax, emin, emax, n=n, max_step=max_step)

        now = time.time()
        print(f'Finding the critical angle took {now-start}s.\n')

        return iota, eta, flag

    def bad_montecarlo(self, imin, imax, emin, emax, n, max_step):
        converged = False
        step = 0
        incr = 0.01

        result_iota = None
        result_eta = None

        last_smallest_distance = 10000

        while not converged and step < max_step:
            parameters = [[(i, e) for i in np.linspace(imin, imax, endpoint=True, num=n)] for e in
                          np.linspace(emin, emax, endpoint=True, num=n)]
            imin, imax, emin, emax, flag, dist = self._generate_solutions(parameters, np.abs(imax - imin) / n,
                                                                    np.abs(emax - emin) / n, step)

            if flag:
                print(f'Converged at step {step} / {max_step}!')
                print(f'- Iota = {imin}, Eta = {emin}.')
                result_iota = imin
                result_eta = emin
                converged = True
                break

            if step % 5 == 0:
                print(f'- now at step {step} / {max_step}.')
                print(f'-- iota between {imin} and {imax}.')
                print(f'-- eta  between {emin} and {emax}')

            if np.abs(1 - last_smallest_distance / dist) < 1e-3:
                print(f'-- the algorithm did not converge close enough; trying again once ...')
                imin -= incr
                imax += incr
                emin -= incr
                emax += incr

                n += 5

            step += 1
            if step == max_step:
                print('The algorithm did not converge fast enough.\n')

            last_smallest_distance = dist

        return result_iota, result_eta, converged

    def _generate_solutions(self, params, di, de, step):
        smallest = 1e10
        iesmall = (0, 0)
        tol = 1e-5

        for fixed_eta in params:
            for i, e in fixed_eta:
                self.solver.set_iota(i, False)
                self.solver.set_eta(e)

                sigma, data = self.solver.solve()

                r = data[:, 2]
                t = data[:, 4]
                p = data[:, 6]

                r, t, p = self._interpolate_around_data(r, t, p, sigma)

                dist = self._get_minimal_distance_to_observer(r, t, p)
                if dist < tol:
                    return i, None, e, None, True, None

                if dist < smallest:
                    iesmall = i, e
                    smallest = dist

        if step > 0:
            #tqdm.write(f'--- smallest distance: {smallest} for iota = {iesmall[0]} and eta = {iesmall[1]}')
            print(f'--- smallest distance: {smallest} for iota = {iesmall[0]} and eta = {iesmall[1]}')
        return iesmall[0] - di, iesmall[0] + di, iesmall[1] - de, iesmall[1] + de, False, smallest


    def _interpolate_around_data(self, r, t, p, sigma):
        idx = np.where(r[r < self.robs + 0.1] > self.robs - 0.1)[0]

        s = sigma[idx]
        new_x = np.linspace(s[0], s[-1], num=10000)

        r = np.interp(new_x, s, r[idx])
        t = np.interp(new_x, s, t[idx])
        p = np.interp(new_x, s, p[idx])

        return r, t, p


    def _get_minimal_distance_to_observer(self, r, t, p):
        x = r * np.cos(p) * np.sin(t)
        y = r * np.sin(p) * np.sin(t)
        z = r * np.cos(t)

        X = self.robs * np.cos(self.phiobs) * np.sin(self.thetaobs)
        Y = self.robs * np.sin(self.phiobs) * np.sin(self.thetaobs)
        Z = self.robs * np.cos(self.thetaobs)

        return np.amin(np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2))
