import numpy as np
import time


class EmitterObserverProblem:
    """
    Class that encapsules the emitter-observer-problem - meaning, finding the emission angles that result in
    a light ray that connects emitter and observer.
    """
    def __init__(self, solver):
        """
        :param solver: solving.solver object; solver that already inherits the emitter and photon properties
        """
        self.solver = solver
        self.robs, self.thetaobs, self.phiobs = self.solver.dm.get_input_observer_config()

    def find_critical_angles(self, imin=0., imax=2*np.pi, emin=0., emax=np.pi, n=15, max_step=40):
        """
        Main routine to find the critical emission angles that result in a light ray that connects emitter and observer.
        See bad_montecarlo for a specificaton of n and max_step.
        :param imin: float; lower limit of the phi-like emission angle
        :param imax: float; upper limit of the phi-like emission angle
        :param emin: float; lower limit of the theta-like emission angle
        :param emax: float; upper limit of the theta-like emission angle
        :param n: int; dimension of montecarlo iteration matrix entries
        :param max_step: int; max number of iteration steps
        :return: [iota, eta, flag]; emission angles and a flag whether the iteration converges.
        """
        # time for finding the computation time:
        start = time.time()

        print(f'Algo for phi0 = {self.solver.phi0} and theta0 = {self.solver.theta0}')

        # main emission angle routine:
        iota, eta, flag = self.bad_montecarlo(imin, imax, emin, emax, n=n, max_step=max_step)

        now = time.time()
        print(f'Finding the critical angle took {now-start}s.\n')

        return iota, eta, flag

    def bad_montecarlo(self, imin, imax, emin, emax, n, max_step):
        """
        Main routine for computing the emission angles. The idea is the following:
        Make a nxn grid for all combinations of (iota, eta) in between (i/e)min and (i/e)max.
        For each pair of emission angles, compute the light ray and compute the minimal distance to the observer
           (interpolate if necessary).
        Take the pair of emission angle that has the least distance. If the distance is below a certain threshold,
           you have a hit! Otherwise,
        Take another iteration for values of iota/eta that are close to the least_distance_emission_angles.
        :param imin: float; lower limit of the phi-like emission angle
        :param imax: float; upper limit of the phi-like emission angle
        :param emin: float; lower limit of the theta-like emission angle
        :param emax: float; upper limit of the theta-like emission angle
        :param n: int; dimension of montecarlo iteration matrix entries
        :param max_step: int; max number of iteration steps
        :return: [iota, eta, flag]; emission angles and a flag whether the iteration converges.
        """

        # initial parameters:
        converged = False
        step = 0
        incr = 0.5

        result_iota = None
        result_eta = None

        # overreact:
        last_smallest_distance = 10000

        while not converged and step < max_step:
            # initialize the parameter matrix:
            parameters = [[(i, e) for i in np.linspace(imin, imax, endpoint=True, num=n)] for e in
                          np.linspace(emin, emax, endpoint=True, num=n)]

            # generate the boundaries of iota and eta, and a flag whether it converged, and the least distance
            imin, imax, emin, emax, flag, dist = self._generate_solutions(parameters, np.abs(imax - imin) / n,
                                                                    np.abs(emax - emin) / n, step)

            if flag:
                print(f'Converged at step {step} / {max_step}!')
                print(f'- Iota = {imin}, Eta = {emin}.')
                result_iota = imin
                result_eta = emin
                converged = True
                break

            # log every 5 steps:
            if step % 5 == 0:
                print(f'- now at step {step} / {max_step}.')
                print(f'-- iota between {imin} and {imax}.')
                print(f'-- eta  between {emin} and {emax}')

            # if the algorithm did not converge fast enough:
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
        """
        This is where the (dark) magic happens.
        Here, calculate for every pair of (iota;eta) in params matrix the light ray and the minimal distance to the
        observer. If there is no hit, return thenew boundaries for iota and eta.
        :param params: list (n, n); nxn matrix of pairs of emission angles
        :param di: float; distance between the iota values
        :param de: float; distance between the eta values
        :param step: int; current step of the algorithm
        :return: [imin, imax, emin, emax, flag, smallest_distance]
        """
        smallest = 1e10
        iesmall = (0, 0)
        tol = 1e-5

        for fixed_eta in params:
            for i, e in fixed_eta:
                # solve for pair of emission angle:
                self.solver.set_iota(i, False)
                self.solver.set_eta(e)

                sigma, data = self.solver.solve()

                r = data[:, 2]
                t = data[:, 4]
                p = data[:, 6]

                if r[r < 2.].shape[0]:
                    continue

                # interpolate around the observer position:
                r, t, p = self._interpolate_around_data(r, t, p, sigma)

                # compute the minimal distance:
                dist = self._get_minimal_distance_to_observer(r, t, p)

                if dist < tol:
                    return i, None, e, None, True, None

                if dist < smallest:
                    iesmall = i, e
                    smallest = dist

        if step > 0:
            print(f'--- smallest distance: {smallest} for iota = {iesmall[0]} and eta = {iesmall[1]}')

        return iesmall[0] - di, iesmall[0] + di, iesmall[1] - de, iesmall[1] + de, False, smallest

    def _interpolate_around_data(self, r, t, p, sigma):
        """
        Routine to interpolate the ODESolver data around the observer position.
        :param r: np.array; array of radial positions along the light ray
        :param t: np.array; array of theta positions along the light ray
        :param p: np.array; array of phi positions along the light ray
        :param sigma: np.array; array of affine parameter along the light ray
        :return: [r, t, p]; tuple of interpolated data
        """
        idx = np.where(r[r < self.robs + 0.1] > self.robs - 0.1)[0]

        s = sigma[idx]
        try:
            new_x = np.linspace(s[0], s[-1], num=10000)
        except:
            print(f'Somehow, the number of values are too small for {r[idx]}.')
            print(self.solver.r0, self.solver.theta0, self.solver.phi0)
            print(self.solver.emitter.T, self.solver.emitter.P)
            raise KeyError

        r = np.interp(new_x, s, r[idx])
        t = np.interp(new_x, s, t[idx])
        p = np.interp(new_x, s, p[idx])

        return r, t, p

    def _get_minimal_distance_to_observer(self, r, t, p):
        """
        Routine to calculate the minimal distance to the observer position
        :param r: np.array; array of radial positions along the light ray
        :param t: np.array; array of theta positions along the light ray
        :param p: np.array; array of phi positions along the light ray
        :return: float; return the minimal (euclidean) distance to the observer
        """
        x = r * np.cos(p) * np.sin(t)
        y = r * np.sin(p) * np.sin(t)
        z = r * np.cos(t)

        X = self.robs * np.cos(self.phiobs) * np.sin(self.thetaobs)
        Y = self.robs * np.sin(self.phiobs) * np.sin(self.thetaobs)
        Z = self.robs * np.cos(self.thetaobs)

        return np.amin(np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2))
