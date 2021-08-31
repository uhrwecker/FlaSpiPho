from solving.solver import ODESolver
from solving.eop import EmitterObserverProblem
from plotting.threed import ThreeDPlotmachine
from utility.util import get_spherical_grid

from tqdm import tqdm
import numpy as np
import sys
import contextlib
import argparse

class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout

file = './demo_input.ini'
iotas = np.linspace(0, 2*np.pi, num=25)
eta = 0.7165623776987919
iota = 1.5
rings = 5
n = 3

solver = ODESolver(file)

robs, tobs, pobs = solver.dm.get_input_observer_config()
_, rc, _, pc = solver.dm.get_input_center_config()
rho, _, _, _ = solver.dm.get_input_sphere_config()

td = ThreeDPlotmachine(robs, tobs, pobs, rc, pc, rho)

eop = EmitterObserverProblem(solver)

td.plot_observer()
td.plot_emitter()
grid = get_spherical_grid(rings, n)
grid = [(0.7330382858376183, 4.319689898685965)]
for item in tqdm(grid, file=sys.stdout):
    with nostdout():
        solver.set_T_and_P(item[0], item[1])
        eop = EmitterObserverProblem(solver)
        iota, eta, flag = eop.find_critical_angles(0, 2*np.pi, 0, np.pi, n=5)
        #iota = 3.6163183517516373
        #eta = 0.57
        solver.set_eta(eta, False)
        solver.set_iota(iota)

        td.plot_test_ray(solver, iota, eta)
        td.scatter_test_points(0, 0, 0)

td.adjust()
td.show()
#- Iota = 4.373178044740911, Eta = 1.1913033092322631.

#[(1.9896753472735356, 3.7233690709212364), (2.0943951023931953, 3.518583772020569)]
