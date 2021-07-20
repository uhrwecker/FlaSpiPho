from solving.solver import ODESolver
from solving.eop import EmitterObserverProblem
from plotting.threed import ThreeDPlotmachine
from utility.util import get_spherical_grid

from tqdm import tqdm
import numpy as np
import sys
import contextlib


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


solver = ODESolver()

robs = 15
pobs = 0
tobs = np.pi / 3

_, rc, _, pc = solver.dm.get_input_center_config()
rho, _, _, _ = solver.dm.get_input_sphere_config()

grid = get_spherical_grid(num_rings=20, max_n=20)
print(len(grid))

grid = [(np.pi/2, 0)]

td = ThreeDPlotmachine(robs, tobs, pobs, rc, pc, rho)

for item in tqdm(grid, file=sys.stdout):
    with nostdout():
        solver.set_T_and_P(item[0], item[1])
        td.plot_observer()
        td.plot_emitter()

        #for p in np.linspace(0, np.pi*2):
        #    td.plot_test_ray(solver, p, np.pi/4)
        td.adjust()
        #td.show()
        eop = EmitterObserverProblem(solver, robs, tobs, pobs)
        iota, eta, flag = eop.find_critical_angles(0.1, 2*np.pi, 0.1, np.pi)

        solver.set_eta(eta, False)
        solver.set_iota(iota)
        #collision = solver.save(False)

        td.plot_test_ray(solver, iota, eta)

td.show()