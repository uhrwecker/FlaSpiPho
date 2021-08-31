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


def parsing_arguments(argus):
    parser = argparse.ArgumentParser(description='Calculating light rays from spinning ball in flat spacetime.')

    parser.add_argument('input_file', type=str, help='File path to the designated input file. See demo_input.ini for an example.')
    parser.add_argument('-s', '--save', action='store_true', default=False, help='Save the data produced.')
    parser.add_argument('-swc', '--save_when_colliding', action='store_true', default=False,
                        help='Save the light ray even if it collides with the ball. Must use the --save flag.')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help='Plot the resulting light ray. Only when computing one light ray.')
    parser.add_argument('-g', '--grid', nargs=2, type=int, default=[0, 0],
                        help='Supply two integers (number_of_rings, number_of_points) for a spherical grid. See the '
                             'documentation for more info.')

    if argus:
        return parser.parse_args(argus)
    else:
        return parser.parse_args()


def main(arguments=None):
    cli_args = parsing_arguments(arguments)
    file = cli_args.input_file
    save = cli_args.save
    save_when_colliding = cli_args.save_when_colliding
    plot = cli_args.plot
    rings, n = cli_args.grid

    solver = ODESolver(file)

    robs, tobs, pobs = solver.dm.get_input_observer_config()
    _, rc, _, pc = solver.dm.get_input_center_config()
    rho, _, _, _ = solver.dm.get_input_sphere_config()

    td = ThreeDPlotmachine(robs, tobs, pobs, rc, pc, rho)

    eop = EmitterObserverProblem(solver)

    if rings and n:
        liste = []
        grid = get_spherical_grid(rings, n)
        for item in tqdm(grid, file=sys.stdout):
            with nostdout():
                solver.set_T_and_P(item[0], item[1])

                try:
                    if not solver.check_duplicate_saving():
                        eop = EmitterObserverProblem(solver)
                        iota, eta, flag = eop.find_critical_angles(0., 2 * np.pi, 0., np.pi, n=10)

                        solver.set_eta(eta, False)
                        solver.set_iota(iota)

                        if save:
                            solver.save(save_when_colliding)

                    else:
                        print('File already exists. Please delete before new calculation, or change the saving directory.')
                except:
                    print('Cant save / calculate this part. See to this later.')
                    liste.append(item)
        solver.dm.write_error_entry(liste)
    else:
        if not solver.check_duplicate_saving():
            iota, eta, flag = eop.find_critical_angles(0., 2 * np.pi, 0., np.pi, n=10)
            solver.set_eta(eta, False)
            solver.set_iota(iota, True)

            if save:
                solver.save(save_when_colliding)

            if plot:
                td.plot_observer()
                td.plot_emitter()
                td.plot_bh()
                td.plot_test_ray(solver, iota, eta)
                td.adjust()
                td.show()

        else:
            print('File already exists. Please delete before new calculation, or change the saving directory.')


if __name__ == '__main__':
    try:
        main()
    except:
        # Your input (when not via CLI):

        fp = './demo_input.ini'
        save = True
        save_when_colliding = False
        grid = 30, 30
        plot = False

        ###################################################################

        fp += ' '

        if save:
            save = '-s '
            if save_when_colliding:
                save_when_colliding = '-swc '
            else:
                save_when_colliding = ''
        else:
            save = ''
            save_when_colliding = ''

        if grid[0] and grid[1]:
            plot = ''
            grid = '-g ' + str(grid[0]) + ' ' + str(grid[1])
            grid += ' '
        else:
            grid = ''

        if plot:
            plot = '-p'
        else:
            plot = ''

        res = fp + save + save_when_colliding + plot + grid
        main(res.split())
