from typing import List
from functools import partial
import warnings

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from tqdm import tqdm

from grid import Grid
from particle import DynamicParticle, Particle, SpeedOfLightError


class Simulator:
    def __init__(self, particles: List[Particle], dt: float = .01,
                 x_max: float = None, y_max: float = None, n_x: int = None, n_y: int = None):
        self.particles = particles
        self.num_particles = len(particles)

        if self.num_particles < 1:
            raise ValueError("Simulator needs at least 1 particle")

        t = particles[0].t_curr
        for particle in particles:
            if particle.t_curr != t:
                raise ValueError("All particles must be at the same time!")

        self.t = t
        self.dt = dt
        self.times = [0]  #only used for plotting and stuff like that

        self.charges = np.array([particle.charge for particle in self.particles])
        self.masses = np.array([particle.mass for particle in self.particles])
        self.positions = np.array([particle.get_r() for particle in self.particles])
        self.velocities = np.array([particle.get_v() for particle in self.particles])
        self.gammas = np.array([particle.get_gamma() for particle in self.particles])

        self.paths = [self.positions]  #note that the first index is timestep, followed by particle
        self.Bs, self.Es = self._calc_fields()

        self.x_max, self.y_max, self.n_x, self.n_y = x_max, y_max, n_x, n_y

    def __str__(self):
        np.set_printoptions(precision=4)
        return f"simulator at current time {self.t}: with particles \n{chr(10).join([str(p) for p in self.particles])}"

    ################
    ##main methods##
    ################

    def run(self, tmax: float, min_dt = 10e-4, max_dt: float = None, error_tolerance = 10e-5) -> None:
        '''
        Runs the simulator until tmax is reached or adaptive timestep would be less than min_dt
        :param tmax: Time after which the simulation will be terminated
        :param min_dt: Minimum timestep allowed. If adaptive timestep would be shorter, the simulation is terminated
        :param max_dt: Maximum timestep allowed. If adaptive timestep would be longer, this value is used instead
        :param error_tolerance: Maximum allowed error of relative particle positions in units/second
        :return: None
        '''
        pass

    def animated_fields_plot(self, fps: float, num_frames: int, t_i: float = 0, filepath: str = None, show: bool = True,
                             plot_B: bool = True, plot_phi: bool = False, plot_E: bool = True) -> None:
        '''
        Allows the user to save and show an animated field plot from t_i to the current simulation time.
        :param fps: frames per second of output plot
        :param num_frames: number of animation frames to generate
        :param t_i: initial time t to be plotted
        :param filepath: if included, saves the plot to the given filepath Must end in .mp4
        :param show: if true, shows the plot
        :param plot_phi: if true, equipotential lines are plotted
        '''
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.y_max)
        ax.set_aspect('equal')
        scale = 8 / max(self.x_max, self.y_max)
        fig.set_size_inches(scale * self.x_max, scale * self.y_max)

        interval = 1000 / fps
        times = np.linspace(t_i, self.t, num_frames)

        all_artists = []
        for t in tqdm(times, leave=True, position=0, desc='plotting', colour="green"): #for loop with progress bar
            frame_artists = self._plot_fields(t, ax, plot_B=plot_B, plot_phi=plot_phi, plot_E=plot_E, E_streamplot=False)

            all_artists.append(frame_artists)

        print("animating...")
        ani = anim.ArtistAnimation(fig, all_artists, interval=interval, blit=False)

        if filepath:
            try:
                print("saving to video...")
                writer = anim.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=9000)
            except FileNotFoundError:
                raise Exception("Install FFMPEG!")
            ani.save(filepath, dpi=300, writer=writer)

        if show: plt.show()

    def positions_plot(self, filepath: str = None, show: bool = True, color_by_dt = False) -> None:
        '''
        Allows the user to save and show a plot of particle positions.
        :param filepath: filepath at which plot would be saved
        :param show: boolean that determines whether to show plot or not
        :return: None
        '''
        fig, ax = plt.subplots()

        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.y_max)
        ax.set_aspect('equal')

        paths = np.array(self.paths).transpose((1, 0, 2))

        t_cmap = plt.get_cmap("cividis")
        t_norm, t_data = None, None

        charge_cmap = plt.get_cmap("bwr")
        charge_norm = colors.CenteredNorm()

        dts = np.diff(self.times).tolist()
        dts.append(self.dt)

        if color_by_dt:
            t_norm = colors.LogNorm(vmin=min(dts), vmax=max(dts))
            t_data = dts
            plt.gcf().colorbar(plt.cm.ScalarMappable(norm=t_norm, cmap=t_cmap), ax=plt.gca(),
                               orientation='vertical', label='dt')
        else:
            t_norm = colors.Normalize(vmin=0, vmax=np.max(self.times))
            t_data = self.times
            plt.gcf().colorbar(plt.cm.ScalarMappable(norm=t_norm, cmap=t_cmap), ax=plt.gca(),
                               orientation='vertical', label='time')

        for i, particle in enumerate(self.particles):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.plot(paths[i, :, 0], paths[i, :, 1], c="silver", linewidth=1, zorder=0)
                plt.scatter(paths[i, :, 0], paths[i, :, 1], s=1.5, c=t_cmap(t_norm(t_data)), zorder=5)
                plt.scatter(paths[i, -1, 0], paths[i, -1, 1], s=8, c=charge_cmap(charge_norm(self.charges[i])), zorder=10)

        if filepath:
            plt.savefig(filepath, dpi=300)

        if show: plt.show()

    def single_fields_plot(self, t, filepath:str = None, show: bool = True,
                           plot_B: bool = True, plot_phi: bool = False, plot_E: bool = True, E_streamplot: bool = False):
        '''
        Allows the user to save and show a single field plot. Can use E field streamlines, which don't work in animated plots
        :param t: time to be plotted
        :param filepath: filepath at which plot would be saved
        :param show: boolean that determines whether to show plot or not
        :param plot_phi: boolean that determines whether scalar potential contour lines should be plotted
        :param E_streamplot: boolean that determines whether E field will be streamplot if true or vector field if false
        :return: None
        '''
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.y_max)
        ax.set_aspect('equal')
        scale = 8 / max(self.x_max, self.y_max)
        fig.set_size_inches(scale * self.x_max, scale * self.y_max)

        _ = self._plot_fields(0, ax, plot_B=plot_B, plot_phi=plot_phi, plot_E=plot_E, E_streamplot=E_streamplot)

        if filepath: plt.savefig(filepath)
        if show: plt.show()

    ######################
    ##simulation helpers##
    ######################
    def _update_values(self):
        self.t = self.particles[0].t_curr

        self.positions = np.array([particle.get_r() for particle in self.particles])
        self.velocities = np.array([particle.get_v() for particle in self.particles])
        self.gammas = np.array([particle.get_gamma() for particle in self.particles])

        self.Bs, self.Es = self._calc_fields()

    def _calc_distance_vecs(self):
        distances = np.zeros((self.num_particles, self.num_particles, 2))

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                distances[i, j, :] = self.positions[i] - self.positions[j]

        if self.num_particles == 1: #if only one particle, use distance from the origin instead
            distances[0, 0] = self.particles[0].get_r()

        return distances

    ##########
    ##fields##
    ##########
    def _total_B(self, loc, particle=None, t=None, background_B = 0):  #find B field at location excluding a particles own field
        if t is None:
            t = self.t

        if t < 0:
            B = 0
        else:
            B = background_B

        for i, source_charge in enumerate(self.particles):
            if not particle:  #just use the given location
                B += source_charge.B(loc, t)
            elif particle is not source_charge:  #use the given location, but skip the particle at that location
                B += source_charge.B(loc, t)
        return B

    def _total_E(self, loc, particle=None, t=None):  #find E field at location excluding a particles own field
        if t is None:
            t = self.t

        E = np.zeros(2)
        for i, source_charge in enumerate(self.particles):
            if not particle:  #just use the given location
                E += source_charge.E(loc, t)
            elif particle is not source_charge:  #use the given location, but skip the particle at that location
                E += source_charge.E(loc, t)
        return E

    def _total_phi(self, loc, t=None):  #function only used for plotting. Finds the scalar potential at a given location
        if t is None:
            t = self.t

        phi = 0
        for i, source_charge in enumerate(self.particles):
            phi += source_charge.phi(loc, t)
        return phi

    def _calc_fields(self):
        B_vec = np.array([self._total_B(particle.get_r(), particle=particle) for particle in self.particles])
        E_vec = np.array([self._total_E(particle.get_r(), particle=particle) for particle in self.particles])

        return B_vec, E_vec

    ################
    ##plot helpers##
    ################
    def _plot_fields(self, t, ax, plot_B = True, plot_phi = False, plot_E = True, E_streamplot = False, B_range = 2.5):
        title = ax.annotate(f"t = {t:.2f}", (0.05, .95), xycoords="axes fraction", ha="left", fontsize=18)

        B_grid, phi_grid, E_grid = self._calc_gridded_fields(self.x_max, self.y_max, self.n_x, self.n_y, t)
        positions = np.empty((self.num_particles, 2))
        for i, particle in enumerate(self.particles):
            positions[i] = particle.get_r(t)

        artists = []
        if plot_B: artists.append(self._plot_B(B_grid, ax, B_range=B_range))
        if plot_phi: artists.append(self._plot_phi(phi_grid, ax))
        if plot_E: artists.append(self._plot_E(E_grid, ax, gridpoints_per_arrow=4, streamplot=E_streamplot))
        artists.append(self._plot_particles(positions, ax))
        artists.append(title)

        return artists

    def _calc_gridded_fields(self, x_max, y_max, n_x, n_y, t=None):
        if t is None:
            t = self.t

        B_grid = Grid(x_max, y_max, n_x, n_y)
        B_grid.apply(partial(self._total_B, t=t))

        phi_grid = Grid(x_max, y_max, n_x, n_y)
        phi_grid.apply(partial(self._total_phi, t=t))

        E_grid = Grid(x_max, y_max, n_x, n_y)
        E_grid.apply(partial(self._total_E, t=t))

        return B_grid, phi_grid, E_grid

    def _plot_B(self, B_grid, ax, B_range=3) -> plt.Axes:
        return ax.pcolormesh(B_grid.xg, B_grid.yg, B_grid.values, vmax=B_range, vmin=-B_range, cmap="bwr_r", zorder=0)

    def _plot_phi(self, phi_grid, ax) -> plt.Axes:
        return ax.contour(phi_grid.xg, phi_grid.yg, phi_grid.values, colors="gray", linewidths=1,
                          levels=[-5, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 5], zorder=5)

    def _plot_E(self, E_grid, ax, gridpoints_per_arrow=4, streamplot=False, E_range=4) -> plt.Axes:
        if streamplot:
            magnitudes = np.linalg.norm(E_grid.values, axis=-1)
            cmap = plt.get_cmap('viridis')
            print(magnitudes)
            norm = colors.LogNorm(vmin=10e-2, vmax=np.max(magnitudes))
            return ax.streamplot(E_grid.xg, E_grid.yg, E_grid.values[:,:,0], E_grid.values[:,:,1],
                          color=magnitudes, cmap=cmap, norm=norm, broken_streamlines=False, density=.25)

        skip = (slice(None, None, gridpoints_per_arrow), slice(None, None, gridpoints_per_arrow))

        clipped_E = np.clip(E_grid.values, -E_range, E_range)
        return ax.quiver(E_grid.xg[skip], E_grid.yg[skip], clipped_E[:, :, 0][skip], clipped_E[:, :, 1][skip],
                         scale=self.n_x, zorder=10)

    def _plot_particles(self, positions, ax) -> plt.Axes:
        return ax.scatter(positions[:, 0], positions[:, 1], c="black", s=10, zorder=15)


class DynamicSim(Simulator):
    def __init__(self, particles: List[DynamicParticle], dt: float = .01,
                 x_max: float = None, y_max: float = None, n_x: int = None, n_y: int = None):
        super().__init__(particles, dt, x_max, y_max, n_x, n_y)
        self.forces, self.E_forces, self.B_forces, self.RR_forces, self.lorentz_forces = self._calc_total_forces(True)

    ################
    ##main methods##
    ################
    def run(self, tmax: float, min_dt: float = 10e-4, max_dt: float = None, error_tolerance=10e-5) -> None:
        '''
        Runs the simulator until tmax is reached or adaptive timestep would be less than min_dt
        :param tmax: Time after which the simulation will be terminated
        :param min_dt: Minimum timestep allowed. If adaptive timestep would be shorter, the simulation is terminated
        :param max_dt: Maximum timestep allowed. If adaptive timestep would be longer, this value is used instead
        :param error_tolerance: Maximum allowed error of relative particle positions in units/second
        :return: None
        '''
        #ignore errors from progress bar due to adaptive timestep
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # progress bar formatting
            digits = int(np.floor(np.abs(np.log10(min_dt))))
            bar_format = "'{l_bar}{bar}| {n:." + str(digits) + "f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'"
            with tqdm(total=tmax, unit="sim_t", position=0, leave=True, desc='simulating', bar_format=bar_format) as pbar:

                #run simulation
                while self.t < tmax and self.dt > min_dt:
                    old_t = self.t
                    self.step_simulator(error_tolerance=error_tolerance, min_dt=min_dt, max_dt=max_dt)

                    pbar.update(self.t-old_t)

        print(f"simulation completed to t = {self.t:.3f}!")

    def step_simulator(self, error_tolerance=10e-5, min_dt=10e-4, max_dt=None):
        '''
        Steps the simulator using an adaptive 4th order Runge Kutta scheme
        :param min_dt: Minimum timestep allowed. If adaptive timestep would be shorter, the simulation is terminated
        :param max_dt: Maximum timestep allowed. If adaptive timestep would be longer, this value is used instead
        :param error_tolerance: Maximum allowed error of relative particle positions in units/second
        :return: None
        '''
        while True:
            # ensure time steps are not too small
            if self.dt < min_dt:
                break

            # try long step
            try:
                self._rkstep(self.dt * 2)
            except SpeedOfLightError:
                self._undo_and_reduce_dt(False)
                continue
            distances_1 = self._calc_distance_vecs()

            # undo long step
            self._undo_move()

            # try 2 short steps
            try:
                twice = False
                self._rkstep(self.dt)
                twice = True
                self._rkstep(self.dt)
            except SpeedOfLightError:
                self._undo_and_reduce_dt(twice)
                continue
            distances_2 = self._calc_distance_vecs()

            # determine new dt, and decide whether to accept current steps
            dt, accept_move = self._calc_adaptive_timestep(distances_1, distances_2, error_tolerance, max_dt)
            if accept_move:
                self.dt = dt
                break
            else:
                self._undo_move()
                self._undo_move()
                self.dt = dt

    ######################
    ##simulation helpers##
    ######################
    def _rkstep(self, dt, return_derivatives=False):
        f_1 = np.array([_scalar_vector(1 / (self.gammas * self.masses), self.forces), self.velocities])
        f_2 = self._substep(*f_1, .5 * dt)
        f_3 = self._substep(*f_2, .5 * dt)
        f_4 = self._substep(*f_3, dt)

        d_dt = (1 / 6) * (f_1 + 2 * f_2 + 2 * f_3 + f_4)
        self._move_particles(*d_dt, dt)
        if return_derivatives:
            return d_dt[0], d_dt[1]

    def _substep(self, dv_dts, dr_dts, dt):
        self._move_particles(dv_dts, dr_dts, dt)
        d_dts = np.array([_scalar_vector(1 / (self.gammas * self.masses), self.forces), self.velocities])
        self._undo_move()

        return d_dts

    def _move_particles(self, dv_dts, dr_dts, dt):
        for i, particle in enumerate(self.particles):
            new_r = self.positions[i] + dr_dts[i] * dt
            new_v = self.velocities[i] + dv_dts[i] * dt
            particle.move_particle(new_r, new_v, dt)

        self._update_values()
        self.times.append(self.t)
        self.paths.append(self.positions)

    def _undo_move(self) -> None:
        particle_times = {particle.t_curr for particle in self.particles}

        for particle in self.particles:
            if particle.t_curr == max(particle_times):
                # Only undoes particles who have already been moved.
                # This is useful when a move has to be undone immediately due to a SpeedOfLightError.
                particle.undo_move()

        self._update_values()
        if self.times:
            self.times.pop()
            self.paths.pop()

    def _undo_and_reduce_dt(self, twice, dt_fraction=.25):
        self._undo_move()
        if twice:
            self._undo_move()
        self.dt *= dt_fraction

    def _calc_adaptive_timestep(self, distances_1, distances_2, error_tolerance, max_dt=None):
        if max_dt is None:
            max_dt = 10e10

        # calcuates the max difference vectors between all interparticle distances 
        dist_diff = np.max(np.linalg.norm(distances_2 - distances_1, axis=-1))

        # determines the ratio between actual and allowed error
        error_ratio = 30 * self.dt * error_tolerance / dist_diff

        #calculates a new timestep and determines whether the current one was good enough
        dt, accept_move = None, None
        if error_ratio >= 1:
            dt = min(self.dt * 1.5, self.dt * error_ratio ** (1 / 4), max_dt)
            accept_move = True
        else:
            dt = self.dt * error_ratio ** (1 / 4)
            accept_move = False

        return dt, accept_move

    def _update_values(self):
        super()._update_values()
        self.forces, self.E_forces, self.B_forces, self.RR_forces, self.lorentz_forces = self._calc_total_forces(True)

    ################
    ##interactions##
    ################
    def _calc_lorentz_forces(self, return_component_forces=False):
        # calculates the lorentz EM forces on our particles
        vxB = np.array((self.velocities[:, 1] * self.Bs, -self.velocities[:, 0] * self.Bs)).T

        lorentz_forces = _scalar_vector(self.charges, self.Es + vxB)
        if return_component_forces:
            return lorentz_forces, _scalar_vector(self.charges, self.Es), _scalar_vector(self.charges, vxB)
        return lorentz_forces

    def _calc_radiation_reaction_forces(self):  #calculates the Abraham-Lorentz-Dirac self-force (actually a result of energy radiation)
        coefficient_term = -(self.charges ** 4 * self.gammas ** 2) / (6 * np.pi * self.masses ** 2)

        np.seterr(divide='ignore')
        Eprojv = _scalar_vector( _dot(self.Es, self.velocities) /
                                      _dot(self.velocities, self.velocities), self.velocities)
        E_perp = self.Es - Eprojv
        vxB = np.array((self.velocities[:, 1] * self.Bs, -self.velocities[:, 0] * self.Bs)).T
        inner_term = E_perp - vxB
        motion_term = _dot(inner_term, inner_term)

        F_RR = np.nan_to_num(_scalar_vector(coefficient_term * motion_term, self.velocities))
        return F_RR

    def _calc_total_forces(self, return_components=False):  #calculate all forces on our particles
        if return_components:
            lorentz_forces, E_forces, B_forces = self._calc_lorentz_forces(True)
        else:
            lorentz_forces = self._calc_lorentz_forces()

        RR_forces = self._calc_radiation_reaction_forces()
        forces = lorentz_forces + RR_forces

        if return_components:
            return forces, E_forces, B_forces, RR_forces, lorentz_forces
        return forces


##################
##vector helpers##
##################
def _dot(a, b):  #where a and b are both N*2 arrays
    return np.sum(a * b, axis=1)

def _scalar_vector(coeffs, a):  #coeffs is an N vec and a is an N*2 array
    return a * coeffs[:, np.newaxis]
