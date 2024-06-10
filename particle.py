import numpy as np
from typing import Callable

from scipy import optimize

class Particle:
    def __init__(self, mass: float, charge: float, initial_dt: float, r_func: Callable, v_func: Callable):
        """
        :param mass: mass of the particle
        :type mass: positive float
        :param charge: charge of the particle
        :type charge: nonzero float
        :param initial_dt: first dt
        :type initial_dt:
        :param r_func:
        :type r_func:
        :param v_func:
        :type v_func:
        """
        if mass <= 0:
            raise ValueError("mass must be positive")
        if initial_dt <= 0:
            raise ValueError("initial_dt must be positive")

        self.times = [0.]
        self.initial_dt = initial_dt
        self.t_curr = 0.
        self.mass, self.charge = mass, charge
        self.r_func, self.v_func = r_func, v_func
        self.r_curr = r_func(0)
        self.v_curr = v_func(0)

        self._check_velocity(self.v_curr, self.t_curr)

    def __str__(self):
        np.set_printoptions(precision=4)
        return f"manual particle motion at current time {self.t_curr}: r: {self.get_r()}, v: {self.get_v()}"

    ##############
    ##properties##
    ##############

    def get_dt(self):
        if len(self.times) > 1:
            return self.times[-1] - self.times[-2]
        else:
            return self.initial_dt

    def get_r(self, t=None) -> np.ndarray:
        pass

    def get_v(self, t=None) -> np.ndarray:
        pass

    def get_gamma(self, t=None):  #get lorentz factor
        return 1 / (1 - np.linalg.norm(self.get_v(t)) ** 2)

    ################
    ##field values##
    ################

    def phi(self, loc, t=None, ret_t_prime=False):  # Scalar Liénard–Wiechert potential
        t_prime, r_prime, v_prime = self._calc_source_values(loc, t)
        phi = self.charge / (
                4 * np.pi * (self._distance(loc, t_prime) - np.dot(self._distance_vec(loc, t_prime), v_prime)))
        return (phi, t_prime) if ret_t_prime else phi

    def A(self, loc: np.ndarray, t=None):  # Vector Liénard–Wiechert potential
        phi, t_prime = self.phi(loc, t, True)
        return phi * self.get_v(t_prime)

    def E(self, loc: np.ndarray, t=None, h=10 ** -6, dt=10 ** -6):  # Electric field vectors
        if t is None:
            t = self.t_curr

        self._check_distance(loc, t, max(h, dt))

        dx, dy = np.array([h, 0]), np.array([0, h])

        dphi_dx = (self.phi(loc + dx, t) - self.phi(loc - dx, t)) / (2 * h)
        dphi_dy = (self.phi(loc + dy, t) - self.phi(loc - dy, t)) / (2 * h)
        grad_phi = np.array([dphi_dx, dphi_dy])

        dA_dt = (self.A(loc, t + dt) - self.A(loc, t - dt)) / (2 * dt)

        return -grad_phi - dA_dt

    def B(self, loc, t=None, h=10 ** -6):
        # Magnetic field z component scalar. x and y components are always 0 in 2 motion
        if t is None:
            t = self.t_curr

        self._check_distance(loc, t, h)

        dx, dy = np.array([h, 0]), np.array([0, h])

        dAy_dx = (self.A(loc + dx, t)[1] - self.A(loc - dx, t)[1]) / (2 * h)
        dAx_dy = (self.A(loc + dy, t)[0] - self.A(loc - dy, t)[0]) / (2 * h)

        return dAy_dx - dAx_dy

    ##############################
    ##field related calculations##
    ##############################

    def _distance_vec(self, loc: np.ndarray, t=None):
        return loc - self.get_r(t)

    def _distance(self, loc: np.ndarray, t=None):
        return np.linalg.norm(self._distance_vec(loc, t))

    def _calc_source_values(self, loc: np.ndarray, t=None):
        """
        Returns the values of the particle seen at a certain location loc accounting for the speed of light.
        t_prime, r_prime, and v_prime are typically known in literature as the "retarded time" as it is delayed.
        """
        if t is None:
            t = self.t_curr

        def light_cone_cond(t_prime_guess, r, t):
            """
            this equals 0 at the time the source particle crosses the position r's light cone,
            ie the most recent time an observer at r is able to get information about the particle.
            """
            return self._distance(r, t=t_prime_guess) - (t - t_prime_guess)

        t_prime = optimize.brentq(light_cone_cond, -10000, self.t_curr, args=(loc, t))

        return t_prime, self.get_r(t_prime), self.get_v(t_prime)

    ##########
    ##helper##
    ##########

    def _check_distance(self, loc, t, threshold=0):
        loc = np.array(loc)
        if self._distance(loc, t) <= threshold:
            raise RuntimeError("Attempting to calculate fields too close to particle")

    def _check_t(self, t):
        pass

    @staticmethod
    def _check_velocity(v, t=None):
        speed = np.linalg.norm(v)
        if speed >= 1:
            if t is not None:
                raise SpeedOfLightError(f"velocity {speed}c at time {t} cannot exceed the speed of light")
            else:
                raise SpeedOfLightError(f"velocity {speed}c cannot exceed the speed of light")


class DynamicParticle(Particle):
    def __init__(self, mass, charge, r_func: Callable, v_func: Callable, initial_dt=.01):
        super().__init__(mass, charge, initial_dt, r_func, v_func)
        self.r = [self.r_curr]
        self.v = [self.v_curr]

    ################
    ##main methods##
    ################

    def move_particle(self, new_r: np.ndarray, new_v: np.ndarray, dt=None):  # manually move the particle
        if dt is None:
            dt = self.get_dt()
        # increment t
        self.t_curr += dt
        self.times.append(self.t_curr)
        # update motion
        self.r_curr = np.array(new_r)
        self.r.append(self.r_curr)
        self.v_curr = np.array(new_v)
        self.v.append(self.v_curr)

        self._check_velocity(self.v_curr, self.t_curr)

    def undo_move(self):
        if self.t_curr == 0:
            raise RuntimeError("cannot undo beyond t=0")
        self.times.pop()
        self.r.pop()
        self.v.pop()

        self.t_curr = self.times[-1]
        self.r_curr = self.r[-1]
        self.v_curr = self.v[-1]

    ##############
    ##properties##
    ##############

    def get_r(self, t=None) -> np.ndarray:
        t = self._check_t(t)
        if t == self.t_curr:
            return self.r[-1]
        elif t <= 0:
            return self.r_func(t)
        else:
            return _interp(t, self.times, self.r)

    def get_v(self, t=None) -> np.ndarray:
        t = self._check_t(t)
        v = None
        if t == self.t_curr:
            v = self.v[-1]
        elif t <= 0:
            v = self.v_func(t)
        else:
            v = _interp(t, self.times, self.v)

        self._check_velocity(v, t)
        return v

    ##########
    ##helper##
    ##########

    def _check_t(self, t):  # checks that time t is valid. If no t is given, returns t curr
        if t is None:
            return self.t_curr
        elif t > self.t_curr:
            raise ValueError("t is greater than t_curr")
        else:
            return t


#UNUSED BUT KEPT JUST IN CASE
class AutoParticle(Particle):
    def __init__(self, mass, charge, r_func: Callable, v_func: Callable, dt=.01):
        super().__init__(mass, charge, dt, r_func, v_func)

    ################
    ##main methods##
    ################

    def move_particle(self, dt=None):  #move the particle according to our functions
        if dt is None:
            dt = self.get_dt()
        # increment t
        self.t_curr += dt
        self.times.append(self.t_curr)
        # update motion
        self.r_curr = self.r_func(self.t_curr)
        self.v_curr = self.v_func(self.t_curr)

        self._check_velocity(self.v_curr, self.t_curr)

    ##############
    ##properties##
    ##############

    def get_r(self, t=None) -> np.ndarray:
        t = self._check_t(t)
        return self.r_func(t)

    def get_v(self, t=None) -> np.ndarray:
        t = self._check_t(t)
        v = self.v_func(t)

        self._check_velocity(v, t)
        return v

    ##########
    ##helper##
    ##########

    def _check_t(self, t=None):  #returns t_curr if no t given
        if t is None:
            return self.t_curr
        return t


class SpeedOfLightError(RuntimeError):
    pass


def _interp(t: float, times: list[float], vals: list[np.ndarray]) -> np.ndarray:
    idx = np.searchsorted(times, t)
    dt = times[idx] - times[idx - 1]
    dr_dt = (vals[idx] - vals[idx - 1]) / dt
    interpolated = vals[idx - 1] + dr_dt * (t - times[idx])
    return interpolated
