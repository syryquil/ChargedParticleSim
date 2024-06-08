import numpy as np

from grid import Grid
from particle import DynamicParticle
from simulator import DynamicSim

def static_particle(mass, charge, loc):
    '''
    :param mass: mass of particle
    :param charge: charge of particle
    :param loc: tuple (x, y) of location of particle
    :return: ManualParticle with no motion before t=0
    '''
    def stat_r(t):
        return np.array(loc)
    def stat_v(t):
        return np.zeros(2)

    return DynamicParticle(mass, charge, r_func=stat_r, v_func=stat_v)

def moving_particle(mass, charge, loc_i, v):
    '''
    :param mass: mass of particle
    :param charge: charge of particle
    :param loc_i: tuple (x, y) of initial location of particle
    :param v: tuple (v_x, v_y) of velocity of particle
    :return: ManualParticle in constant motion before t=0
    '''
    loc_i = np.array(loc_i)
    v = np.array(v)
    def move_r(t):
        return loc_i + v*t
    def move_v(t):
        return v

    return DynamicParticle(mass, charge, r_func=move_r, v_func=move_v)

def oscillating_particle(mass, charge, loc_i, v_x, amp, freq, phase=0):
    '''
    :param mass: mass of particle
    :param charge: charge of particle
    :param loc_i: tuple (x, y) of particle location at t=0
    :param v_x: x velocity
    :param amp: amplitude of oscillation in y
    :param freq: frequency of oscillation in y
    :param phase: phase shift of oscillation in y
    :return: manual particle moving at constant velocity in x while oscillating in y before t=0
    '''
    offset = amp*np.cos(phase)

    def osc_r(t):
        return np.array([v_x * t + loc_i[0],
                         amp * np.cos(2*np.pi*freq*t + phase) + loc_i[1] - offset])
    def osc_v(t):
        return np.array([v_x, -amp * 2 * np.pi * freq * np.sin(2*np.pi*freq*t + phase)])

    return DynamicParticle(mass, charge, r_func=osc_r, v_func=osc_v)

def accelerating_particle(mass, charge, loc_i, v_i, v_f, t_i, t_f):
    '''
    :param mass: mass of particle
    :param charge: charge of particle
    :param loc_i: tuple of (x, y) position. This is the location the particle would be located at t=0 if it had moved
                  at velocity v_i
    :param v_i: velocity before acceleration
    :param v_f: velocity after acceleration
    :param t_i: time acceleration starts (must be <0)
    :param t_f: time acceleration stops (must be <0)
    :return: ManualParticle that moves at v_i at t<t_i, accelerates to v_f, and moves at v_f at t>t_f before t=0
    '''
    a = (v_f - v_i)/(t_f-t_i)
    def acc_r(t):
        if t < t_i:
            return np.array([v_i * t + loc_i[0], loc_i[1]])
        elif t < t_f:
            return np.array([v_i * t_i + v_i * (t - t_i) + .5*a*(t - t_i)**2 + loc_i[0], loc_i[1]])
        elif t >= t_f:
            return np.array([v_i * t_i + v_i * (t_f - t_i) + .5*a*(t_f - t_i)**2 + v_f * (t - t_f) + loc_i[0], loc_i[1]])
    def acc_v(t):
        if t < t_i:
            return np.array([v_i, 0])
        elif t < t_f:
            return np.array([v_i + a*(t - t_i), 0])
        elif t >= t_f:
            return np.array([v_f, 0])

    return DynamicParticle(mass, charge, r_func=acc_r, v_func=acc_v)

#moving -- change background_B in _total_B to do the B field turning on
# p1 = moving_particle(1.2, 4, (.4, .4), (.99, 0))
# particles = [p1]

#wavy
# p1 = oscillating_particle(1, 2, (.5, 1.55), .92, .25, .24)
# p2 = oscillating_particle(1, 2, (2.5, 1.45), -.92, -.25, .24)
#
# particles = [p1, p2]

#orbiting
p1 = moving_particle(2, 2, (.5, 1.5), (.05, -.4))
p2 = moving_particle(2, 2, (1.5001, 2.5), (-.4, -.05))
p3 = moving_particle(2, 2, (2.5, 1.5), (-.05, .4))
p4 = moving_particle(2, 2, (1.5, .5), (.4, .05))
p5 = static_particle(8, -4.6, (1.5, 1.5))

particles = [p1, p2, p3, p4, p5]

if __name__ == "__main__":
    maxi = 1
    sim = DynamicSim(particles, dt=.001, x_max=maxi, y_max=maxi, n_x=100, n_y=100)

    sim.run(tmax=20, min_dt=.000001, max_dt=.001, error_tolerance=10e-5)
    sim.positions_plot("out/spiral.png", show=False, color_by_dt=False)
    try:
        sim.animated_fields_plot(fps=20, num_frames=100, t_i=0, filepath="out/wavy_last.mp4", plot_phi=True, show=False)
    finally:
        Grid.kill_multithreading()