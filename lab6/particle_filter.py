from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np
np.random.seed(RANDOM_SEED)
from itertools import product


def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []

    for particle in particles:
        x, y, h = particle.xyh
        dx, dy, dh = odom
        c, d = rotate_point(dx, dy, h)
        nx, ny, nh = add_odometry_noise((x+c, y+d, h+dh), heading_sigma=ODOM_HEAD_SIGMA, trans_sigma=ODOM_TRANS_SIGMA)
        newParticle = Particle(nx, ny, nh%360)
        motion_particles.append(newParticle)

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    num_random_sample = 25
    measured_particles = []
    weight = []

    if len(measured_marker_list) > 0:
        for particle in particles:
            x, y = particle.xy
            if grid.is_in(x, y) and grid.is_free(x, y):
                markers_visible_to_particle = particle.read_markers(grid)
                markers_visible_to_robot = measured_marker_list.copy()

                marker_pairs = []
                while len(markers_visible_to_particle) > 0 and len(markers_visible_to_robot) > 0:
                    all_pairs = product(markers_visible_to_particle, markers_visible_to_robot)
                    pm, rm = min(all_pairs, key=lambda p: grid_distance(p[0][0], p[0][1], p[1][0], p[1][1]))
                    marker_pairs.append((pm, rm))
                    markers_visible_to_particle.remove(pm)
                    markers_visible_to_robot.remove(rm)

                prob = 1.
                for pm, rm in marker_pairs:
                    d = grid_distance(pm[0], pm[1], rm[0], rm[1])
                    h = diff_heading_deg(pm[2], rm[2])

                    exp1 = (d**2)/(2*MARKER_TRANS_SIGMA**2)
                    exp2 = (h**2)/(2*MARKER_ROT_SIGMA**2)

                    likelihood = math.exp(-(exp1+exp2))
                    # The line is the key to this greedy algorithm
                    # prob *= likelihood
                    prob *= max(likelihood, DETECTION_FAILURE_RATE*SPURIOUS_DETECTION_RATE)

                # In this case, likelihood is automatically 0, and max(0, DETECTION_FAILURE_RATE) = DETECTION_FAILURE_RATE
                prob *= (DETECTION_FAILURE_RATE**len(markers_visible_to_particle))
                # Probability for the extra robot observation to all be spurious
                prob *= (SPURIOUS_DETECTION_RATE**len(markers_visible_to_robot))
                weight.append(prob)

            else:
                weight.append(0.)
    else:
        weight = [1.]*len(particles)

    norm = float(sum(weight))

    if norm != 0:
        weight = [i/norm for i in weight]
        measured_particles = Particle.create_random(num_random_sample, grid)
        measured_particles += np.random.choice(particles, PARTICLE_COUNT-num_random_sample, p=weight).tolist()
    else:
        measured_particles = Particle.create_random(PARTICLE_COUNT, grid)

    return measured_particles
