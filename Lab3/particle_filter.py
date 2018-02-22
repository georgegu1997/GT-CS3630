from grid import *
from particle import Particle
from utils import *
from setting import *

# import nunmpy for np.random.choice
import numpy as np


def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    # if the robot doesn't move, the particles shouldn't move either
    if odom[0] == 0 and odom[1] == 0 and odom[2] == 0:
        return particles
    # The xyh of particles is in world frame
    # The odom is in particle frame
    # Transformation of x and y is needed
    motion_particles = []
    for p in particles:
        px, py, ph = p.xyh
        # add noise in the particle frame
        (nx, ny, dh) = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        # from particle frame to world frame -> rotate back using the particle current heading angle
        dx, dy = rotate_point(nx, ny, -ph)
        # the heading rotation is the same in both frames.
        np = Particle(px + dx, py + dy, ph + dh)
        motion_particles.append(np)

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
    # if not markers seen, no update
    if len(measured_marker_list) <= 0:
        return particles

    # 2 lists are matched for (particle, weight) pairs
    measured_particles = []
    weights = []

    for p in particles:
        # particles within an obstacle or outside the map should have a weight of 0
        if p.x > grid.width or p.x < 0  or p.y > grid.height or p.y < 0:
            weights.append(0)
            continue

        prob = 1.0
        simulated_marker_list = p.read_markers(grid)

        # # if the lengths of two marker lists are not consistent
        # if len(simulated_marker_list) > len(measured_marker_list):
        #     # the robot fails to detect some markers
        #     prob *= DETECTION_FAILURE_RATE ** (len(simulated_marker_list) - len(measured_marker_list))
        # elif len(simulated_marker_list) < len(measured_marker_list):
        #     # the robot detects some spurious markers
        #     prob *= SPURIOUS_DETECTION_RATE ** (len(measured_marker_list) - len(simulated_marker_list))

        # pair the markers by distance
        for measured_marker in measured_marker_list:
            min_dist = float('Inf')
            candidate_pairing = None

            for simulated_marker in simulated_marker_list:
                dist = grid_distance(measured_marker[0], measured_marker[1], simulated_marker[0], simulated_marker[1])
                # update minimum distance and candidate pairing
                if dist < min_dist:
                    candidate_pairing = simulated_marker
                    min_dist = dist

            if candidate_pairing != None:
                diff_angle = diff_heading_deg(measured_marker[2], simulated_marker[2])
                prob *= np.exp(-(min_dist**2)/(2*MARKER_TRANS_SIGMA**2)-(diff_angle**2)/(2*MARKER_ROT_SIGMA**2))
                # remove the marker in simulated_marker_list to avoid double pairing
                simulated_marker_list.remove(candidate_pairing)

        weights.append(prob)

    # normalize weights
    weights = np.divide(weights, np.sum(weights))

    # resample
    measured_particles = np.random.choice(particles, size = len(particles), replace = True, p = weights)

    # maintain some small percentage of random samples
    measured_particles = np.ndarray.tolist(measured_particles) + Particle.create_random(100, grid)

    return measured_particles
