## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet

# try:
#     import matplotlib
#     matplotlib.use('TkAgg')
# except ImportError:
#     pass

from skimage import color
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image

from math import atan2
import random


from markers import detect, annotator

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

# Constant
PI = 3.14159
POSITION_TOL = 1.5
HEADING_TOL = 12.0

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera.
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh)
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)

    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf

    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)

    ###################

    # YOUR CODE HERE
    start = time.time()
    converged = False
    convergence_score = 0
    arrived = False

    while True:
        if arrived:
            while not robot.is_picked_up:
                await robot.drive_straight(distance_mm(0), speed_mmps(0)).wait_for_completed()

		# Detect whether it is kidnapped
        if robot.is_picked_up:
            # indicate to re-localize and continue
            # print("Picked up")
            flag_odom_init = False
            arrived = False
            converged = False
            convergence_score = 0
            await robot.drive_wheels(0.0, 0,0)
            # Have the robot act unhappy when we pick it up for kidnapping
            await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabUnhappy).wait_for_completed()

            while robot.is_picked_up:
                await robot.drive_straight(distance_mm(0), speed_mmps(0)).wait_for_completed()
            continue


        # use the flag_odom_init to indicate whether it is kidnapped
        if flag_odom_init == False:
            # Reset the last pose
            last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))

            # Reset particle filter to a uniform distribution
            pf.particles = Particle.create_random(PARTICLE_COUNT, grid)

            flag_odom_init = True

        # Get the current pose
        curr_pose = robot.pose

        # Obtain odometry information
        odom = compute_odometry(curr_pose, cvt_inch=True)
        last_pose = robot.pose

        # Obtain list of currently seen markers and their poses
        marker_list, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=True)

        # Update the particle filter using the above information
        # Not that the the first element in marker_list is the list
        (m_x, m_y, m_h, m_confident) = pf.update(odom, marker_list)

        gui.show_particles(pf.particles)
        gui.show_mean(m_x, m_y, m_h)
        gui.show_camera_image(camera_image)
        gui.updated.set()
        # print("marker_list:", marker_list)

        print(m_x, m_y, m_h, m_confident)

        # if converged once
        if m_confident:
            convergence_score += 3

        # Converged many times means good estimate
        if convergence_score > 30:
            converged = True

        # If has converged but diverge again
        if converged and not m_confident:
            convergence_score -= 2

        # if diverge too much
        if convergence_score < 0:
            converged = False
            convergence_score = 0

        if not converged:
            # the localization has not converged -- global localization problem
            # Have the robot actively look around (spin in place)
            if ((time.time() - start) // 1) % 8 < 3 or len(marker_list) <= 0:
                await robot.drive_wheels(15.0, -15,0)
            elif len(marker_list) > 0:
                if (marker_list[0][0] > 10):
                    await robot.drive_wheels(20.0, 20,0)
                if (marker_list[0][0] < 7):
                    await robot.drive_wheels(-20.0, -20,0)

            # if ((time.time() - start) // 1) % 8 == 0 or len(marker_list) <= 0:
            #     await robot.turn_in_place(degrees(random.randint(-60, 60))).wait_for_completed()
            # elif ((time.time() - start) // 1) % 2 == 0 and len(marker_list) > 0:
            #     await robot.drive_straight(distance_mm(random.randint(-50, 50)), speed_mmps(50)).wait_for_completed()
            # else:
            #     pass

            # if ((time.time() - start) // 2) % 3 == 0:
            #     await robot.drive_wheels(30.0, 30,0)
            # elif ((time.time() - start) // 2) % 3 == 1:
            #     await robot.drive_wheels(-30.0, -30,0)
            # elif ((time.time() - start) // 2) % 3 == 2:
            #     await robot.drive_wheels(-30.0, 30,0)

        else:
            # await robot.drive_wheels(0.0, 0,0)

            # dx = goal[0] - m_x;
            # dy = goal[1] - m_y;
            # target_heading = atan2(dy, dx) * 180.0 / PI
            #
            # dh_deg = diff_heading_deg(target_heading, m_h)
            # dist = grid_distance(m_x, m_y, goal[0], goal[1])
            # dh_deg_2 = diff_heading_deg(goal[2], target_heading)
            #
            # await robot.turn_in_place(degrees(dh_deg)).wait_for_completed()
            # await robot.drive_straight(distance_mm(dist * grid.scale), speed_mmps(50)).wait_for_completed()
            # await robot.turn_in_place(degrees(dh_deg_2)).wait_for_completed()
            #
            # arrived = True

            # Detect whether the robot is in goal
            diff_x = abs(m_x - goal[0])
            diff_y = abs(m_y - goal[1])
            diff_h = abs(m_h - goal[2])

            # if arrived at the goal state (postion and heading)
            if diff_x <= POSITION_TOL and diff_y <= POSITION_TOL and diff_h <= HEADING_TOL:
                print("arrived at the goal state (postion and heading)")
                # Have the robot play a happy animation, then stand still
                await robot.drive_wheels(0.0, 0,0)
                await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()
                arrived = True
                continue

            # If not at the goal position
            elif diff_x > POSITION_TOL or diff_y > POSITION_TOL:
                print("not at the goal position")
                # the localization has converged -- position tracking problem

                # calculate the difference in position and heading
                dx = goal[0] - m_x;
                dy = goal[1] - m_y;
                target_heading = atan2(dy, dx) * 180.0 / PI
                dh_deg = diff_heading_deg(target_heading, m_h)
                dist = grid_distance(m_x, m_y, goal[0], goal[1])

                # first adjust the heading towards the goal
                angular_speed = min(20, abs(dh_deg / 3))
                # angular_speed = 10
                speed = 40
                if dh_deg > HEADING_TOL:
                    await robot.drive_wheels(-angular_speed, angular_speed)
                elif dh_deg < - HEADING_TOL:
                    await robot.drive_wheels(angular_speed, -angular_speed)
                # then drive to the goal
                else:
                    await robot.drive_wheels(50.0, 50,0)

            # if at the goal position, then adjust the heading
            else:
                await robot.drive_wheels(0.0, 0,0)
                print("at the goal position, then adjust the heading")
                dh_deg = diff_heading_deg(goal[2], m_h)
                angular_speed = min(20, abs(dh_deg / 3))
                if dh_deg > HEADING_TOL:
                    await robot.drive_wheels(-angular_speed, angular_speed)
                elif dh_deg < - HEADING_TOL:
                    await robot.drive_wheels(angular_speed, -angular_speed)
                else:
                    await robot.drive_wheels(0.0, 0,0)


    ###################

class CozmoThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()
