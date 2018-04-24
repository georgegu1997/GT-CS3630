## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass

from skimage import color
import cozmo
import pickle
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
import json
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
from imgclassification import ImageClassifier

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

data = json.load(open('map_arena.json'))
WEIGHT = data['width']
HEIGHT = data['height']
INCHE_IN_MILLIMETERS = data["scale"]

IDLE = "none"
DRONE = "drone"
PLANE = "plane"
INSPECTION = "inspection"
PLACE = "place"
WINDOW_SIZE = 7

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

sign = lambda x: (1, -1)[x < 0]

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

def normalize_degree(degree):
    while degree > 180:
        degree -= 360
    while degree < -180:
        degree += 360
    return degree

# Load the pre-trained classifer from file
# https://blog.oldj.net/2010/05/26/python-pickle/
def loadClassifier(f = "clf.txt"):
    return pickle.load(open(f, "rb"))

class CoordinateTransformer():

    def __init__(self, origin_pose, map_pose):
        '''
        origin_pose: (x, y, h) from robot.pose in millimeters and degree
        map_pose: (x, y, h) from particle filter in inches and degree
        '''
        self.map_to_origin_in_millimeters_degrees = (
            origin_pose[0] - map_pose[0] * INCHE_IN_MILLIMETERS,
            origin_pose[1] - map_pose[1] * INCHE_IN_MILLIMETERS,
            normalize_degree(origin_pose[2] - map_pose[2])
        )

    def map_to_origin(self, map_pose):
        return (
            map_pose[0] * INCHE_IN_MILLIMETERS + self.map_to_origin_in_millimeters_degrees[0],
            map_pose[1] * INCHE_IN_MILLIMETERS + self.map_to_origin_in_millimeters_degrees[1],
            normalize_degree(map_pose[2] + self.map_to_origin_in_millimeters_degrees[2])
        )

    def origin_to_map(self, origin_pose):
        return (
            (origin_pose[0] - self.map_to_origin_in_millimeters_degrees[0]) / INCHE_IN_MILLIMETERS,
            (origin_pose[1] - self.map_to_origin_in_millimeters_degrees[1]) / INCHE_IN_MILLIMETERS,
            normalize_degree(origin_pose[2] - self.map_to_origin_in_millimeters_degrees[2])
        )

class Localizer():

    def __init__(self):
        self.positions = np.array([])

    def push(self, position):
        if self.positions.size == 0:
            self.positions = np.array(position)
        else:
            self.positions = np.vstack((self.positions, position))

    def get(self):
        return np.average(self.positions, axis=0)

def rotate_point(x, y, heading_deg):
    c = math.cos(math.radians(heading_deg))
    s = math.sin(math.radians(heading_deg))
    xr = x * c + y * -s
    yr = x * s + y * c
    return xr, yr

def recognize_cube(cube_pose):
    x, y = cube_pose

    if x < WEIGHT/2:
        if y < HEIGHT/2:
            return 'C'
        else:
            return 'A'
    else:
        if y < HEIGHT/2:
            return 'D'
        else:
            return 'B'


async def marker_processing(robot, camera_settings, show_diagnostic_image=False, return_unwarped_image=False):
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



    if return_unwarped_image == True:
      unwarped_image_list = [marker['unwarped_image'] for marker in markers]
      return marker_list, annotated_image, unwarped_image_list

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf

    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(5)).wait_for_completed()
    await robot.set_lift_height(0.0).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)

    marker_location = {
        'drone': Localizer(),
        'inspection': Localizer(),
        'order': Localizer(),
        'plane': Localizer(),
        'truck': Localizer(),
        'hands': Localizer(),
        'place': Localizer()
    }
    cube_target = {
        'A': 'drone',
        'B': 'plane',
        'C': 'inspection',
        'D': 'place'
    }

    clf = loadClassifier()

    ###################

    converged = False
    converge_score = 0

    while True:
        # lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
        if not converged:
            # Keep rotating
            await robot.drive_wheels(10.0, -10,0)

            '''Update the particle filter'''
            # Get the current pose
            curr_pose = robot.pose

            # Obtain odometry information
            odom = compute_odometry(curr_pose, cvt_inch=True)
            last_pose = curr_pose

            # Obtain list of currently seen markers and their poses
            marker_list, camera_image, unwarped_image_list = await marker_processing(robot, camera_settings, show_diagnostic_image=True, return_unwarped_image=True)

            if len(unwarped_image_list) > 0:
                unwarped = unwarped_image_list[0]
                related_position = marker_list[0]

                # Mapping from grey-scale (one channel)
                # To RGBA (four channel) according to the matplotlib PNG conversion
                cm = matplotlib.cm.ScalarMappable()
                cm.set_cmap("viridis")
                img = cm.to_rgba(unwarped)
                img = (np.around(img*255))
                image_arr = np.array([img], dtype=np.int16)

                # Classify the image
                feature_arr = clf.extract_image_features(image_arr)
                labels = clf.predict_labels(feature_arr)

                print(labels)

                # calculate the position of the cube in the origin frame
                # And store the result for further usage.
                robot_pose = (robot.pose.position.x, robot.pose.position.y, robot.pose.rotation.angle_z.radians)
                related_to_robot = [i * INCHE_IN_MILLIMETERS for i in related_position[:2]]
                related_to_origin = (
                    robot_pose[0] + related_to_robot[0] * math.cos(robot_pose[2]) - related_to_robot[1] * math.sin(robot_pose[2]),
                    robot_pose[1] + related_to_robot[0] * math.sin(robot_pose[2]) + related_to_robot[1] * math.cos(robot_pose[2]),
                )
                marker_location[labels[0]].push(related_to_origin)

            # Update the particle filter using the above information
            # Not that the the first element in marker_list is the list
            (m_x, m_y, m_h, m_confident) = pf.update(odom, marker_list)

            # Update the GUI
            gui.show_particles(pf.particles)
            gui.show_mean(m_x, m_y, m_h)
            gui.show_camera_image(camera_image)
            gui.updated.set()

            # Trusted if only 10 continuous converge
            if m_confident:
                print("converge score:", converge_score)
                converge_score += 1
            else:
                converge_score = 0

            # Point of convergence
            if converge_score > 10:
                converged = True

                # Initiate the Transformer between two frames
                origin_pose = (robot.pose.position.x, robot.pose.position.y, robot.pose.rotation.angle_z.degrees)
                map_pose = (m_x, m_y, m_h)
                ct = CoordinateTransformer(origin_pose, map_pose)

                # Add the obstacle
                object_pose_map = (WEIGHT / 2 -0.5, HEIGHT / 2 - 0.5, 0)
                object_pose_origin = ct.map_to_origin(object_pose_map)
                fixed_object = await robot.world.create_custom_fixed_object(
                    cozmo.util.Pose(object_pose_origin[0], object_pose_origin[1], 0, angle_z=degrees(object_pose_origin[2])),
                    INCHE_IN_MILLIMETERS,
                    INCHE_IN_MILLIMETERS,
                    INCHE_IN_MILLIMETERS,

                )

                # Wait until observe 3 distinct cubes
                cubes = await robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=60, include_existing = True)

        else:
            # await robot.drive_wheels(10.0, -10,0)
            await robot.drive_wheels(0, 0)
            print(len(cubes))
            for i, cube in enumerate(cubes):
                # Recognize the cube
                map_pose = ct.origin_to_map((cube.pose.position.x, cube.pose.position.y, cube.pose.rotation.angle_z.degrees))
                cube_label = recognize_cube((map_pose[0], map_pose[1]))
                print('Type of cube: ', cube_label)
                print("Picking up.")

                # Before picking up, go back to the center
                # And adjust the angle towards the next cube
                cube_pose = (cube.pose.position.x, cube.pose.position.y)
                start_angle = math.atan2(
                                    cube_pose[1],
                                    cube_pose[0]
                                ) / PI * 180.0
                await robot.go_to_pose(cozmo.util.Pose(0, 0, 0, angle_z=degrees(start_angle)))\
                           .wait_for_completed()

                # Go to pick it up
                await robot.pickup_object(cube, num_retries=100).wait_for_completed()
                print("Going to the marker:", cube_target[cube_label])

                # Adjust the heading at droping point
                # angle is from the origin to the marker
                marker_pose = marker_location[cube_target[cube_label]].get()
                end_angle = math.atan2(
                                marker_pose[1],
                                marker_pose[0]
                            ) / PI * 180.0

                # 0.75 here is used to place the cube at somewhat distance from the marker
                await robot.go_to_pose(cozmo.util.Pose(marker_pose[0]*0.75, marker_pose[1]*0.75, 0, angle_z=degrees(end_angle)))\
                           .wait_for_completed()
                print("Dropping the cube.")
                await robot.place_object_on_ground_here(cube).wait_for_completed()

            break

    ###################

class CozmoThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)

def main():
    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()

def testLocalizer():
    a = [1,2,3]
    b = [4,5,6]
    c = [4,5,6]
    l = Localizer()
    l.push(a)
    l.push(b)
    l.push(c)
    print(l.get())


if __name__ == '__main__':
    main()
    # testLocalizer()




# pip3 install --user cozmo[3dviewer]
