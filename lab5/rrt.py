"""
CS 3630 Lab5
Group: 65
Teammates: Qiao Gu, Guanzhi Wang
"""


import cozmo
import math
import sys
import time
import random

from cmap import *
from gui import *
from utils import *

MAX_NODES = 20000


def step_from_to(node0, node1, limit=75):
    ########################################################################
    # TODO: please enter your code below.
    # 1. If distance between two nodes is less than limit, return node1
    # 2. Otherwise, return a node in the direction from node0 to node1 whose
    #    distance to node0 is limit. Recall that each iteration we can move
    #    limit units at most
    # 3. Hint: please consider using np.arctan2 function to get vector angle
    # 4. Note: remember always return a Node object

    distance = get_dist(node0, node1)

    # If distance between two nodes is less than limit, return node1
    if distance < limit:
        return node1

    # definite proportion and division point
    # Ding Bi Fen Dian Gong Shi
    new_node = Node((
        node0.x * (1 - limit / distance) + node1.x * (limit / distance),
        node0.y * (1 - limit / distance) + node1.y * (limit / distance)
    ))
    ############################################################################

    return new_node


def node_generator(cmap):
    rand_node = None
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object

    # with a 5% chance the goal location is returned
    '''CANNOT RETURN THE GOAL ITSELF, OR INFINITE LOOP'''
    if random.random() < 0.05:
        goal = cmap.get_goals()[0]
        return Node((goal.x, goal.y))

    # Use CozMap.is_inbound and CozMap.is_inside_obstacles to
    # determine the legitimacy of the random node.
    while rand_node == None \
            or cmap.is_inside_obstacles(rand_node) \
            or (not cmap.is_inbound(rand_node)):
        # Use CozMap width and height to get a uniformly distributed random node
        rand_node = Node((
            random.random() * cmap.width,
            random.random() * cmap.height
        ))
    ############################################################################

    return rand_node


def RRT(cmap, start):
    cmap.add_node(start)
    map_width, map_height = cmap.get_size()
    while (cmap.get_num_nodes() < MAX_NODES):
        ########################################################################
        # TODO: please enter your code below.
        # 1. Use CozMap.get_random_valid_node() to get a random node. This
        #    function will internally call the node_generator above
        # 2. Get the nearest node to the random node from RRT
        # 3. Limit the distance RRT can move
        # 4. Add one path from nearest node to random node
        #

        # Use CozMap.get_random_valid_node() to get a random node
        rand_node = cmap.get_random_valid_node()

        # Get the nearest node to the random node from RRT
        nearest_node = None
        nearest_dist = 1e7
        for n in cmap.get_nodes():
            if get_dist(rand_node, n) < nearest_dist:
                nearest_node = n
                nearest_dist = get_dist(rand_node, n)

        # Limit the distance RRT can move
        new_node = step_from_to(nearest_node, rand_node)
        ########################################################################
        time.sleep(0.01) # ????

        # Add one path from nearest node to random node
        cmap.add_path(nearest_node, new_node)

        if cmap.is_solved():
            break

    path = cmap.get_path()
    smoothed_path = cmap.get_smooth_path()

    if cmap.is_solution_valid():
        print("A valid solution has been found :-) ")
        print("Nodes created: ", cmap.get_num_nodes())
        print("Path length: ", len(path))
        print("Smoothed path length: ", len(smoothed_path))
    else:
        print("Please try again :-(")

def get_current_pose_on_cmap(robot):
    start_x = 6*25.4
    start_y = 10*25.4
    current_x = robot.pose.position.x
    current_y = robot.pose.position.y
    # angle is not useful?
    current_angle = robot.pose.rotation.angle_z

    return Node((current_x + start_x, current_y + start_y))


async def CozmoPlanning(robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent

    ########################################################################
    # TODO: please enter your code below.
    # Description of function provided in instructions

    # get the width and the height of the map
    map_width, map_height = cmap.get_size()
    # print(map_width/25.4, map_height/25.4)

    start_x = 6*25.4
    start_y = 10*25.4

    # initialize starting global position of the robot
    cozmo_pos = Node((start_x, start_y))
    cozmo_angle = 0.0

    # initialize marked
    marked = {}

    # set starting position of cmap
    # cmap.set_start(cozmo_pos)

    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # A counter
    i = 0

    while True:
        print("counter:",i)
        i += 1

        # Make an observation every iteration
        # Note that each iteration there can be a complete movement
        # from current node to the next node.
        cmap.set_start( get_current_pose_on_cmap(robot) )
        update_cmap, goal_center = await detect_cube_and_update_cmap(robot, marked, get_current_pose_on_cmap(robot))

        # If new change to the map, clear the path and re-compute
        if update_cmap:
            print("new change to the map, clear the path and re-compute")
            cmap.reset()

        # If not solved, try to solve the map
        '''NOTE that we may need to kidnap the robot to set origin_id'''
        if not cmap.is_solved():
            print("not solved, try to solve the map")
            # if cannot see the center and no goal has been observed
            # go to the center and rotate
            if goal_center == None and len(cmap.get_goals()) == 0:
                print("cannot see the center and no goal has been observed go to the center and rotate")
                next_pose = cozmo.util.Pose(
                                map_width/2 - start_x,
                                map_height/2 - start_y,
                                0,
                                angle_z = cozmo.util.Angle((i%12-6) * 30)
                            )

                await robot.go_to_pose(next_pose).wait_for_completed()
                # continue and make a new observation
                continue

            # If we get a goal on cmap
            # set the start and solve RRT.
            if len(cmap.get_goals()) > 0:
                print("we get a goal on cmap set the start and solve it.")

                cmap.set_start( get_current_pose_on_cmap(robot) )

                RRT(cmap, cmap.get_start())
                if cmap.is_solved():
                    path = cmap.get_smooth_path()
                    # path = cmap.get_path()
                    next_way_point_index = 1

        # If the path is known
        # head to the goal
        if cmap.is_solved():
            print("the path is known head to the goal")
            if next_way_point_index == len(path):
                print("Arrived")
                continue

            last_way_point = path[next_way_point_index - 1]
            next_way_point = path[next_way_point_index]
            end_angle = math.atan2(
                            next_way_point.y - last_way_point.y,
                            next_way_point.x - last_way_point.x
                        )

            next_pose = cozmo.util.Pose(
                            next_way_point.x - start_x,
                            next_way_point.y - start_y,
                            0,
                            angle_z = cozmo.util.Angle(end_angle)
                        )
            await robot.go_to_pose(next_pose).wait_for_completed()
            next_way_point_index += 1


def get_global_node(local_angle, local_origin, node):
    """Helper function: Transform the node's position (x,y) from local coordinate frame specified
                        by local_origin and local_angle to global coordinate frame.
                        This function is used in detect_cube_and_update_cmap()
        Arguments:
        local_angle, local_origin -- specify local coordinate frame's origin in global coordinate frame
        local_angle -- a single angle value
        local_origin -- a Node object

        Outputs:
        new_node -- a Node object that decribes the node's position in global coordinate frame
    """
    ########################################################################
    # TODO: please enter your code below.
    new_node = None

    x = node[0]
    y = node[1]
    c = math.cos(local_angle)
    s = math.sin(local_angle)
    xr = x * c + y * -s + local_origin[0]
    yr = x * s + y * c + local_origin[1]

    new_node = Node((xr, yr))
    return new_node


async def detect_cube_and_update_cmap(robot, marked, cozmo_pos):
    """Helper function used to detect obstacle cubes and the goal cube.
       1. When a valid goal cube is detected, old goals in cmap will be cleared and a new goal corresponding to the approach position of the cube will be added.
       2. Approach position is used because we don't want the robot to drive to the center position of the goal cube.
       3. The center position of the goal cube will be returned as goal_center.

        Arguments:
        robot -- provides the robot's pose in G_Robot
                 robot.pose is the robot's pose in the global coordinate frame that the robot initialized (G_Robot)
                 also provides light cubes
        cozmo_pose -- provides the robot's pose in G_Arena
                 cozmo_pose is the robot's pose in the global coordinate we created (G_Arena)
        marked -- a dictionary of detected and tracked cubes (goal cube not valid will not be added to this list)

        Outputs:
        update_cmap -- when a new obstacle or a new valid goal is detected, update_cmap will set to True
        goal_center -- when a new valid goal is added, the center of the goal cube will be returned
    """
    global cmap

    # Padding of objects and the robot for C-Space
    cube_padding = 60.
    cozmo_padding = 100.

    # Flags
    update_cmap = False
    goal_center = None

    # Time for the robot to detect visible cubes
    time.sleep(1)

    for obj in robot.world.visible_objects:

        if obj.object_id in marked:
            continue

        # Calculate the object pose in G_Arena
        # obj.pose is the object's pose in G_Robot
        # We need the object's pose in G_Arena (object_pos, object_angle)
        dx = obj.pose.position.x - robot.pose.position.x
        dy = obj.pose.position.y - robot.pose.position.y

        object_pos = Node((cozmo_pos.x+dx, cozmo_pos.y+dy))
        object_angle = obj.pose.rotation.angle_z.radians

        # The goal cube is defined as robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id
        if robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id == obj.object_id:

            # Calculate the approach position of the object
            local_goal_pos = Node((0, -cozmo_padding))
            goal_pos = get_global_node(object_angle, object_pos, local_goal_pos)

            # Check whether this goal location is valid
            if cmap.is_inside_obstacles(goal_pos) or (not cmap.is_inbound(goal_pos)):
                print("The goal position is not valid. Please remove the goal cube and place in another position.")
            else:
                cmap.clear_goals()
                cmap.add_goal(goal_pos)
                goal_center = object_pos

        # Define an obstacle by its four corners in clockwise order
        obstacle_nodes = []
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)
        marked[obj.object_id] = obj
        update_cmap = True

    return update_cmap, goal_center


class RobotThread(threading.Thread):
    """Thread to run cozmo code separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        # Please refrain from enabling use_viewer since it uses tk, which must be in main thread
        cozmo.run_program(CozmoPlanning,use_3d_viewer=False, use_viewer=False)
        stopevent.set()


class RRTThread(threading.Thread):
    """Thread to run RRT separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        while not stopevent.is_set():
            RRT(cmap, cmap.get_start())
            time.sleep(100)
            cmap.reset()
        stopevent.set()


if __name__ == '__main__':
    global cmap, stopevent
    stopevent = threading.Event()
    robotFlag = False
    for i in range(0,len(sys.argv)):
        if (sys.argv[i] == "-robot"):
            robotFlag = True
    if (robotFlag):
        cmap = CozMap("maps/emptygrid.json", node_generator)
        robot_thread = RobotThread()
        robot_thread.start()
    else:
        cmap = CozMap("maps/map2.json", node_generator)
        sim = RRTThread()
        sim.start()
    visualizer = Visualizer(cmap)
    visualizer.start()
    stopevent.set()
