import cozmo
import math
import sys
import time

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
    return node1
    ############################################################################
       
def node_generator(cmap):
    rand_node = None
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object
    pass
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
        rand_node = None
        nearest_node = None
        pass
        ########################################################################
        time.sleep(0.01)
        cmap.add_path(nearest_node, rand_node)
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

async def CozmoPlanning(robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent

    ########################################################################
    # TODO: please enter your code below.
    # Description of function provided in instructions

async def detect_cube_and_update_cmap(robot, marked, cozmo_pos):
    #updates the map with observed cubes and sets the goal if it is found
    #marked can be initialized to {}
    
    global cmap
    cube_padding = 60.
    cozmo_padding = 100.
    goal_cube_found = False
    update_cmap = False
    goal_center = None
    for obj in robot.world.visible_objects:
        if obj.object_id in marked:
            continue

        print(obj)
        update_cmap = True
        is_goal_cube = robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id == obj.object_id

        robot_pose = robot.pose
        object_pose = obj.pose

        dx = object_pose.position.x - robot_pose.position.x
        dy = object_pose.position.y - robot_pose.position.y

        object_pos = Node((cozmo_pos.x+dx, cozmo_pos.y+dy))

        angle = object_pose.rotation.angle_z.radians

        if is_goal_cube:
            local_goal_pos = Node((0, -cozmo_padding))
            goal_pos = get_global_node(angle, object_pos, local_goal_pos)
            cmap.clear_goals()
            cmap.add_goal(goal_pos)
            goal_cube_found = True
            goal_center = object_pos

        obstacle_nodes = []
        obstacle_nodes.append(get_global_node(angle, object_pos, 
                                              Node((cube_padding, cube_padding))))
        obstacle_nodes.append(get_global_node(angle, object_pos, 
                                              Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(angle, object_pos, 
                                              Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(angle, object_pos, 
                                              Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)

        marked[obj.object_id] = obj

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
