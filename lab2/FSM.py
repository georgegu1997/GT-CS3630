import pickle
import sys
import cozmo
import time
import asyncio
import numpy as np
from imgclassification import ImageClassifier
from cozmo.util import degrees, distance_mm, speed_mmps

IDLE = "none"
DRONE = "drone"
ORDER = "order"
INSPECTION = "inspection"
WINDOW_SIZE = 5

# Load the pre-trained classifer from file
# https://blog.oldj.net/2010/05/26/python-pickle/
def loadClassifier(f = "clf.txt"):
    return pickle.load(open(f, "rb"))

# reference cozmo_sdk_examples_1.2.1/tutorials/04_cubes_and_objects/07_lookaround.py

def droneState(robot: cozmo.robot.Robot):
    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    
    # try to find a block
    cube = None
    
    try:
        cube = robot.world.wait_for_observed_light_cube(timeout=30)
        print("Found cube", cube)

    except asyncio.TimeoutError:
        print("Didn't find a cube :-(")

    finally:
        # whether we find it or not, we want to stop the behavior
        look_around.stop()
    
    if cube is None:
        robot.play_anim_trigger(cozmo.anim.Triggers.MajorFail)
        return

    print("Yay, found cube")

    cube.set_lights(cozmo.lights.green_light.flash())

    # pick up the cube
    action = robot.pickup_object(cube)
    print("got action", action)
    result = action.wait_for_completed(timeout=30)
    print("got action result", result)
    
    # drive forward with the cube for 10cm
    robot.drive_straight(distance_mm(100), speed_mmps(50)).wait_for_completed()
    
    # put down the cube
    action = robot.place_object_on_ground_here(cube)
    print("got action", action)
    result = action.wait_for_completed(timeout=30)
    print("got action result", result)
    
    # drive forward 10cm
    robot.drive_straight(distance_mm(-100), speed_mmps(50)).wait_for_completed()
    
    cube.set_light_corners(None, None, None, None)
    
# reference cozmo_sdk_examples_1.2.1/tutorials/01_basics/05_motors.py

def orderState(robot: cozmo.robot.Robot):
    
    # Tell Cozmo to drive the left wheel at 25 mmps (millimeters per second),
    # and the right wheel at 50 mmps (so Cozmo will drive Forwards while also
    # turning to the left
    robot.drive_wheels(25, 50)

    # wait for 3 seconds (the head, lift and wheels will move while we wait)
    time.sleep(3)
    
# reference cozmo_sdk_examples_1.2.1/tutorials/01_basics/04_drive_square.py

def inspectionState(robot: cozmo.robot.Robot):
    
    # Use a "for loop" to repeat the indented code 4 times
    # Note: the _ variable name can be used when you don't need the value
    for _ in range(4):
        robot.drive_straight(distance_mm(200), speed_mmps(50)).wait_for_completed()
        robot.turn_in_place(degrees(90)).wait_for_completed()
#         robot.set_lift_height(cozmo.robot.MAX_LIFT_HEIGHT, in_parallel=True, duration=2.5).wait_for_completed()
#         robot.set_lift_height(cozmo.robot.MIN_LIFT_HEIGHT, in_parallel=True, duration=2.5).wait_for_completed()


def finiteStateMachine(sdk_conn):
    clf = loadClassifier()

    # copied from collectImages.py
    robot = sdk_conn.wait_for_robot()
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()

    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # using the sliding window method
    images = np.array([])
    i = -1

    while True:
        # copied from collectImages.py
        latest_image = robot.world.latest_image
        new_image = latest_image.raw_image

        # Need to add one extra dimension here and remove it when getting the feature back.
        image_arr = np.array([np.array(new_image)])
        # comma here is a trick to remove the extra dimension
        feature_arr, = clf.extract_image_features(image_arr)

        if images.ndim <= 1:
            # Initialized with whatever the dimension of the feature data
            images = np.zeros((WINDOW_SIZE, *(feature_arr.shape) ))

        i += 1

        if i < WINDOW_SIZE:
            images[i] = feature_arr
            continue
        else:
            images[i % WINDOW_SIZE] = feature_arr
            labels = clf.predict_labels(images)

            # use major vote to decide the state found
            u, counts = np.unique(np.array(labels), return_counts = True)
            decision = u[np.argmax(counts)]

            if decision != IDLE:
                print(decision)
                # say the discovered state
                robot.say_text(decision).wait_for_completed()
                
                # switch into corresponding state
                if decision == DRONE:
                    cozmo.run_program(droneState)
                elif decision == ORDER:
                    cozmo.run_program(orderState)
                elif decision == INSPECTION:
                    cozmo.run_program(inspectionState)
                else:
                    print("Oops! Unexpected state decision!")
                print("Cozmo has just executed " + decision + "!")
                
                # reset the counter so that we can start over the image classification
                i = -1



def main():
    cozmo.setup_basic_logging()

    try:
        cozmo.connect(finiteStateMachine)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)

if __name__ == "__main__":
    main()

