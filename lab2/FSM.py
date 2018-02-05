import pickle
import cozmo
import time
import numpy as np
from imgclassification import ImageClassifier

IDLE = "none"
DRONE = "drone"
ORDER = "order"
INSPECTION = "inspection"
WINDOW_SIZE = 5

# Load the pre-trained classifer from file
# https://blog.oldj.net/2010/05/26/python-pickle/
def loadClassifier(f = "clf.txt"):
    return pickle.load(open(f, "rb"))

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

            if decision != "none":
                print(decision)
                # say the discovered state
                robot.say_text(decision).wait_for_completed()
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
