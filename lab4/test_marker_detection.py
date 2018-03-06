import cozmo
import time
import numpy as np
from skimage import color
from PIL import ImageDraw, ImageFont
from markers import detect, annotator

stop = False

def run(robot: cozmo.robot.Robot):
    '''The run method runs once Cozmo is connected.'''
    global stop
    
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    
    marker_annotator = annotator.MarkerAnnotator(robot.world.image_annotator)
    robot.world.image_annotator.add_annotator('Marker', marker_annotator)

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)
 
    while not stop: 
        time.sleep(0.05)
        
        if not robot.world.latest_image:
            continue
            
        # Get the latest image from Cozmo and convert it to grayscale
        image = np.array(robot.world.latest_image.raw_image)
        image = color.rgb2gray(image)
        
        # Detect the marker
        markers = detect.detect_markers(image, camera_settings)

        # Show the markers on the image window
        marker_annotator.markers = markers
        
        # Process each marker
        for marker in markers:
            
            # Get the cropped, unwarped image of just the marker
            marker_image = marker['unwarped_image']

            # ...
            # label = my_classifier_function(marker_iamge)
            
            # Get the estimated location/heading of the marker
            pose = marker['pose']
            x, y, h = marker['xyh']

            # print('X: {:0.2f} mm'.format(x))
            # print('Y: {:0.2f} mm'.format(y))
            # print('H: {:0.2f} deg'.format(h))
            # print()
    
try:
    stop = False
    cozmo.run_program(run, use_viewer=True)
except cozmo.exceptions.ConnectionError:
    print('Could not connect to Cozmo')
except KeyboardInterrupt:
    print('Stopped by user')
    stop = True