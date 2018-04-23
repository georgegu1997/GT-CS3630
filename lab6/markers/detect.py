from skimage import io, feature, color, transform, filters, morphology, segmentation, measure, exposure, util, draw, future
import numpy as np
import random
import os
import math
import uuid

from . import pose


def image_for_filename(image_name):
    img = io.imread(image_name)
    img = color.rgb2gray(img)
    return img


def binary_image(image):
    return (image < filters.threshold_mean(image))


# Template images for the bottom left and top right corners of a merker
TEMPLATE_BOTTOM_LEFT = binary_image(image_for_filename(os.path.join(os.path.dirname(__file__), 'marker-template-bottom-left.png')))
TEMPLATE_TOP_RIGHT = binary_image(image_for_filename(os.path.join(os.path.dirname(__file__), 'marker-template-top-right.png')))

# Marker coordinate frame (mm)
MARKER_WIDTH_HALF = 96.5 / 2
MARKER_CORNERS_OBJECT_COORDS = np.array([
    [-MARKER_WIDTH_HALF, -MARKER_WIDTH_HALF, 0], # top-left
    [ MARKER_WIDTH_HALF, -MARKER_WIDTH_HALF, 0], # top-right
    [ MARKER_WIDTH_HALF,  MARKER_WIDTH_HALF, 0], # bottom-right
    [-MARKER_WIDTH_HALF,  MARKER_WIDTH_HALF, 0], # bottom-left
], dtype=np.float)

# Default camera intrinsic matrix
DEFAULT_CAMERA_INTRINSICS = np.array([
    [296.54,      0, 160],    # fx   0  cx
    [     0, 296.54, 120],    #  0  fy  cy
    [     0,      0,   1]     #  0   0   1
], dtype=np.float)

def estimate_corner_lines(corner_region, corner_type, image_size):
    '''
    Estimate the line parameters for the edges of the given marker corner

    Inputs:
        - corner_region: regionprops for the given corner of the marker
        - corner_type: either 'BL' for bottom-left or 'TR' for top-right
        - image_size: tuple of original camera image's size
    Return:
        A list of length 2 holding the line parameters of horizontal and vertical edges of the marker
    '''
    corner_miny, corner_minx, corner_maxy, corner_maxx = corner_region.bbox

    # Pad the corner image to match the size of the entire image
    corner_image = util.pad(
        corner_region.intensity_image,
        ((corner_miny, image_size[0] - corner_maxy), (corner_minx, image_size[1] - corner_maxx)),
        mode='constant',
        constant_values=0
    )

    # Perform edge detection and Hough line transform
    corner_edges = feature.canny(corner_image, sigma=0)
    corner_hough = transform.hough_line(corner_edges, theta=np.linspace(-np.pi/2, np.pi/2, 360))
    corner_hough_peaks = list(zip(*transform.hough_line_peaks(*corner_hough, min_angle=45, num_peaks=2)))

    corner_lines = []

    def is_horizontal(peak):
        return abs(np.rad2deg(peak[1])) > 75

    def is_vertical(peak):
        return abs(np.rad2deg(peak[1])) <= 15

    # Categorized the detected lines as vertical or horizontal
    horizontal_peaks = list(filter(is_horizontal, corner_hough_peaks))
    vertical_peaks = list(filter(is_vertical, corner_hough_peaks))

    # Add the first estimated horizontal line
    if horizontal_peaks:
        corner_lines.append(horizontal_peaks[0])
    else:
        # Create a horizontal line from the bottom edge (if BL type) or top edge (if TR type)
        angle = np.pi/2
        dist = corner_maxy if corner_type == 'BL' else corner_miny
        corner_lines.append((0, angle, dist))

    # Add the first estimated vertical line
    if vertical_peaks:
        corner_lines.append(vertical_peaks[0])
    else:
        # Create a vertical line from the left edge (if BL type) or right edge (if TR type)
        angle = 0.001
        dist = corner_minx if corner_type == 'BL' else corner_maxx
        corner_lines.append((0, angle, dist))

    return corner_lines


def intersection(hline1, hline2):
    """Finds the intersection of two lines given in Hesse normal form.

    Taken from:
        https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    """
    _, theta1, rho1 = hline1
    _, theta2, rho2 = hline2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return (x0, y0)


def unwarp_marker(image, left_region, right_region, width=320, height=240, margin=20):
    '''
    Using the four corners of the detected marker, estimate the inverse
    projective transform to obtain a cropped, "unwarped" view of the marker

    Input:
        - image: grayscale image that contains marker
        - left_region: regionprops corresponding to lower-left corner
        - right_region: regionprops corresponding to upper-right corner
        - width, height: output "unwarped" image dimensions
        - margin: increase the "unwarped" bounding box by the given margin
    Returns: the "unwarped" image that contains just the marker
    '''

    li = left_region.intensity_image
    ri = right_region.intensity_image

    left_miny, left_minx, left_maxy, left_maxx = left_region.bbox
    right_miny, right_minx, right_maxy, right_maxx = right_region.bbox

    # Compute the coordinates of the corners
    top_right = (right_maxx, right_miny)
    bottom_left = (left_minx, left_maxy)

    # Compute the coordinates of the other two corners by estimating edges
    corner_bl_lines = estimate_corner_lines(left_region, 'BL', image.shape)
    corner_tr_lines = estimate_corner_lines(right_region, 'TR', image.shape)
    bottom_right = intersection(corner_bl_lines[0], corner_tr_lines[1])
    top_left = intersection(corner_bl_lines[1], corner_tr_lines[0])

    corners = (top_left, top_right, bottom_right, bottom_left)

    # Estimate the transform
    m = margin
    src = np.array([
        [m, m],
        [width - m, m],
        [m, height - m],
        [width - m, height - m]
    ])
    dst = np.array([
        top_left,      # top left     -> (m, m)
        top_right,     # top right    -> (width - m, m)
        bottom_left,   # bottom left  -> (m, height - m)
        bottom_right   # bottom right -> (width - m, height - m)
    ])

    t = transform.ProjectiveTransform()
    t.estimate(src, dst)

    unwarped = transform.warp(image, t, output_shape=(height, width), mode='constant', cval=1.0)
    cropped = np.copy(image)[right_miny:left_maxy, left_minx:right_maxx]

    # Draw the original estimated bounds atop the regular image
    marked = color.gray2rgb(np.copy(image))
    draw.set_color(marked, draw.line(top_left[1], top_left[0], bottom_left[1], bottom_left[0]), (1.0, 0, 0))
    draw.set_color(marked, draw.line(top_left[1], top_left[0], top_right[1], top_right[0]), (1.0, 0, 0))
    draw.set_color(marked, draw.line(bottom_right[1], bottom_right[0], bottom_left[1], bottom_left[0]), (1.0, 0, 0))
    draw.set_color(marked, draw.line(bottom_right[1], bottom_right[0], top_right[1], top_right[0]), (1.0, 0, 0))
    draw.set_color(marked, draw.circle(top_left[1], top_left[0], 5), (0, 1.0, 0))
    draw.set_color(marked, draw.circle(top_right[1], top_right[0], 5), (0, 1.0, 0))
    draw.set_color(marked, draw.circle(bottom_left[1], bottom_left[0], 5), (0, 1.0, 0))
    draw.set_color(marked, draw.circle(bottom_right[1], bottom_right[0], 5), (0, 1.0, 0))

    return unwarped, cropped, marked, corners


def overlap_measure(image1, image2):
    '''
    Measures the ratio of pixels common to both images of the same size

    Inputs: two images of identical dimensions
    Returns: ratio of pixels in common
    '''
    overlap_area = np.count_nonzero(np.equal(image1, image2))
    total_area = image1.size

    return overlap_area / total_area

def region_filter_heuristic(region, orientation_deviation=15, overlap_minimum=0.8):
    '''
    For a given region, determines whether to consider it as a possible marker corner
    using a variety of factors:
        - large enough area
        - closeness to 45 degree orientation
        - similarity to corner templates

    Inputs:
        - region: regionprops to determine whether to consider or not
        - orientation_deviation: allowed deviation from 45 degree orientation for inclusion
        - overlap_minimum: minimum template similarity measure for inclusion
    Returns: true if region should be considered, false otherwise
    '''
    orientation = np.rad2deg(abs(region.orientation))
    area = region.area

    # Area must be large enough
    if area < 50:
        return False

    # The markers should have orientations close to 45 degrees
    if orientation < (45 - orientation_deviation) or orientation > (45 + orientation_deviation):
        return False

    # The markers should look like the templates
    region_image = region.intensity_image
    template_left = transform.resize(TEMPLATE_BOTTOM_LEFT, region_image.shape)
    template_right = transform.resize(TEMPLATE_TOP_RIGHT, region_image.shape)

    # Compute the overlap measure for both template images

    overlap_left = overlap_measure(region_image, template_left)
    overlap_right = overlap_measure(region_image, template_right)
    overlap_ratio = max(overlap_left, overlap_right)

    if overlap_ratio < overlap_minimum:
        return False

    return True


def select_best_pairs(regions):
    '''
    Returns pairs of regions determined to be the corners of a marker

    Input: list of (filtered) regionprops corresponding to possible marker corners
    Returns: list of 2-tuples, each being (left corner region, right corner region) if a
        candidate pair of corners for a marker is found
    '''
    pairs = []
    candidates = regions[:]

    # Find pairs in the candidate list of regions, until there are no more pairs to be found
    while len(candidates) >= 2:
        # Select the two best marker regions
        best_pair = None
        best_measure = 0

        # Compare each pair of regions
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                # Determine which region is the left and which is the right
                if candidates[i].bbox[1] < candidates[j].bbox[1]:
                    region_left_index = i
                    region_right_index = j
                else:
                    region_left_index = j
                    region_right_index = i

                region_left = candidates[region_left_index]
                region_right = candidates[region_right_index]

                # Left region should be strictly left+lower of the right region
                rl_min_y, rl_min_x, rl_max_y, rl_max_x = region_left.bbox
                rr_min_y, rr_min_x, rr_max_y, rr_max_x = region_right.bbox

                if rl_max_x > rr_min_x or rl_min_y < rr_max_y:
                    continue

                # Areas of both regions should be similar (closer to 1 means identical area)
                area_measure = min(region_left.area, region_right.area) / max(region_left.area, region_right.area)

                # The horizontal and vertical distances between the two regions should be similar (closer to 1 means perfect square)
                distance_horizontal = region_right.bbox[3] - region_left.bbox[1]
                distance_vertical = region_left.bbox[2] - region_right.bbox[0]
                distance_measure = min(distance_horizontal, distance_vertical) / max(distance_horizontal, distance_vertical)

                # If the width/height measure is too skewed, most likely a false marker detection
                if distance_measure < 0.5:
                    continue

                # Regions should be similar to the templates
                image_left = region_left.intensity_image
                image_right = region_right.intensity_image
                template_left = transform.resize(TEMPLATE_BOTTOM_LEFT, image_left.shape)
                template_right = transform.resize(TEMPLATE_TOP_RIGHT, image_right.shape)

                overlap_left_measure = overlap_measure(image_left, template_left)
                overlap_right_measure = overlap_measure(image_right, template_right)

                # If any overlap measure is too low, then false marker possibly detected
                if overlap_left_measure < 0.5 or overlap_right_measure < 0.5:
                    continue

                # Closer to 1 is better
                pair_measure = area_measure * distance_measure * overlap_left_measure * overlap_right_measure

                if pair_measure > best_measure:
                    best_pair = (region_left_index, region_right_index)
                    best_measure = pair_measure

        if best_pair:
            # Get the regions for the best pair
            region_left = candidates[best_pair[0]]
            region_right = candidates[best_pair[1]]
            pairs.append((region_left, region_right))

            # Remove the best pair from candidates on the next iteration
            candidates.remove(region_left)
            candidates.remove(region_right)
        else:
            # Break out of the loop since no pair will ever be found
            break

    return pairs


def process_regions(image, blur_sigma=0.01, opening_size=1, orientation_deviation=15, overlap_minimum=0.8):
    '''
    Attempt to find any possible marker corner regions in a given image

    Inputs:
        - image: grayscale image that may contain a marker
        - blur_sigma: parameter for Gaussian blur to use on image
        - opening_size: parameter for morphological opening to use on image
        - orientation_deviation: see orientation parameter used by region_filter_heuristic(...)
        - overlap_minimum: see similarity parameter used by region_filter_heuristic(...)
    Returns: a 2-tuple of:
        - the image after pre-processing steps like blurring, thresholding, etc.
        - the list of regionprops that may be possible marker corners
    '''
    # Blur and equalize the image
    image = exposure.equalize_hist(image)
    image = filters.gaussian(image, sigma=blur_sigma)

    # Use local thresholding
    image = (image <= filters.threshold_sauvola(image, k=0.1))

    # if opening_size > 0:
    #     image = morphology.erosion(image, selem=morphology.disk(opening_size))

    # Label components in the image
    labeled = measure.label(image, connectivity=2)
    components = measure.regionprops(labeled, intensity_image=image)

    # image_label_overlay = color.label2rgb(labeled, image=image)
    # filename = 'debug-images/{}.jpg'.format(uuid.uuid4())
    # io.imsave(filename, image*255)

    # Sort the components by our rectangle heuristic
    return image, labeled, [r for r in components if region_filter_heuristic(r, orientation_deviation, overlap_minimum)]


def estimate_marker_pose(image_corner_coords, camera_settings, opencv=False):
    '''
    Using the detected marker and the camera's setting, estimate the pose of the marker.

    Inputs:
        - image_coords: a list or tuple containing the 4 corner points of the marker
        - camera_settings: 3x3 matrix that encodes the camera focal lengths and center point
    Returns:
        - R: 3x3 rotation matrix of the marker in the camera coordinate frame
        - t: 3x1 translation vector of the marker in the camera coordinate frame
    '''
    # Solve the perspective n-point problem to estimate pose
    if not opencv:
        _, R, t = pose.estimate_pose(MARKER_CORNERS_OBJECT_COORDS, image_corner_coords, camera_settings)
    else:
        _, R, t = pose.estimate_pose_opencv(MARKER_CORNERS_OBJECT_COORDS, image_corner_coords, camera_settings)

    return R, t


def xyh_from_pose(R, t):
    ''' Use the rotation matrix and translation vector to compute the location and heading relative from the camera '''
    x = t[2] # in the grid, x is forward-backward
    y = -t[0] # in the grid, y is left-right

    # heading = np.rad2deg(math.atan2(R[1,0], R[0,0]))
    # heading = np.rad2deg(math.atan2(y, x))

    R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
    R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
    R_2p_1p = np.matmul(np.matmul(np.linalg.inv(R_2_2p), np.linalg.inv(R)), R_1_1p)
    yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0]) + math.pi
    heading = np.rad2deg(yaw)

    return x, y, heading


def detect_markers(image, camera_settings=DEFAULT_CAMERA_INTRINSICS, include_diagnostics=False, opencv=False):
    '''
    Attempts to detect markers in the given grayscale image.

    Since your Cozmo's camera may be calibrated (very) slightly differently from the default,
    you can create the `camera_settings` matrix for your Cozmo using its camera config (example below).
    Using your own Cozmo's camera intrinsic settings may (very slightly) improve the accuracy of
    the estimated location and heading of the detected markers.

    Input:
        - image: grayscale image that may contain markers
        - camera_settings: 3x3 matrix encoding the camera's intrinsic settings (focal lengths, principal point)
        - include_diagnostics:
            if false, this function returns only the list of detected markers
            if true, then it also returns a dictionary with diagnostic properties for debugging

    Returns:
        - list of marker detection dicts, each of which contain the following properties:
            {
                'pose': 2-tuple of the marker's estimated rotation matrix and translation vector, relative to the camera
                'xyh': 3-tuple (x, y, h) of the marker's estimated position (x, y) in millimeters and heading (h) in degrees, relative to the camera

                'corner_coords': 4-tuple of corner points of the marker (top left, top right, bottom right, bottom left)
                'cropped_image': image cropped to just contain the marker
                'unwarped_image': (320x240) image containing just the marker, but unwarped/unskewed
                'marked_image': (320x240) original image with marker boundary drawn on top
            }

        - if `include_diagnostics` is true, also returns a diagnostic dict (for debugging) with the following properties:
            {
                'regions': list of skimage.measure.regionprops, all the candidate regions in the image
                'filtered_image': image after processing (blurs, threshold, etc.)
            }

    Example:::

        fx, fy = robot.camera.config.focal_length.x_y
        cx, cy = robot.camera.config.center.x_y
        camera_settings = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float)
        #...
        markers = detect.detect_markers(image, camera_settings)
        x, y, h = markers['xyh']

    '''

    # Obtain the possible regions for the markers in the image
    filtered_image, labeled_image, marker_regions = process_regions(image)

    # If two components were not found after processing, relax the requirements
    if len(marker_regions) < 2:
        blur_sigma = 0.001
        opening_size = 0
        orientation_deviation = 25
        overlap_minimum = 0.65

        filtered_image, labeled_image, marker_regions = process_regions(image, blur_sigma, opening_size, orientation_deviation, overlap_minimum)

    # Keep the rectangle regions (sorted left to right)
    pairs = select_best_pairs(marker_regions)

    # Track the marker detections
    detections = []

    for pair in pairs:
        # Compute properties of the candidate marker
        left, right = pair
        unwarped, cropped, marked, corners = unwarp_marker(image, left, right)

        # Estimate the pose using the corner points and camera settings
        R, t = estimate_marker_pose(np.array(corners, dtype=np.float), camera_settings, opencv=opencv)
        xyh = xyh_from_pose(R, t)

        detections.append({
            'pose': (R, t),
            'xyh': xyh,

            'corner_coords': corners,
            'unwarped_image': unwarped,
            'cropped_image': cropped,
            'marked_image': marked
        })

    # Diagnostic properties
    if include_diagnostics:
        return (detections, {'regions': marker_regions, 'filtered_image': filtered_image})

    return detections
