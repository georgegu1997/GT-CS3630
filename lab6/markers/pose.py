import numpy as np
import math

def estimate_pose_opencv(object_coords, image_coords, camera_intrinsics, camera_distortion=np.zeros((4,1)), cv2_flag=None):
    import cv2
    success, r, t = cv2.solvePnP(object_coords, image_coords, camera_intrinsics, camera_distortion, cv2_flag or cv2.SOLVEPNP_P3P)
    
    # Convert the rotation vector into a matrix
    R, _ = cv2.Rodrigues(r)
    t = np.squeeze(t)

    return success, R, t


def estimate_pose(object_coords, image_coords, camera_intrinsics, camera_distortion=np.zeros((4,1))):
    '''
    Computes the 3D pose of the object relative to the camera, using 4 corresponding points in 
    the object space and the image space.

        - object_coords: 4x3 matrix, four 3D points of the object in its own coordinate frame
        - image_coords: 4x2 matrix, four 2D points of the object in the image coordinate frame
        - camera_intrinsics: (3, 3) matrix encoding the camera intrinsics (focal length, principal point)
        - camera_distortion (4,1) matrix encoding the camera distortion (**CURRENTLY UNUSED**)
    Returns:
        - success: bool, whether the PnP estimation succeeded
        - R: 4x3 rotation matrix of the object in the camera/world coordinate frame
        - t: 3-vector of the object's position in the camera/world coordinate frame
            
    Notes:

    Uses an implementation of the Perspective-3-Point problem, which requires 4 point correspondences
    between the image space and the object space to determine the best pose of the object
    in the camera/world space.

    Steps:
        0. Select 3 object/image point correspondences
        1. Use the camera intrinsics to normalize the 3 points in the image plane
        2. Pre-compute values related to law of cosines needed to solve the P3P system [1]
        3. Solve the quartic P3P system (this results in at most 4 possible pose estimates)
        4. Select the best pose estimate by finding which results in the smallest reprojection error 
           on the 4th point:
            - Given a pose estimate, determine the location of the 4th point in the world space using that pose
            - Using the camera intrinsics, project the estimated 4th point from world space into image space
            - Compute the reprojection error, the distance between the actual 4th image point and the 
              image space projection of the estimated 4th point
            - The pose estimate that resulted in the smallest reprojection error is the best estimate

    References:
        1. "The P3P (Perspective-Three-Point) Principle": 
            - See: http://iplimage.com/blog/p3p-perspective-point-overview/
            - This guide provides a good step-by-step procedure for solving the P3P problem
              (based on the original paper at http://www.mmrc.iss.ac.cn/~xgao/paper/ieee.pdf).
              Steps 1 and 2 below are adapted from this guide, but since the coefficients in the
              quartic solving stage are typeset and difficult to copy, Steps 3 and 4 are adapted 
              from OpenCV source code

        2. OpenCV 3 source (p3p.cpp):
            - See: https://github.com/opencv/opencv/blob/master/modules/calib3d/src/p3p.cpp#L206
            - The OpenCV P3P implements the same approach as described in "The P3P Principle",
              so Steps 3 and 4 are taken from this existing code, modified to convert from
              C++ to Python and with adapted variable names

    '''

    ##
    ## Step 1: normalize image points (undistort first)
    ##
    fx, fy, cx, cy = (camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0,2], camera_intrinsics[1,2])
    U, V, W = [normalize_image_point(p, fx, fy, cx, cy) for p in image_coords[:3]]

    ##
    ## Step 2: prepare P3P system of equations
    ##

    cos_uv = np.dot(U, V)
    cos_uw = np.dot(U, W)
    cos_vw = np.dot(V, W)

    A = np.array(object_coords[0])
    B = np.array(object_coords[1])
    C = np.array(object_coords[2])

    dist_AB = np.linalg.norm(B-A)
    dist_AC = np.linalg.norm(C-A)
    dist_BC = np.linalg.norm(C-B)

    a = (dist_BC**2) / (dist_AB**2)
    b = (dist_AC**2) / (dist_AB**2)

    ##
    ## Step 3: solve P3P system
    ##

    solutions = []

    p = 2*cos_vw
    q = 2*cos_uw
    r = 2*cos_uv

    a2 = a*a
    b2 = b*b
    p2 = p*p
    q2 = q*q
    r2 = r*r
    pr = p*r
    pqr = q*pr
    ab = a*b
    a_2 = 2*a
    a_4 = 4*a

    # Solve for x in the quartic, Ax^4 + Bx^3 + Cx^2 + Dx + E = 0
    A = -2 * b + b2 + a2 + 1 + ab*(2 - r2) - a_2;
    B = q*(-2*(ab + a2 + 1 - b) + r2*ab + a_4) + pr*(b - b2 + ab);
    C = q2 + b2*(r2 + p2 - 2) - b*(p2 + pqr) - ab*(r2 + pqr) + (a2 - a_2)*(2 + q2) + 2;
    D = pr*(ab-b2+b) + q*((p2-2)*b + 2 * (ab - a2) + a_4 - 2);
    E = 1 + 2*(b - a - ab) + b2 - b*p2 + a2;

    xs = poly_solve_quartic(A, B, C, D, E)
    xs = np.real(xs[np.isreal(xs)])    

    # Solve for y in the equation, b1*y - b0 = 0
    temp = (p2*(a-1+b) + r2*(a-1-b) + pqr - a*pqr)
    b0 = b * temp * temp

    r3 = r2*r
    pr2 = p*r2
    r3q = r3 * q

    for x in xs:
        if x <= 0:
            continue

        x2 = x*x
        b1 = (
            ((1-a-b)*x2 + (q*a-q)*x + 1 - a + b) *
            (((r3*(a2 + ab*(2 - r2) - a_2 + b2 - 2*b + 1)) * x +
            (r3q*(2*(b-a2) + a_4 + ab*(r2 - 2) - 2) + pr2*(1 + a2 + 2*(ab-a-b) + r2*(b - b2) + b2))) * x2 +
            (r3*(q2*(1-2*a+a2) + r2*(b2-ab) - a_4 + 2*(a2 - b2) + 2) + r*p2*(b2 + 2*(ab - b - a) + 1 + a2) + pr2*q*(a_4 + 2*(b - ab - a2) - 2 - r2*b)) * x +
            2*r3q*(a_2 - b - a2 + ab - 1) + pr2*(q2 - a_4 + 2*(a2 - b2) + r2*b + q2*(a2 - a_2) + 2) +
            p2*(p*(2*(ab - a - b) + a2 + b2 + 1) + 2*q*r*(b + a_2 - a2 - ab - 1)))
        )

        if b1 <= 0:
            continue

        y = b1 / b0
        v = x2 + y*y - x*y*r

        if v <= 0:
            continue

        Z = dist_AB / np.sqrt(v)
        X = x * Z
        Y = y * Z

        solutions.append(np.array((U*X, V*Y, W*Z)))

    ##
    ## Step 4: pick the best translation vector
    ##

    min_reprojection = np.inf
    best_R = None
    best_t = None
    success = False
    
    all_t = []

    for s in solutions:
        # Compute the rotation matrix and translation vectors for this solution
        R, t = rigid_transform_3D(object_coords[:3], s)
        D = object_coords[3][:, np.newaxis]

        Dcamera = np.matmul(R, D)[:, 0] + t
        Dproj = np.array([cx + fx*Dcamera[0]/Dcamera[2], cy + fy*Dcamera[1]/Dcamera[2]])
        reprojection_error = np.linalg.norm(Dproj - image_coords[3])

        if reprojection_error < min_reprojection:
            min_reprojection = reprojection_error
            best_R = R
            best_t = t
            success = True
            
        all_t.append((t, reprojection_error, Dproj, image_coords[3]))

    return success, best_R, best_t  #, all_t


def normalize_image_point(p, fx, fy, cx, cy):
    ''' Normalize a 2D image point using the camera's focal lengths and center point '''
    
    px, py = p

    # Use camera intrinsics to adjust image point
    px_ = (px - cx) / fx
    py_ = (py - cy) / fy
    pz_ = 1

    # Normalize
    P = np.array([px_, py_, pz_])
    N = np.linalg.norm(P)

    return P/N


def rigid_transform_3D(A, B):
    '''
    Computes the rotation matrix and translation vector needed to 
    make a correspondence from A to B.
    
    Lightly modified from http://nghiaho.com/?page_id=671
    '''

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.matmul(Vt.T, U.T)

    t = np.matmul(-R, centroid_A.T) + centroid_B.T

    return R, t


def poly_solve_quartic(a, b, c, d, e):
    return np.roots([a, b, c, d, e])