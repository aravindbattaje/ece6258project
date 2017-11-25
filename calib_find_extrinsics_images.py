import argparse
import os
import cv2
import numpy as np
from utils.display import Display
from utils.config import LoadConfig
from utils.config import SaveConfig

# Thresholds to get the markers
# of the points (circles) on the table put
# in by the user. This makes sure we
# filter out the markers quickly.
MARKER_R_MIN_THRESH = 135
MARKER_R_MAX_THRESH = 145
MARKER_G_MIN_THRESH = 250
MARKER_G_MAX_THRESH = 255
MARKER_B_MIN_THRESH = 245
MARKER_B_MAX_THRESH = 255

# The physical locations of the standard
# positions on the ping pong table that
# the user has to mark. The "standard"
# positions are the vertices of  the quarter
# on the table which is opposite to the
# camera view.
# +++++++++++++++++++-------------------
# +                 +                   |
# +                 +                   |
# +++++++++++++++++++-------------------
# |                 |                   |
# |                 |                   |
# --------------------C------------------
#           C is current camera
#    + is the area marked for extrinsic
# The coordinates are all in meters.
TABLE_MARKERS = np.array([
    [0, 0, 0],
    [0, 1.37, 0],
    [0.7625, 0, 0],
    [0.7625, 1.37, 0]
])

# Coordinates of origin
# and three axes to help
# visualize projection.
AXIS_AT_ORIGIN = np.array([
    [0.25, 0, 0],
    [0, 0.25, 0],
    [0, 0, 0.25],
    [0, 0, 0]
])


def get_args():
    # Setup argument parser
    # and parse the inputs
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-i', '--image', required=True, help='Input marked image file')

    arg_parser.add_argument(
        '-m', '--model', required=True, help='Camera model - Pinhole (P)/Fisheye (F)'
    )

    args = vars(arg_parser.parse_args())

    if not os.path.exists(args['image']):
        arg_parser.error('{} file not found'.format(args['image']))

    if args['model'].upper() != 'P' and args['model'].upper() != 'F':
        arg_parser.error(
            '{} not supported. Enter P for Pinhole or F for Fisheye'.format(
                args['model']))

    return args


def main():
    args = get_args()

    # Read in intrinsic calibration configuration
    if not os.path.exists('config/intrinsic_calib_{}.npz'.format(
            args['model'].lower())):
        print 'First perform intrinsic calibration and save it in config'
        return
    load_config = LoadConfig('config/intrinsic_calib_{}.npz'.format(
        args['model'].lower()), 'calib')
    calib = load_config.load()

    # TODO: Error handling if image not readable by OpenCV
    img = cv2.imread(args['image'])

    # Threshold image to isolate the markers
    # put in by the user at the "standard" positions
    img_markers = cv2.inRange(img, (MARKER_B_MIN_THRESH, MARKER_G_MIN_THRESH, MARKER_R_MIN_THRESH),
                              (MARKER_B_MAX_THRESH, MARKER_G_MAX_THRESH, MARKER_R_MAX_THRESH))

    # Detect the circular markers
    circles = cv2.HoughCircles(img_markers, cv2.HOUGH_GRADIENT, 1.5,
                               10, param1=150, param2=15, minRadius=5, maxRadius=35)

    # Draw circles for visualization
    if circles is not None:
        # Make 'circles' easy to index
        circles = circles[0, :]

        # Check if we have exactly 4 markers
        if circles.shape[0] != 4:
            print 'Number of markers not 4'
            return

        for (xx, yy, rr) in circles:
            # Convert to ints
            x, y, r = int(xx), int(yy), int(rr)

            # Draw circle outlines
            cv2.circle(img, (x, y), r, (0, 0, 0), 2)

            # Draw dots at their centers
            cv2.rectangle(img, (x - 2, y - 2),
                          (x + 2, y + 2), (0, 128, 255), -1)

    # Find scaled intrinsics according to pinhole model
    if args['model'].upper() == 'P':
        # TODO: Complete this part out although its not required currently
        print 'Extrinsic calibration via pinhole model currently not supported.'
        return

    # Find scaled intrinsics according to fisheye model
    elif args['model'].upper() == 'F':
        # Make sure distortion coeffecients
        # follow fisheye model
        if calib['dist_coeffs'].shape[0] != 4:
            print 'Input configuration probably not fisheye'
            return

        # Harcoded image size as
        # this is a test script.
        # TODO: Change this later.
        img_size = (1920, 1080)

        # Create scaled intrinsics. The alpha parameter
        # in pinhole model is equivalent to balance parameter here.
        new_calib_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            calib['camera_matrix'], calib['dist_coeffs'], img_size, np.eye(3), balance=1)

    # Solve a point to point correspondence with
    # ideal points in the 3D world with projections
    # as we obtained from the user.
    # NOTE: Found it the hard way that solvePnP
    # unfortunately requires the image and object
    # points to be contiguous elements in the memory
    # This is probably to do with the method using
    # some old style pointers.
    obj_points = np.ascontiguousarray(TABLE_MARKERS.reshape((-1, 1, 3)))
    img_points = np.ascontiguousarray(circles[:, :2].reshape((-1, 1, 2)))
    rvecs, tvecs = cv2.solvePnPRansac(
        obj_points, img_points, new_calib_matrix, np.zeros((5, 1)))[1:3]

    # Project the origin of the table back
    # on to imager according to the found
    # extrinsics rotation and translation
    projected_points = cv2.projectPoints(
        AXIS_AT_ORIGIN, rvecs, tvecs, new_calib_matrix, np.zeros((5, 1)))[0]

    # Draw the axes at the origin on to image
    img = cv2.line(img, tuple(projected_points[0].ravel().astype(int)), tuple(
        projected_points[3].ravel().astype(int)), (255, 0, 0), 5)
    img = cv2.line(img, tuple(projected_points[1].ravel().astype(int)), tuple(
        projected_points[3].ravel().astype(int)), (0, 255, 0), 5)
    img = cv2.line(img, tuple(projected_points[2].ravel().astype(int)), tuple(
        projected_points[3].ravel().astype(int)), (0, 0, 255), 5)

    # Display the image with detected markers
    # and the projected axis.
    img_display = Display({'name': 'Image'})
    img_display.refresh(img)

    # Save extrinsics in the current folder
    save_config = SaveConfig('new_extrinsics', 'extrinsics')
    save_config.save(rvec=rvecs, tvec=tvecs)

    # Wait indefinitely
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
