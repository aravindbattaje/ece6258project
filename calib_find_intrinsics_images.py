import cv2
import argparse
import os
import glob
from utils.config import SaveConfig
from calib_generate_charuco import dictionary, board
import numpy as np

# Dependent on number of markers
# in the dictionary. Make this
# programmatic later than just
# hardcoding it.
MAX_ARUCO_IDS = 40
MAX_CHARUCO_IDS = 63


def get_args():
    # Setup argument parser
    # and parse the inputs
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-v', '--video', required=True, help='Input video file')

    arg_parser.add_argument(
        '-m', '--model', required=True, help='Camera model - Pinhole (P)/Fisheye (F)'
    )

    args = vars(arg_parser.parse_args())

    if not os.path.exists(args['video']):
        arg_parser.error('{} file not found'.format(args['video']))

    if args['model'].upper() != 'P' and args['model'].upper() != 'F':
        arg_parser.error(
            '{} not supported. Enter P for Pinhole or F for Fisheye'.format(
                args['model']))

    return args


def main():
    args = get_args()

    # Get video file name
    video_file_name = os.path.basename(args['video'])

    # Check for calib_images path
    if not os.path.exists('calib_images'):
        print '"calib_images" folder required'
        return

    # Get all images with video file name as prefix
    video_name_as_prefix = video_file_name.lower().strip('.mp4')
    calib_img_paths = glob.glob(
        'calib_images/{}_*[!markers].png'.format(video_name_as_prefix))

    # List to store all completely detected chArUco markers
    charuco_corners_list = []
    charuco_ids_list = []

    print 'Processing {} images to find markers'.format(len(calib_img_paths))
    for calib_img_path in calib_img_paths:
        img = cv2.imread(calib_img_path)

        # Detect just the ArUco (fiducial) markers in the image with
        # the same dictionary we used to generate the board (imported)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
            img, dictionary)

        # Refine the detected markers to find the missing
        # IDs. Required when we have boards with lots of IDs.
        corners, ids = cv2.aruco.refineDetectedMarkers(
            img, board, corners, ids, rejected_img_points)[:2]

        # If any markers in the dictionary are found, go ahead to
        # find the chessboard around the markers. Also, draw
        # the detected markers back on to the image.
        if ids is not None:
            img_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
            num_charuco, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, board)

            if charuco_corners is not None:
                img_markers = cv2.aruco.drawDetectedCornersCharuco(
                    img_markers, charuco_corners, charuco_ids)

                if ids.shape[0] == MAX_ARUCO_IDS \
                        and num_charuco == MAX_CHARUCO_IDS:
                    charuco_corners_list.append(charuco_corners)
                    charuco_ids_list.append(charuco_ids)
        else:
            img_markers = img

        # Write the image with markers back into the same folder
        cv2.imwrite(calib_img_path.replace(
            '.png', '_markers.png'), img_markers)

    print 'Found reliable markers in {}/{} images'.format(
        len(charuco_corners_list), len(calib_img_paths))

    # Pinhole model
    # The nice calibration routine along with chArUco
    # is used. It is fairly straight forward. Note the
    # the distortion coefficients for pinhole model
    # are [k1, k2, p1, p2, k3]
    if args['model'].upper() == 'P':
        img_size = img.shape[:2]
        error, camera_matrix, dist_coeffs = cv2.aruco.calibrateCameraCharuco(
            charuco_corners_list, charuco_ids_list, board, img_size, None, None)[:3]

    # Fisheye model
    # The not-so-nice fisheye contrib module is used.
    # Note the distortion coefficients for fisheye
    # model are [k1, k2, k3, k4]
    elif args['model'].upper() == 'F':
        # Make the detected chArUco corners into raw form
        # The object points are expected to be in 3D. So just
        # add another dummy 2nd dimension, and repeat the same
        # object points for each case.
        obj_points, img_points = [board.chessboardCorners.reshape(
            1, -1, 3)] * len(charuco_corners_list), charuco_corners_list

        # For whatever f***ing reason someone reversed the indexing
        # for image dimensions in the API. WHY???
        img_size = (img.shape[1], img.shape[0])

        # Dummy rvecs and tvecs to satisfy the calibration routine.
        # Well, another reason API is not-so-nice
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64)
                 for i in range(len(charuco_corners_list))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64)
                 for i in range(len(charuco_corners_list))]

        test_calib_matrix = np.array([
            [680., 0., 960.],
            [0, 680., 540.],
            [0, 0, 1]
        ])
        test_calib_matrix_orig = np.copy(test_calib_matrix)

        # The quite important calibration flags
        #   CALIB_RECOMPUTE_EXTRINSIC forces extrinsic to be
        #   recalculated for each iteration of intrinsic optimization
        #   CALIB_FIX_SKEW fix skew in instrinsics to zero
        #   CALIB_FIX_PRINCIPAL_POINT fix it to center of the
        #   imager. This and last one important, or else
        #   we'll need too many poses to constrain.
        #   CALIB_CHECK_COND makes sure we didn't mess up with
        #   any of the inputs.
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + \
            cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT + cv2.fisheye.CALIB_CHECK_COND 

        # Finally, the monster. We allow termination criteria
        # to be either max iterations of 100 or epsilon of 1e-6
        error, camera_matrix, dist_coeffs = cv2.fisheye.calibrate(
            obj_points, img_points, img_size,
            np.zeros((3, 3)), np.zeros((4, 1)),
            rvecs, tvecs, calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))[:3]

    print 'Calibrated camera in the {} model with {} RMS error'.format(
        args['model'], error)

    # Save instrinsics in the current folder
    save_config = SaveConfig('new_calib_{}'.format(
        args['model'].lower()), 'calib')
    save_config.save(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


if __name__ == '__main__':
    main()
