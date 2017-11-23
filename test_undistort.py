import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os
from utils.config import LoadConfig
from utils.config import SaveConfig
from calib_generate_charuco import dictionary, board
from collections import deque
import numpy as np

play_or_pause = 'Pause'
seek_callback_action = False
MAX_ARUCO_IDS = 10
MAX_CHARUCO_IDS = 12


def seek_callback(value):
    global cur_seek_pos
    global seek_callback_action
    cur_seek_pos = value
    seek_callback_action = True


def playpause_callback(value):
    global play_or_pause
    if value == 0:
        play_or_pause = 'Pause'
    else:
        play_or_pause = 'Play'


def setup_trackbars(window_name, trackbar_config=None):
    window = cv2.namedWindow(window_name, 0)

    cv2.createTrackbar('Seek', window_name, 0, 100, seek_callback)
    cv2.createTrackbar('Playback', window_name, 0, 1, playpause_callback)


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

    # Read in configuration
    load_config = LoadConfig('new_calib_{}.npz'.format(
        args['model'].lower()), 'calib')
    calib = load_config.load()

    # Setup video displays
    video_disp = Display({'name': 'Video'})

    # Setup controls
    setup_trackbars('Controls')  # , thresholds)

    # Get input video
    video = Video(args['video'])
    num_frames = video.get_num_frames()

    # Get the first frame to start with
    frame = video.next_frame()

    global seek_callback_action

    while True:
        if play_or_pause == 'Play':
            if not seek_callback_action:
                frame = video.next_frame()
            else:
                frame = video.get_frame(cur_seek_pos * num_frames / 100)
                seek_callback_action = False

        if video.end_reached():
            # Wait indefinitely if end of video reached
            # Or until keypress and then exit
            cv2.waitKey(0)
            break

        # Undistort according to pinhole model
        if args['model'].upper() == 'P':
            # Make sure distortion coeffecients
            # follow pinhole model
            if calib['dist_coeffs'].shape[1] != 5:
                print 'Input configuration probably not pinhole'
                return

            # Harcoded image size as
            # this is a test script
            img_size = (1080, 1920)

            # First create scaled intrinsics because we will undistort
            # into region beyond original image region
            new_calib_matrix, _ = cv2.getOptimalNewCameraMatrix(
                calib['camera_matrix'], calib['dist_coeffs'], img_size, 1)

            # Then calculate new image size according to the scaling
            # Unfortunately the Python API doesn't directly provide the
            # the new image size. They forgot?
            new_img_size = (int(img_size[0] + (new_calib_matrix[1, 2] - calib['camera_matrix'][1, 2])), int(
                img_size[1] + (new_calib_matrix[0, 2] - calib['camera_matrix'][0, 2])))

            # Standard routine of creating a new rectification
            # map for the given intrinsics and mapping each
            # pixel onto the new map with linear interpolation
            map1, map2 = cv2.initUndistortRectifyMap(
                calib['camera_matrix'], calib['dist_coeffs'], None, new_calib_matrix, new_img_size, cv2.CV_16SC2)
            img_undistorted = cv2.remap(
                frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Undistort according to fisheye model
        elif args['model'].upper() == 'F':
            # Make sure distortion coeffecients
            # follow fisheye model
            if calib['dist_coeffs'].shape[0] != 4:
                print 'Input configuration probably not fisheye'
                return

            # Harcoded image size as
            # this is a test script.
            # As already ranted before
            # someone messed with the image
            # size indexing and reversed it.
            img_size = (1920, 1080)

            # Also, the basic undistortion DOES NOT work
            # with the fisheye module
            # img_undistorted = cv2.fisheye.undistortImage(
            #   frame, calib['camera_matrix'], calib['dist_coeffs'])

            # First create scaled intrinsics because we will undistort
            # into region beyond original image region. The alpha
            # parameter in pinhole model is equivalent to balance parameter here.
            new_calib_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                calib['camera_matrix'], calib['dist_coeffs'], img_size, np.eye(3), balance=1)

            # Then calculate new image size according to the scaling
            # Well if they forgot this in pinhole Python API,
            # can't complain about Fisheye model. Note the reversed
            # indexing here too.
            new_img_size = (int(img_size[0] + (new_calib_matrix[0, 2] - calib['camera_matrix'][0, 2])), int(
                img_size[1] + (new_calib_matrix[1, 2] - calib['camera_matrix'][1, 2])))

            # Standard routine of creating a new rectification
            # map for the given intrinsics and mapping each
            # pixel onto the new map with linear interpolation
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                calib['camera_matrix'], calib['dist_coeffs'], np.eye(3), new_calib_matrix, new_img_size, cv2.CV_16SC2)
            img_undistorted = cv2.remap(
                frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Update GUI with new image
        video_disp.refresh(img_undistorted)

        # Add quitting event
        if video_disp.can_quit():
            break


if __name__ == '__main__':
    main()
