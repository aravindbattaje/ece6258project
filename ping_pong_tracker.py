# To force floating point division
from __future__ import division

# Standard libs
import argparse
import os
import sys

# Parallel processing
import multiprocessing
import threading
from collections import deque

# For calculations
import numpy as np
import cv2

# For visualization
from mayavi import mlab

# Utilities
from utils.display import Display
from utils.input import Video
from utils.config import LoadConfig

# Custom
import mean_shift_tracking_frame
from histogram_comparison import histogram_comparison

PING_PONG_DIAMETER = 40  # mm
DEFAULT_FOCAL_LENGTH = 14  # mm

# Main worker of which 2
# instances will be called
# Processes video from each camera
# to work out ball position independently


def main_worker(id, video_file, camera_model, K, D, R, T, measurements, quit_event):
    # Setup video displays
    video_disp = Display({'name': 'Camera_{}'.format(id)})

    # Get input video
    video = Video(video_file)

    # Setup the undistortion stuff
    if camera_model == 'P':
        # Harcoded image size as
        # this is a test script
        img_size = (1920, 1080)

        # First create scaled intrinsics because we will undistort
        # into region beyond original image region
        new_K = cv2.getOptimalNewCameraMatrix(K, D, img_size, 0.35)[0]

        # Then calculate new image size according to the scaling
        # Unfortunately the Python API doesn't directly provide the
        # the new image size. They forgot?
        new_img_size = (int(img_size[0] + (new_K[0, 2] - K[0, 2])), int(
            img_size[1] + (new_K[1, 2] - K[1, 2])))

        # Standard routine of creating a new rectification
        # map for the given intrinsics and mapping each
        # pixel onto the new map with linear interpolation
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, new_img_size, cv2.CV_16SC2)

    elif camera_model == 'F':
        # Harcoded image size
        img_size = (1920, 1080)

        # First create scaled intrinsics because we will undistort
        # into region beyond original image region. The alpha
        # parameter in pinhole model is equivalent to balance parameter here.
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, img_size, np.eye(3), balance=1)

        # Then calculate new image size according to the scaling
        # Well if they forgot this in pinhole Python API,
        # can't complain about Fisheye model. Note the reversed
        # indexing here too.
        new_img_size = (int(img_size[0] + (new_K[0, 2] - K[0, 2])), int(
            img_size[1] + (new_K[1, 2] - K[1, 2])))

        # Standard routine of creating a new rectification
        # map for the given intrinsics and mapping each
        # pixel onto the new map with linear interpolation
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, new_img_size_1, cv2.CV_16SC2)

    # Set up foreground and background separation
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Averaging kernel that will be used in opening
    kernel = np.ones((6, 6), np.uint8)

    # Code commented out because not using
    # confidence currently, but could be 
    # used again with changes later
    # # Will be used for histogram comparison
    # # (Confidence measure)
    # ball_image_file = 'ball_image.jpg'
    # ball_image = cv2.imread(ball_image_file)


    # 2D ball detection and 3D ball tracking setup
    ball_position_frame = None
    ball_wc = [0, 0, 0]

    while not video.end_reached() and not quit_event.value:
        # Get each frame
        frame = video.next_frame()

        # Undistort the current frame
        img_undistorted = cv2.remap(
            frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Convert to HSV and threshold range of ball
        img_hsv = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, np.array(
            (15, 190, 200)), np.array((25, 255, 255)))

        # Foreground and background separation mask
        fgmask = fgbg.apply(img_undistorted)
        mask_color_bgs = cv2.bitwise_and(mask, mask, mask=fgmask)
        masked_and_opened = cv2.morphologyEx(
            mask_color_bgs, cv2.MORPH_OPEN, kernel)

        # Hough transform to detect ball (circle)
        circles = cv2.HoughCircles(masked_and_opened, cv2.HOUGH_GRADIENT, dp=3,
                                   minDist=2500, param1=300, param2=5, minRadius=3, maxRadius=30)
        if circles is not None:
            # Make indexing easier and
            # convert everything to int
            circles = circles[0, :]
            circles = np.round(circles).astype("int")

            # Take only the first
            # (and hopefully largest)
            # circle detected
            x, y, r = circles[0]
            ball_position_frame = [x - r, y - r, 2 * r, 2 * r]
        else:
            ball_position_frame = None

        # Determine the correct ball radius
        mask_ball_radius = cv2.bitwise_and(fgmask, fgmask, mask=cv2.inRange(
            img_hsv, np.array((10, 150, 180)), np.array((40, 255, 255))))
        if ball_position_frame:
            x1, y1, w1, h1 = ball_position_frame
            ball_crop_temp = mask_ball_radius[(
                y1 + h1 // 2 - 50):(y1 + h1 // 2 + 50), (x1 + w1 // 2 - 50):(x1 + w1 // 2 + 50)]   
            height, width = ball_crop_temp.shape
            if height and width:
                # Successfully cropped image
                ball_crop = ball_crop_temp
                cnts = cv2.findContours(
                    ball_crop.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                center = None
                if len(cnts) > 0:
                    # contour detected
                    c = max(cnts, key=cv2.contourArea)
                    ellipse = cv2.fitEllipse(c)
                    width = min(ellipse[1])
                    ball_position_frame = [
                        ball_position_frame[0], ball_position_frame[1], 2 * width, 2 * width]

                # Code commented out because not using
                # confidence currently, but could be 
                # used again with changes later
                # # Calculate confidence
                # confidence = histogram_comparison(ball_image, img_undistorted, ball_position_frame)
                # print confidence

        if ball_position_frame:
            x1, y1, w1, h1 = ball_position_frame
            pixels_per_mm = (
                K[0, 0] + K[1, 1]) / 2 / DEFAULT_FOCAL_LENGTH
            z = PING_PONG_DIAMETER * \
                DEFAULT_FOCAL_LENGTH / (w1 / pixels_per_mm)
            x = ((x1 - K[0, 2]) /
                 pixels_per_mm) * z / DEFAULT_FOCAL_LENGTH
            y = ((y1 - K[1, 2]) /
                 pixels_per_mm) * z / DEFAULT_FOCAL_LENGTH

            ball_cc = np.array([x, y, z]) / 1000
            ball_wc = np.dot(R.T, ball_cc - T.ravel())

        # Push measurements to be processed/visualized
        measurement = {
            'id': id,
            'frame_num': video.get_cur_frame_num(),
            'ball_ic': ball_position_frame,
            'ball_wc': ball_wc
        }
        measurements.put(measurement)

        # Update video display
        video_disp.refresh(img_undistorted)

        # Add quitting event
        if video_disp.can_quit():
            break

    # Setting this will signal
    # the other parallel process
    # to exit too.
    quit_event.value = 1


def update_visu(measurements_camera_1, measurements_camera_2, visu, quit_event):
    # Setup a little local deck of measurements
    # so that this thread maybe continued in
    # case of 1 or 2 frames out of sync
    deck_camera_maxlen = 10
    deck_camera_1 = deque(maxlen=deck_camera_maxlen)
    deck_camera_2 = deque(maxlen=deck_camera_maxlen)

    # Setup a deck to store ball trail
    deck_ball_maxlen = 5
    deck_ball = deque(maxlen=deck_ball_maxlen)

    # Keep track of latest frame processed
    latest_frame_processed = 0

    # Setup balls last known position as some
    # default "known" position; near player 1
    last_known_position = [0, 137, 50]

    # Rotation for measurements from camera 2
    def camera_2_to_camera_1(wc):
        # Rotation matrix will be
        # -1 0 0
        # 0 -1 0
        # 0  0 1
        # since it is just a flip on the Z-axis
        wc[0] = -wc[0]
        wc[1] = -wc[1]
        # No change to wc[2]
        return wc

    while not quit_event.value:
        deck_camera_1.append(measurements_camera_1.get())
        deck_camera_2.append(measurements_camera_2.get())

        # Sequentially look for the next frame
        # to process from camera 1
        for i in range(deck_camera_maxlen):
            if deck_camera_1[i]['frame_num'] > latest_frame_processed:
                camera_1 = deck_camera_1[i]
                break

        # Find the same frame in camera 2
        for i in range(deck_camera_maxlen):
            if deck_camera_2[i]['frame_num'] == camera_1['frame_num']:
                camera_2 = deck_camera_2[i]
                break

        # If ball in image coordinates available
        # from either camera update as predicted from either.
        # But if both available, predict the average
        if camera_1['ball_ic'] and camera_2['ball_ic']:
            ball_wc = (camera_1['ball_wc'] +
                       camera_2_to_camera_1(camera_2['ball_wc'])) / 2
        elif camera_1['ball_ic']:
            ball_wc = camera_1['ball_wc']
        elif camera_2['ball_ic']:
            ball_wc = camera_2_to_camera_1(camera_2['ball_wc'])
        else:
            ball_wc = None

        # Store a history of ball measurements
        deck_ball.append(ball_wc)

        # Update latest frame processed
        latest_frame_processed = camera_1['frame_num']

        # Process the ball deck to visualize a ball trail
        # If there are maxlen continuous (valid) measurements,
        # form the entire trail. Or else form part of the trail.
        # Also NOTE the reversed indices
        visu_scalars = np.zeros(deck_ball_maxlen)
        visu_x = np.zeros_like(visu_scalars)
        visu_y = np.zeros_like(visu_scalars)
        visu_z = np.zeros_like(visu_scalars)
        for i in sorted(range(len(deck_ball)), reverse=True):
            if deck_ball[i] is None:
                break

            # The table is in cm units
            visu_x[i] = deck_ball[i][0] * 100
            visu_y[i] = deck_ball[i][1] * 100
            visu_z[i] = deck_ball[i][2] * 100

            # Update last known position
            last_known_position = deck_ball[i] * 100

        # Scale the ball sizes depending
        # on number of measurements we got
        scaling_template = [16, 12, 10, 6, 0]
        visu_scalars[i:] = scaling_template[i:][::-1]

        # If no valid measurements, then
        # update visu with last known position
        # but with smaller representation
        if i == len(deck_ball) - 1:
            visu_x[i] = last_known_position[0]
            visu_y[i] = last_known_position[1]
            visu_z[i] = last_known_position[2]
            visu_scalars[i] = 4

        # Update ball in visu
        visu.set(x=visu_x, y=visu_y, z=visu_z, scalars=visu_scalars)

    # If thread wants to exit, should ideally
    # take down visu window with it. But there
    # seems to be some problem with the close
    # function in mayavi. Ask user to manually
    # close that window
    print 'Please close the visualization window too if not done already'


# Argument parser
def get_args():
    # Setup argument parser
    # and parse the inputs
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-v1', '--video-1', required=True, help='Video file from Camera 1')

    arg_parser.add_argument(
        '-v2', '--video-2', required=True, help='Video file from Camera 2')

    arg_parser.add_argument(
        '-m', '--model', default='P', help='Camera model - Pinhole (P)/Fisheye (F)'
    )

    args = vars(arg_parser.parse_args())

    if not os.path.exists(args['video_1']):
        arg_parser.error('{} file not found'.format(args['video_1']))

    if not os.path.exists(args['video_2']):
        arg_parser.error('{} file not found'.format(args['video_2']))

    if args['model'].upper() != 'P' and args['model'].upper() != 'F':
        arg_parser.error(
            '{} not supported. Enter P for Pinhole or F for Fisheye'.format(
                args['model']))

    return args

# Setup table and other stuff
# for visualization


def setup_visu():
    # Create a Mayavi figure
    fig = mlab.figure('Ball tracking visualization')

    # Measurements have to be whole numbers
    TABLE_WIDTH = 152  # cm
    TABLE_LENGTH = 274  # cm
    table_structure = np.ones((TABLE_WIDTH, TABLE_LENGTH))

    # Draw out the lines on the table
    # Creases
    table_structure[1:-2, 1] = 0
    table_structure[1:-2, -2] = 0
    table_structure[1, 1:-2] = 0
    table_structure[-2, 1:-2] = 0

    # Center line
    table_structure[TABLE_WIDTH // 2, 1:-2] = 0

    # Representing net
    table_structure[1:-2, TABLE_LENGTH // 2 - 1] = 0.4
    table_structure[1:-2, TABLE_LENGTH // 2] = 0.4
    table_structure[1:-2, TABLE_LENGTH // 2 + 1] = 0.4

    # Create the table layout in the figure
    table = mlab.imshow(table_structure, colormap='Blues', opacity=0.6)

    # Setup a 3D point plotter with
    # proper shapes, ie., no. of points
    # for the ball and its trail
    x = y = z = s = np.zeros(5)
    point_plt = mlab.points3d(x, y, z, s, scale_factor=1, color=(1, 0.4, 0))

    # Put text to indicate table orientation
    mlab.text3d(0, TABLE_LENGTH // 2 + 10, -40, 'Player 1',
                scale=10, orient_to_camera=True)
    mlab.text3d(0, -(TABLE_LENGTH // 2 + 10), -40,
                'Player 2', scale=10, orient_to_camera=True)

    return point_plt.mlab_source


def main():
    args = get_args()

    # Read in intrinsic calibration
    load_config_1 = LoadConfig('config/intrinsic_calib_{}_camera_1.npz'.format(
        args['model'].lower()), 'calib_camera_1')
    intrinsic_1 = load_config_1.load()
    K_1 = intrinsic_1['camera_matrix']
    D_1 = intrinsic_1['dist_coeffs']
    load_config_2 = LoadConfig('config/intrinsic_calib_{}_camera_2.npz'.format(
        args['model'].lower()), 'calib_camera_2')
    intrinsic_2 = load_config_2.load()
    K_2 = intrinsic_2['camera_matrix']
    D_2 = intrinsic_2['dist_coeffs']

    # Read in extrinsic calibration
    load_config_e_1 = LoadConfig('config/extrinsic_calib_{}_camera_1.npz'.format(
        args['model'].lower()), 'extrinsic_camera_1')
    extrinsic_1 = load_config_e_1.load()
    R_1 = cv2.Rodrigues(extrinsic_1['rvec'])[0]
    T_1 = extrinsic_1['tvec']
    load_config_e_2 = LoadConfig('config/extrinsic_calib_{}_camera_2.npz'.format(
        args['model'].lower()), 'extrinsic_camera_2')
    extrinsic_2 = load_config_e_2.load()
    R_2 = cv2.Rodrigues(extrinsic_2['rvec'])[0]
    T_2 = extrinsic_2['tvec']

    # Setup visualization
    visu = setup_visu()

    # Setup atomic queues to send measurement information from
    # worker processes back to main process. Also limit queue
    # size to maintain sync between frames. Too big queue
    # can result in run away condition, while too less may
    # not give enough time process keyboard event.
    measurements_camera_1 = multiprocessing.Queue(maxsize=3)
    measurements_camera_2 = multiprocessing.Queue(maxsize=3)

    # Setup a shared variable to share keyboard event
    # across the main worker processes and the thread
    quit_event = multiprocessing.Value('b', False)

    # Setup a thread that calculates final ball position
    # and updates that into the visualization
    thread_update_visu = threading.Thread(target=update_visu, args=(
        measurements_camera_1, measurements_camera_2, visu, quit_event))
    thread_update_visu.start()

    # Start the two main worker processes
    # NOTE: When using Python 3, to maintain compatibility with
    # MACOSX, *spawn* may have to be used instead of *fork*
    # multiprocessing.set_start_method('spawn')
    process_camera_1 = multiprocessing.Process(target=main_worker, args=(
        1, args['video_1'], args['model'], K_1, D_1, R_1, T_1, measurements_camera_1, quit_event))
    process_camera_2 = multiprocessing.Process(target=main_worker, args=(
        2, args['video_2'], args['model'], K_2, D_2, R_2, T_2, measurements_camera_2, quit_event))
    process_camera_1.start()
    process_camera_2.start()

    # Show the visualization interactively
    # NOTE: This is a blocking event loop
    # And this **has** to be done in main thread
    mlab.show()

    # Wait for everyone to complete
    process_camera_1.join()
    process_camera_2.join()
    thread_update_visu.join()


if __name__ == '__main__':
    main()
