from __future__ import division
import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os
from utils.config import LoadConfig
from utils.config import SaveConfig
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

from multiprocessing import Array, Value
from threading import Thread

import mean_shift_tracking_frame
import histogram_comparison

PING_PONG_DIAMETER = 40  # mm
FOCAL_LENGTH = 14  # mm


def visualize_table(end_reached, ball_wc):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p = Rectangle((-0.7625, -1.37), 1.525, 2.74, alpha=0.5)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    sca = ax.scatter([0], [1.37], [0.5], color='r', s=50, alpha=0.25)

    plt.draw()

    prev_ball_wc = [0, 0, 0]

    def compare_coordinates(one, two):
        if one[0] == two[0] and one[1] == two[1] and one[2] == two[2]:
            return True
        return False

    while end_reached.value != 1 and plt.get_fignums() > 0:
        if not compare_coordinates(ball_wc, prev_ball_wc):
            prev_ball_wc[0] = ball_wc[0]
            prev_ball_wc[1] = ball_wc[1]
            prev_ball_wc[2] = ball_wc[2]
            sca.set_offsets([[ball_wc[0]], [ball_wc[1]]])
            sca.set_3d_properties([[ball_wc[2]]], zdir='z')
            sca.set_alpha(1)
            fig.canvas.draw_idle()
            plt.pause(0.033)

        sca.set_alpha(0.25)
        fig.canvas.draw_idle()
        plt.pause(0.0015)
    plt.close('all')


def get_args():
    # Setup argument parser
    # and parse the inputs
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-v1', '--video-1', required=True, help='Video file from Camera 1')

    arg_parser.add_argument(
        '-v2', '--video-2', required=True, help='Video file from Camera 2')

    arg_parser.add_argument(
        '-m', '--model', required=True, help='Camera model - Pinhole (P)/Fisheye (F)'
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

    # Setup video displays
    video_disp_1 = Display({'name': 'Camera_1'})
    video_disp_2 = Display({'name': 'Camera_2'})

    # Get input video
    video_1 = Video(args['video_1'])
    video_2 = Video(args['video_2'])

    # Get the first frame; to see
    # if video framework works
    frame_1 = video_1.next_frame()
    frame_2 = video_2.next_frame()

    # Original code was to multiprocess, but
    # found MACOSX doesn't like forking processes
    # with GUIs. However, retaining Array from
    # multiproc for future.
    shared_var = Array('d', [0, 0, 0])
    end_reached = Value('b', False)
    # visu = Process(target=visualize_table, args=(shared_var,))
    # visu.start()
    visu = Thread(target=visualize_table, args=(end_reached, shared_var))
    visu.daemon = True
    visu.start()

    # Setup the undistortion stuff

    if args['model'].upper() == 'P':
        # Harcoded image size as
        # this is a test script
        img_size = (1920, 1080)

        # First create scaled intrinsics because we will undistort
        # into region beyond original image region
        new_K_1 = cv2.getOptimalNewCameraMatrix(K_1, D_1, img_size, 0.35)[0]
        new_K_2 = cv2.getOptimalNewCameraMatrix(K_2, D_2, img_size, 0.35)[0]

        # Then calculate new image size according to the scaling
        # Unfortunately the Python API doesn't directly provide the
        # the new image size. They forgot?
        new_img_size_1 = (int(img_size[0] + (new_K_1[0, 2] - K_1[0, 2])), int(
            img_size[1] + (new_K_1[1, 2] - K_1[1, 2])))
        new_img_size_2 = (int(img_size[0] + (new_K_2[0, 2] - K_2[0, 2])), int(
            img_size[1] + (new_K_2[1, 2] - K_2[1, 2])))

        # Standard routine of creating a new rectification
        # map for the given intrinsics and mapping each
        # pixel onto the new map with linear interpolation
        map1_1, map2_1 = cv2.initUndistortRectifyMap(
            K_1, D_1, None, new_K_1, new_img_size_1, cv2.CV_16SC2)
        map1_2, map2_2 = cv2.initUndistortRectifyMap(
            K_2, D_2, None, new_K_2, new_img_size_2, cv2.CV_16SC2)

    elif args['model'].upper() == 'F':
        # Harcoded image size
        img_size = (1920, 1080)
        # First create scaled intrinsics because we will undistort
        # into region beyond original image region. The alpha
        # parameter in pinhole model is equivalent to balance parameter here.
        new_K_1 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_1, D_1, img_size, np.eye(3), balance=1)
        new_K_2 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_2, D_2, img_size, np.eye(3), balance=1)

        # Then calculate new image size according to the scaling
        # Well if they forgot this in pinhole Python API,
        # can't complain about Fisheye model. Note the reversed
        # indexing here too.
        new_img_size_1 = (int(img_size[0] + (new_K_1[0, 2] - K_1[0, 2])), int(
            img_size[1] + (new_K_1[1, 2] - K_1[1, 2])))
        new_img_size_2 = (int(img_size[0] + (new_K_2[0, 2] - K_2[0, 2])), int(
            img_size[1] + (new_K_2[1, 2] - K_2[1, 2])))

        # Standard routine of creating a new rectification
        # map for the given intrinsics and mapping each
        # pixel onto the new map with linear interpolation
        map1_1, map2_1 = cv2.fisheye.initUndistortRectifyMap(
            K_1, D_1, np.eye(3), new_K_1, new_img_size_1, cv2.CV_16SC2)
        map1_2, map2_2 = cv2.fisheye.initUndistortRectifyMap(
            K_2, D_2, np.eye(3), new_K_2, new_img_size_2, cv2.CV_16SC2)

    # STUFF
    corr_threshold = -1
    radius_change_threshold = 5
    ball_image_file = 'ball_image.jpg'
    # will be used for histogram comparison
    ball_image = cv2.imread(ball_image_file)

    fgbg1 = cv2.createBackgroundSubtractorMOG2()
    fgbg2 = cv2.createBackgroundSubtractorMOG2()

    kernel = np.ones((6, 6), np.uint8)
    ball_position_frame1 = None
    ball_position_frame2 = None
    prev_frame1 = None
    prev_frame2 = None
    ball_wc = [0, 0, 0]

    while not video_1.end_reached() and not video_2.end_reached():

        frame_1 = video_1.next_frame()
        frame_2 = video_2.next_frame()

        img_undistorted_1 = cv2.remap(
            frame_1, map1_1, map2_1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        img_undistorted_2 = cv2.remap(
            frame_2, map1_2, map2_2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # STUFF1
        frame_1_hsv = cv2.cvtColor(img_undistorted_1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(frame_1_hsv, np.array(
            (15, 190, 200)), np.array((25, 255, 255)))
        fgmask1 = fgbg1.apply(img_undistorted_1)
        mask1_color_bgs = cv2.bitwise_and(mask1, mask1, mask=fgmask1)
        frame1_hsv_bgs = cv2.bitwise_and(
            frame_1_hsv, frame_1_hsv, mask=mask1_color_bgs)

        # opening
        b1 = cv2.morphologyEx(mask1_color_bgs, cv2.MORPH_OPEN, kernel)
        circles1 = cv2.HoughCircles(b1, cv2.HOUGH_GRADIENT, dp=3,
                                    minDist=2500, param1=300, param2=5, minRadius=3, maxRadius=30)
        if circles1 is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles1 = np.round(circles1[0, :]).astype("int")
            x, y, r = circles1[0]
            ball_position_frame1 = [x - r, y - r, 2 * r, 2 * r]
            # loop over the (x, y) coordinates and radius of the circles
        else:
            ball_position_frame1 = None

        mask_ball_radius1 = cv2.bitwise_and(fgmask1, fgmask1, mask=cv2.inRange(
            frame_1_hsv, np.array((10, 150, 180)), np.array((40, 255, 255))))

        # determine the correct radius
        if ball_position_frame1 != None:
            x1, y1, w1, h1 = ball_position_frame1
            ball_crop_temp1 = mask_ball_radius1[(
                y1 + h1 // 2 - 30):(y1 + h1 // 2 + 30), (x1 + w1 // 2 - 30):(x1 + w1 // 2 + 30)]
            height, width = ball_crop_temp1.shape
            if height != 0 and width != 0:
                # successfully cropped image
                ball_crop1 = ball_crop_temp1
                cnts = cv2.findContours(
                    ball_crop1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                center = None
                if len(cnts) > 0:
                    # contour detected
                    c = max(cnts, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)
                    width, height = rect[1]
                    ball_position_frame1 = [ball_position_frame1[0], ball_position_frame1[1], min(
                        width, height), min(width, height)]

        prev_frame1 = img_undistorted_1

        if ball_position_frame1:
            x1, y1, w1, h1 = ball_position_frame1
            pixels_per_mm = (
                K_1[0, 0] + K_1[1, 1]) / 2 / FOCAL_LENGTH
            z = PING_PONG_DIAMETER * FOCAL_LENGTH / (w1 / pixels_per_mm)
            x = ((x1 - K_1[0, 2]) /
                 pixels_per_mm) * z / FOCAL_LENGTH
            y = ((y1 - K_1[1, 2]) /
                 pixels_per_mm) * z / FOCAL_LENGTH

            ball_cc1 = np.array([x, y, z]) / 1000
            ball_wc1 = np.dot(R_1.T, ball_cc1 - T_1.ravel())

        # STUFF2
        frame_2_hsv = cv2.cvtColor(img_undistorted_2, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(frame_2_hsv, np.array(
            (15, 190, 200)), np.array((25, 255, 255)))
        fgmask2 = fgbg2.apply(img_undistorted_2)
        mask2_color_bgs = cv2.bitwise_and(mask2, mask2, mask=fgmask2)
        frame2_hsv_bgs = cv2.bitwise_and(
            frame_2_hsv, frame_2_hsv, mask=mask2_color_bgs)

        # opening
        b2 = cv2.morphologyEx(mask2_color_bgs, cv2.MORPH_OPEN, kernel)
        circles2 = cv2.HoughCircles(b2, cv2.HOUGH_GRADIENT, dp=3,
                                    minDist=2500, param1=300, param2=5, minRadius=3, maxRadius=30)
        if circles2 is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles2 = np.round(circles2[0, :]).astype("int")
            x, y, r = circles2[0]
            ball_position_frame2 = [x - r, y - r, 2 * r, 2 * r]
            # loop over the (x, y) coordinates and radius of the circles
        else:
            ball_position_frame2 = None

        mask_ball_radius2 = cv2.bitwise_and(fgmask2, fgmask2, mask=cv2.inRange(
            frame_2_hsv, np.array((10, 150, 180)), np.array((40, 255, 255))))

        # determine the correct radius
        if ball_position_frame2 != None:
            x2, y2, w2, h2 = ball_position_frame2
            ball_crop_temp2 = mask_ball_radius2[(
                y2 + h2 // 2 - 30):(y2 + h2 // 2 + 30), (x2 + w2 // 2 - 30):(x2 + w2 // 2 + 30)]
            height, width = ball_crop_temp2.shape
            if height != 0 and width != 0:
                # successfully cropped image
                ball_crop2 = ball_crop_temp2
                cnts = cv2.findContours(
                    ball_crop2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                center = None
                if len(cnts) > 0:
                    # contour detected
                    c = max(cnts, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)
                    width, height = rect[1]
                    ball_position_frame2 = [ball_position_frame2[0], ball_position_frame2[1], min(
                        width, height), min(width, height)]

        prev_frame2 = img_undistorted_2

        if ball_position_frame2:
            x2, y2, w2, h2 = ball_position_frame2
            pixels_per_mm = (
                K_2[0, 0] + K_2[1, 1]) / 2 / FOCAL_LENGTH
            z = PING_PONG_DIAMETER * FOCAL_LENGTH / (w2 / pixels_per_mm)
            x = ((x2 - K_2[0, 2]) /
                 pixels_per_mm) * z / FOCAL_LENGTH
            y = ((y2 - K_2[1, 2]) /
                 pixels_per_mm) * z / FOCAL_LENGTH

            ball_cc2 = np.array([x, y, z]) / 1000
            ball_wc2 = np.dot(R_2.T, ball_cc2 - T_2.ravel())

            # Additional rotation for absolute coordinates
            R = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
            ball_wc2 = np.dot(R, ball_wc2)

        # Combine STUFF1 and STUFF2
        # If positions from either available,
        # update as predicted from either.
        # But if both available, predict the average
        if ball_position_frame1 and ball_position_frame2:
            ball_wc = (ball_wc1 + ball_wc2) / 2        
        elif ball_position_frame1:
            ball_wc = ball_wc1
        elif ball_position_frame2:
            ball_wc = ball_wc2

        # print 'Ball IC', np.array([x2, y2]), 'Dia', w2
        # print 'Ball CC', ball_cc2
        # print 'Ball WC', ball_wc2

        shared_var[0] = ball_wc[0]
        shared_var[1] = ball_wc[1]
        shared_var[2] = ball_wc[2]

        # Update GUI with new image
        video_disp_1.refresh(img_undistorted_1)
        video_disp_2.refresh(img_undistorted_2)

        # Add quitting event
        if video_disp_2.can_quit() or video_disp_1.can_quit():
            break

    end_reached.value = True

    visu.join()


if __name__ == '__main__':
    main()
