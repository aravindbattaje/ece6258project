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

from multiprocessing import Process, Array

import mean_shift_tracking_frame
import histogram_comparison

PING_PONG_DIAMETER = 40  # mm
FOCAL_LENGTH = 14  # mm

main_process_end_reached = False

def visualize_table(ball_wc):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p = Rectangle((-0.7625, -1.37), 1.525, 2.74, alpha=0.5)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 3)
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

    while not main_process_end_reached or plt.get_fignums():
        if not compare_coordinates(ball_wc, prev_ball_wc):
            prev_ball_wc[0] = ball_wc[0]
            prev_ball_wc[1] = ball_wc[1]
            prev_ball_wc[2] = ball_wc[2]
            sca.set_offsets([[ball_wc[0]], [ball_wc[1]]])
            sca.set_3d_properties([[ball_wc[2]]], zdir='z')
            sca.set_alpha(1)
            fig.canvas.draw_idle()
            plt.pause(0.1)

        sca.set_alpha(0.25)
        fig.canvas.draw_idle()
        plt.pause(0.05)

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

    # Read in intrinsic calibration
    load_config = LoadConfig('config/intrinsic_calib_{}.npz'.format(
        args['model'].lower()), 'calib')
    calib = load_config.load()

    # Read in extrinsic calibration
    load_config_e = LoadConfig('config/extrinsic_calib_p_camera_1.npz', 'extrinsics')
    extrinsics = load_config_e.load()

    # Setup video display
    video_disp = Display({'name': 'Video'})

    # Get input video
    video = Video(args['video'])
    num_frames = video.get_num_frames()

    # Get the first frame; to see
    # if video framework works
    frame = video.next_frame()

    shared_var = Array('d', [0, 0, 0])
    visu = Process(target=visualize_table, args=(shared_var,))
    visu.start()

    # Setup the undistortion stuff

    if args['model'].upper() == 'P':
        # Harcoded image size as
        # this is a test script
        img_size = (1920, 1080)

        # First create scaled intrinsics because we will undistort
        # into region beyond original image region
        new_calib_matrix, _ = cv2.getOptimalNewCameraMatrix(
            calib['camera_matrix'], calib['dist_coeffs'], img_size, 0.35)

        # Then calculate new image size according to the scaling
        # Unfortunately the Python API doesn't directly provide the
        # the new image size. They forgot?
        new_img_size = (int(img_size[0] + (new_calib_matrix[0, 2] - calib['camera_matrix'][0, 2])), int(
            img_size[1] + (new_calib_matrix[1, 2] - calib['camera_matrix'][1, 2])))

        # Standard routine of creating a new rectification
        # map for the given intrinsics and mapping each
        # pixel onto the new map with linear interpolation
        map1, map2 = cv2.initUndistortRectifyMap(
            calib['camera_matrix'], calib['dist_coeffs'], None, new_calib_matrix, new_img_size, cv2.CV_16SC2)

    elif args['model'].upper() == 'F':
        # Harcoded image size
        img_size = (1920, 1080)
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

    # STUFF
    corr_threshold = -1
    radius_change_threshold = 5
    ball_image_file = 'ball_image.jpg'
    # will be used for histogram comparison
    ball_image = cv2.imread(ball_image_file)

    fgbg2 = cv2.createBackgroundSubtractorMOG2()

    kernel = np.ones((6, 6), np.uint8)
    ball_position_frame2 = None
    prev_frame2 = None

    # Get the rotation matrix
    R = cv2.Rodrigues(extrinsics['rvec'])[0]

    while not video.end_reached():

        frame = video.next_frame()
        img_undistorted = cv2.remap(
            frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # STUFF
        img_hsv = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(img_hsv, np.array(
            (15, 190, 200)), np.array((25, 255, 255)))
        fgmask2 = fgbg2.apply(img_undistorted)
        mask2_color_bgs = cv2.bitwise_and(mask2, mask2, mask=fgmask2)
        frame2_hsv_bgs = cv2.bitwise_and(
            img_hsv, img_hsv, mask=mask2_color_bgs)

        frame_hsv = img_hsv
        frame_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(frame_hsv, np.array(
            (10, 150, 150)), np.array((40, 255, 255)))
        open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        frame_thresholded_opened_gray = cv2.bitwise_and(
            frame_gray, frame_gray, mask=open_mask)
        frame_thresholded_opened_gray_smoothed = cv2.GaussianBlur(
            frame_thresholded_opened_gray, (11, 11), 0)
        # opening
        a = cv2.inRange(frame_thresholded_opened_gray_smoothed, 10, 256)
        b = cv2.morphologyEx(mask2_color_bgs, cv2.MORPH_OPEN, kernel)
        circles = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, dp=3,
                                   minDist=2500, param1=300, param2=5, minRadius=3, maxRadius=30)
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            ball_position_frame2 = [x - r, y - r, 2 * r, 2 * r]
            # loop over the (x, y) coordinates and radius of the circles
        else:
            ball_position_frame2 = None

        frame2 = img_undistorted

        mask2_ball_radius = cv2.bitwise_and(fgmask2, fgmask2, mask=cv2.inRange(img_hsv, np.array((10, 150, 180)), np.array((40, 255, 255))))
        if ball_position_frame2!=None:
            x2,y2,w2,h2 = ball_position_frame2
            ball_crop_temp = mask2_ball_radius[(y2+h2/2-30):(y2+h2/2+30),(x2+w2/2-30):(x2+w2/2+30)]
            ball_crop_color = frame2[(y2+h2/2-30):(y2+h2/2+30),(x2+w2/2-30):(x2+w2/2+30)]
            height, width = ball_crop_temp.shape
        else:
            ball_crop_temp = []
            height = 0
            width = 0
            

        if height != 0 and width != 0:
            ball_crop = ball_crop_temp



        cnts = cv2.findContours(ball_crop.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	if len(cnts) > 0 and ball_position_frame2:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            width,height = rect[1]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(ball_crop_color,[box],0,(0,0,255),2)
            ball_position_frame2 = [ball_position_frame2[0],ball_position_frame2[1],min(width,height),min(width,height)]

        prev_frame2 = frame2

        # print ball_position_frame2
        if ball_position_frame2:
            x2, y2, w2, h2 = ball_position_frame2
            # x = (x2 - 960) / PIXELS_PER_MM
            # y = (y2 - 540) / PIXELS_PER_MM
            pixels_per_mm = (
                new_calib_matrix[0, 0] + new_calib_matrix[1, 1]) / 2 / FOCAL_LENGTH
            z = PING_PONG_DIAMETER * FOCAL_LENGTH / (w2 / pixels_per_mm)
            x = ((x2 - new_calib_matrix[0, 2]) /
                 pixels_per_mm) * z / FOCAL_LENGTH
            y = ((y2 - new_calib_matrix[1, 2]) /
                 pixels_per_mm) * z / FOCAL_LENGTH

            ball_cc = np.array([x, y, z]) / 1000
            #ball_wc = np.dot(R.T, extrinsics['tvec'].ravel() - ball_cc)
            ball_wc = np.dot(R.T, ball_cc - extrinsics['tvec'].ravel())

            print 'Ball IC', np.array([x2, y2]), 'Dia', w2
            print 'Ball CC', ball_cc
            print 'Ball WC', ball_wc

            shared_var[0] = ball_wc[0]
            shared_var[1] = ball_wc[1]
            shared_var[2] = ball_wc[2]

        # Update GUI with new image




        
        video_disp.refresh(frame2)

        #print "Pixels", ball_position_frame2

        # Add quitting event
        if video_disp.can_quit():
            break

    global main_process_end_reached
    main_process_end_reached = True
    visu.join()


if __name__ == '__main__':
    main()
