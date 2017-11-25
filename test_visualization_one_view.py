from __future__ import division
import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os
from utils.config import LoadConfig
from utils.config import SaveConfig
import numpy as np

import mean_shift_tracking_frame
import histogram_comparison

PING_PONG_DIAMETER = 0.04 # m
PIXELS_PER_MM = 49
FOCAL_LENGTH = 14 # mm

def get_args():
    # Setup argument parser
    # and parse the inputs
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-v', '--video', required=True, help='Input video file')

    args = vars(arg_parser.parse_args())

    if not os.path.exists(args['video']):
        arg_parser.error('{} file not found'.format(args['video']))

    return args


def main():
    args = get_args()

    # Read in intrinsic calibration
    load_config = LoadConfig('config/intrinsic_calib_f.npz', 'calib')
    calib = load_config.load()

    # Read in extrinsic calibration
    load_config_e = LoadConfig('new_extrinsics.npz', 'extrinsics')
    extrinsics = load_config_e.load()

    # Setup video display
    video_disp = Display({'name': 'Video'})

    # Get input video
    video = Video(args['video'])
    num_frames = video.get_num_frames()

    # Get the first frame; to see
    # if video framework works
    frame = video.next_frame()

    # Setup the undistortion stuff

    # Harcoded image size as
    # this is a test script.
    # As already ranted before
    # someone messed with the image
    # size indexing and reversed it.
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
    corr_threshold = 0.20
    radius_change_threshold = 5
    ball_image_file = 'ball_image.jpg'
    ball_image = cv2.imread(ball_image_file) # will be used for histogram comparison

    fgbg2 = cv2.createBackgroundSubtractorMOG2()

    kernel = np.ones((3,3),np.uint8)
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
        mask2 = cv2.inRange(img_hsv, np.array((10, 150,200)), np.array((30,255,255)))
        fgmask2 = fgbg2.apply(img_undistorted)
        mask2_color_bgs = cv2.bitwise_and(mask2, mask2, mask = fgmask2)
        frame2_hsv_bgs = cv2.bitwise_and(img_hsv, img_hsv, mask = mask2_color_bgs)


        frame_hsv = img_hsv
        frame_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(frame_hsv, np.array((10, 150,150)), np.array((40,255,255)))
        open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        frame_thresholded_opened_gray = cv2.bitwise_and(frame_gray, frame_gray, mask = open_mask)
        frame_thresholded_opened_gray_smoothed = cv2.GaussianBlur(frame_thresholded_opened_gray,(11,11),0)
        # one more opening
        a = cv2.inRange(frame_thresholded_opened_gray_smoothed,10,256)
        b = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
        circles  = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, dp=1.5, minDist = 2500,param1 = 150, param2=15,minRadius = 5, maxRadius = 35)
        if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            x,y,r = circles[0]
            ball_position_frame2_new = [x-r,y-r,2*r,2*r]
            # loop over the (x, y) coordinates and radius of the circles
        else:
            ball_position_frame2_new = None

        frame2 = img_undistorted
        if ball_position_frame2_new == None and ball_position_frame2!=None:
            ball_position_frame2_new = mean_shift_tracking_frame.mean_shift_tracking_frame(ball_position_frame2,frame2,prev_frame2)

        if ball_position_frame2_new!=None:
            # check if matches correctly by comparing histogram
            # if not match, not consider the current point as a ball
            hist_corr2 = histogram_comparison.histogram_comparison(ball_image,frame2,ball_position_frame2_new)
            if hist_corr2<corr_threshold:
                ball_position_frame2_new = None
            
        # check if radius changes by more than the threshold between frames
        if ball_position_frame2_new!=None and ball_position_frame2!=None:
            x2_new,y2_new,w2_new,h2_new = ball_position_frame2_new
            x2,y2,w2,h2 = ball_position_frame2   
            if (w2_new+h2_new)/2.0-(w2+h2)/2.0>radius_change_threshold:
                ball_position_frame2 = [x2_new,y2_new,w2_new+radius_change_threshold,h2_new+radius_change_threshold]
            elif (w2_new+h2_new)/2.0-(w2+h2)/2.0<-radius_change_threshold:
                ball_position_frame2 = [x2_new,y2_new,w2_new-radius_change_threshold,h2_new-radius_change_threshold]
            else:
                ball_position_frame2 = ball_position_frame2_new
        elif ball_position_frame2_new!=None:
            ball_position_frame2 = ball_position_frame2_new
        elif ball_position_frame2_new==None:
            ball_position_frame2 = None

        prev_frame2 = frame2

        print ball_position_frame2
        if ball_position_frame2:
            x2,y2,w2,h2 = ball_position_frame2 
            x = (x2 - 960) / PIXELS_PER_MM
            y = (y2 - 540) / PIXELS_PER_MM
            z = PING_PONG_DIAMETER * FOCAL_LENGTH / (w2 / PIXELS_PER_MM)

            ball_cc = np.array([x, y, z]) / 1000
            ball_wc = np.dot(R.T, extrinsics['tvec'].ravel() - ball_cc)
            
            print 'Ball', ball_wc

        # Update GUI with new image
        video_disp.refresh(img_undistorted)

        # Add quitting event
        if video_disp.can_quit():
            break


if __name__ == '__main__':
    main()
