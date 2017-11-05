import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os
import numpy as np

cur_seek_pos = 0
seek_callback_action = False
controls_window_name = 'Controls'
play_or_pause = 'Pause'


def dummy_callback(value):
    pass


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


def setup_trackbars(window_name):
    window = cv2.namedWindow(window_name, 0)

    cv2.createTrackbar('H_MIN', window_name, 0, 255, dummy_callback)
    cv2.createTrackbar('H_MAX', window_name, 255, 255, dummy_callback)
    cv2.createTrackbar('S_MIN', window_name, 0, 255, dummy_callback)
    cv2.createTrackbar('S_MAX', window_name, 255, 255, dummy_callback)
    cv2.createTrackbar('V_MIN', window_name, 0, 255, dummy_callback)
    cv2.createTrackbar('V_MAX', window_name, 255, 255, dummy_callback)

    cv2.createTrackbar('Seek', window_name, 0, 100, seek_callback)
    cv2.createTrackbar('Playback', window_name, 0, 1, playpause_callback)


def get_thresholds(window_name):
    h_min = cv2.getTrackbarPos('H_MIN', window_name)
    h_max = cv2.getTrackbarPos('H_MAX', window_name)
    s_min = cv2.getTrackbarPos('S_MIN', window_name)
    s_max = cv2.getTrackbarPos('S_MAX', window_name)
    v_min = cv2.getTrackbarPos('V_MIN', window_name)
    v_max = cv2.getTrackbarPos('V_MAX', window_name)

    return h_min, h_max, s_min, s_max, v_min, v_max


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

    # Setup video displays
    orig_video_disp = Display({'name': 'Original_Video'})
    thresh_video_disp = Display({'name': 'Thresholded_Video'})
    mean_shift_video_disp = Display({'name': 'Mean-Shift Tracking Video'})

    # Setup controls
    setup_trackbars(controls_window_name)

    # Get input video
    video = Video(args['video'])
    num_frames = video.get_num_frames()

    # Get the first frame to start with
    frame = video.next_frame()

    global seek_callback_action

    # setup initial location of window
    top, length, left, width = 450, 36, 1000, 43  # simply hardcoded the values
    track_window = (left, top, width, length)

    # set up the ROI for tracking
    roi = frame[top:top + length, left:left + width]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_min, h_max, s_min, s_max, v_min, v_max = get_thresholds(
        controls_window_name)
    mask = cv2.inRange(hsv_roi, np.array(
        (h_min, s_min, v_min)), np.array((h_max, s_max, v_max)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)

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

        # Refresh original video display
        orig_video_disp.refresh(frame)

        # Get threshold values
        h_min, h_max, s_min, s_max, v_min, v_max = get_thresholds(
            controls_window_name)

        # Convert image to HSV and apply threshold
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        frame_thresh = cv2.inRange(
            frame_hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

        # threshold image in hsv domain
        frame_hsv_threshold = cv2.bitwise_and(
            frame_hsv, frame_hsv, mask=frame_thresh)

        # Refresh thresholded video display
        thresh_video_disp.refresh(frame_hsv_threshold)

        # Find the backprojection of the histogram
        dst = cv2.calcBackProject([frame_hsv_threshold], [
                                  0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_criteria)

        # Draw it on image
        x, y, w, h = track_window
        frame_mean_shift = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        # Refresh mean shift tracking video display
        mean_shift_video_disp.refresh(frame_mean_shift)


if __name__ == '__main__':
    main()
