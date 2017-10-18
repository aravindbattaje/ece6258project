import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os
import cPickle as pickle

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


def setup_trackbars(window_name, trackbar_config):
    window = cv2.namedWindow(window_name, 0)

    cv2.createTrackbar('H_MIN', window_name,
                       trackbar_config['h_min'], 255, dummy_callback)
    cv2.createTrackbar('H_MAX', window_name,
                       trackbar_config['h_max'], 255, dummy_callback)
    cv2.createTrackbar('S_MIN', window_name,
                       trackbar_config['s_min'], 255, dummy_callback)
    cv2.createTrackbar('S_MAX', window_name,
                       trackbar_config['s_max'], 255, dummy_callback)
    cv2.createTrackbar('V_MIN', window_name,
                       trackbar_config['v_min'], 255, dummy_callback)
    cv2.createTrackbar('V_MAX', window_name,
                       trackbar_config['v_max'], 255, dummy_callback)

    cv2.createTrackbar('Erode_size', window_name, 0, 31, dummy_callback)
    cv2.createTrackbar('Dilate_size', window_name, 0, 31, dummy_callback)

    cv2.createTrackbar('Seek', window_name, 0, 100, seek_callback)
    cv2.createTrackbar('Playback', window_name, 0, 1, playpause_callback)


def get_params(window_name):
    h_min = cv2.getTrackbarPos('H_MIN', window_name)
    h_max = cv2.getTrackbarPos('H_MAX', window_name)
    s_min = cv2.getTrackbarPos('S_MIN', window_name)
    s_max = cv2.getTrackbarPos('S_MAX', window_name)
    v_min = cv2.getTrackbarPos('V_MIN', window_name)
    v_max = cv2.getTrackbarPos('V_MAX', window_name)

    erode_size = cv2.getTrackbarPos('Erode_size', window_name)
    dilate_size = cv2.getTrackbarPos('Dilate_size', window_name)

    return h_min, h_max, s_min, s_max, v_min, v_max, erode_size, dilate_size


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

    # Read in configuration
    with open('config/thresholds.pickle', 'r') as f:
        thresholds = pickle.load(f)

    # Setup video displays
    orig_video_disp = Display({'name': 'Original_Video'})
    processed_video_disp = Display({'name': 'Processed_Video'})

    # Setup controls
    setup_trackbars(controls_window_name, thresholds)

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

        # Refresh original video display
        orig_video_disp.refresh(frame)

        # Get threshold values
        h_min, h_max, s_min, s_max, v_min, v_max, erode_size, dilate_size = get_params(
            controls_window_name)

        # Convert image to HSV and apply threshold
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_thresh = cv2.inRange(
            frame_hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

        # Apply erosion
        # Create a kernel first and then apply kernel
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size + 1, erode_size + 1))
        frame_erode = cv2.erode(frame_thresh, erode_kernel)

        # Apply dilate
        # Create a kernel first and then apply kernel
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size + 1, dilate_size + 1))
        frame_dilate = cv2.dilate(frame_erode, dilate_kernel)

        # Refresh thresholded video display
        processed_video_disp.refresh(frame_dilate)

        # Add quitting event
        if orig_video_disp.can_quit() or processed_video_disp.can_quit():
            break

    # On quit, save the params
    with open('new_erode_dilate.pickle', 'w') as f:
        params = {
            'erode_size': erode_size,
            'dilate_size': dilate_size,
        }
        print 'Saved {} to {}'.format(params, f.name)
        pickle.dump(params, f)


if __name__ == '__main__':
    main()
