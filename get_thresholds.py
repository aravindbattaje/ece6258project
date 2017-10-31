import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os
from utils.config import SaveConfig

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
    thresh_video_disp = Display({'name': 'Tresholded_Video'})

    # Setup controls
    setup_trackbars(controls_window_name)

    # Get input video
    video = Video(args['video'])
    num_frames = video.get_num_frames()

    # Get the first frame to start with
    frame = video.next_frame()

    # To communicate with seek callback
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
        h_min, h_max, s_min, s_max, v_min, v_max = get_thresholds(
            controls_window_name)

        # Convert image to HSV and apply threshold
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_thresh = cv2.inRange(
            frame_hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

        # Refresh thresholded video display
        thresh_video_disp.refresh(frame_thresh)

        # Add quitting event
        if orig_video_disp.can_quit() or thresh_video_disp.can_quit():
            break

    # On quit, save the thresholds
    save_config = SaveConfig('new_thresholds', 'thresholds')
    save_config.save(h_min=h_min, h_max=h_max, s_min=s_min,
                     s_max=s_max, v_min=v_min, v_max=v_max)


if __name__ == '__main__':
    main()
