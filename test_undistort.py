import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os
from utils.config import LoadConfig
from utils.config import SaveConfig
from calib_generate_charuco import dictionary, board
from collections import deque

play_or_pause = 'Pause'
seek_callback_action = False
MAX_ARUCO_IDS = 10
MAX_CHARUCO_IDS = 12
MAX_NUM_IMAGES_FOR_CALIB = 30


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

    args = vars(arg_parser.parse_args())

    if not os.path.exists(args['video']):
        arg_parser.error('{} file not found'.format(args['video']))

    return args


def main():
    args = get_args()

    # Read in configuration
    load_config = LoadConfig('new_calib.npz', 'calib')
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

    # Deck for storing calib images
    calib_img_deck = deque(maxlen=MAX_NUM_IMAGES_FOR_CALIB)

    # Deck for storing charuco info
    charuco_corners_deck = deque(maxlen=MAX_NUM_IMAGES_FOR_CALIB)
    charuco_ids_deck = deque(maxlen=MAX_NUM_IMAGES_FOR_CALIB)

    skip_count = 0

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

        img_undistorted = cv2.undistort(frame, calib['camera_matrix'], calib['dist_coeffs'])

        video_disp.refresh(img_undistorted)


        # Add quitting event
        if video_disp.can_quit():
            break

    # On quit, save the params
    # save_config = SaveConfig('new_erode_dilate', 'erode_dilate')
    # save_config.save(dilate_size=dilate_size, erode_size=erode_size)
    #img_size = calib_img_deck[0].shape[:2]
    # print charuco_ids_deck


if __name__ == '__main__':
    main()
