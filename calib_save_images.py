import cv2
from utils.display import Display
from utils.input import Video
import argparse
import os

play_or_pause = 'Pause'
seek_callback_action = False


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

    # Check for calib_images folder
    if not os.path.exists('calib_images'):
        print 'Please create a directory "calib_images"'
        return

    # Setup video display
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

        video_disp.refresh(frame)

        cur_frame_num = video.get_cur_frame_num()

        # Service the key events
        # if s is pressed, save image
        # if b is pressed, go back 1s
        # if n is pressed, go ahead 1s
        if video_disp.key_pressed('s'):
            video_file = os.path.basename(args['video']).lower()
            img_file_name = 'calib_images/{}_{}.png'.format(
                video_file.strip('.mp4'), cur_frame_num)
            if cv2.imwrite(img_file_name, frame):
                print 'Saved', img_file_name
        elif video_disp.key_pressed('n'):
            seek_callback(min((((cur_frame_num + 60) * 100) // num_frames), num_frames))
        elif video_disp.key_pressed('b'):
            seek_callback(max((((cur_frame_num - 60) * 100) // num_frames), 0))

        # Add quitting event
        if video_disp.can_quit():
            break


if __name__ == '__main__':
    main()
