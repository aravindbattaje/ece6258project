import cv2
from utils.input import Video
from utils.display import Display
import argparse
import os

# Scheme - 1
#     Vanilla playback


def scheme_1(file_name):
    video = Video(file_name)
    display = Display()

    # Emulate a do-while loop
    # with "while True" and breaking
    # if condition fails after executing
    while True:
        frame = video.next_frame()
        if video.end_reached():
            break

        # Do some operation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Refresh display with new image
        display.refresh(gray)
        if display.can_quit():
            break


# Scheme - 2
#   Optimized for sequential playback
#   But with random seeks, playback will be slow
def scheme_2(file_name):
    video = Video(file_name)
    display = Display()

    # First get the number of frames
    num_frames = video.get_num_frames()

    # Get each frame from video and display
    # If step is greater than one (simulating random
    # seeks,) the playback will be slow
    for i in range(num_frames):
        # Get the frame desired
        frame = video.get_frame(i)

        # Do some operation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Refresh display with new image
        display.refresh(gray)

        # Quit midway if required
        if display.can_quit():
            break


def get_args():
    # Setup argument parser
    # and parse the inputs
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-v', '--video', required=True, help='Input video file')
    arg_parser.add_argument('-s', '--scheme', required=True, type=int,
                            help='Scheme to run (1 or 2)')
    args = vars(arg_parser.parse_args())

    if not os.path.exists(args['video']):
        arg_parser.error('{} file not found'.format(args['video']))

    if args['scheme'] != 1 and args['scheme'] != 2:
        arg_parser.error('{} not valid scheme'.format(args['scheme']))

    return args


def main():
    args = get_args()

    if args['scheme'] == 1:
        scheme_1(args['video'])
    else:
        scheme_2(args['video'])


if __name__ == '__main__':
    main()
