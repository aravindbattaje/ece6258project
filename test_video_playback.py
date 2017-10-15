import cv2
from utils.input import Video
from utils.display import Display

video = Video('captures/GOPR8960.MP4')
display = Display()

# # Scheme - 1
# #     Vanilla playback

# # Emulate a do-while loop
# # with "while True" and breaking
# # if condition fails after executing
# while True:
#     frame = video.next_frame()
#     if video.end_reached():
#         break

#     # Do some operation
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Refresh display with new image
#     display.refresh(gray)
#     if display.can_quit():
#         break

# Scheme - 2
#   Optimized for sequential playback
#   But with random seeks, playback will be slow

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
