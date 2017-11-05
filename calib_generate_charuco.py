import cv2
import numpy as np

# Use a predefined dictionary
# and for a bunch of fiducial markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Draw a Chessboard pattern + ArUco pattern board
board = cv2.aruco.CharucoBoard_create(4, 5, 0.04, 0.02, dictionary)
board_img = board.draw((600, 900), 10, 1)

# Save the image
cv2.imwrite('calib_charuco_board.png', board_img)