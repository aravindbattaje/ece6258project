import cv2
import numpy as np

# Use a predefined dictionary
# and for a bunch of fiducial markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Generate a Chessboard pattern + ArUco pattern board
# Input params:
# Num. squares in X direction = 8
# Num. squares in Y direction = 10
# Chess board square length = 2 cm
# Aruco marker square length = 1 cm
board = cv2.aruco.CharucoBoard_create(8, 10, 0.02, 0.01, dictionary)

def main():
    # Draw the board on a canvas of
    # 600 x 900 px with some borders
    board_img = board.draw((600, 900), 10, 1)

    # Save the image
    cv2.imwrite('calib_charuco_board.png', board_img)

if __name__ == '__main__':
    main()
    