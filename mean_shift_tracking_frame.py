import cv2
import numpy as np

def mean_shift_tracking_frame(track_window,frame,prev_frame):
    # track_window is a length-4 list [x,y,w,h] where (x,y) corresponds to top left corner and (x+w,y+h) bottom right
    # find the track_window that has the best histogram match with the histogram of the track_window from prev_frame 

    x,y,w,h = track_window
    
    # set up the ROI for tracking
    roi = prev_frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((12, 170,200)), np.array((30,255,255))) # orange ball
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1 )
    
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_frame = cv2.inRange(frame_hsv, np.array((12, 170,200)), np.array((30,255,255)))
    frame_hsv_threshold = cv2.bitwise_and(frame_hsv,frame_hsv, mask= mask_frame)

    
    dst = cv2.calcBackProject([frame_hsv_threshold],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, (x,y,w,h), term_crit)

    return track_window

