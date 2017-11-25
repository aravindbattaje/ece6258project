import cv2
import numpy as np

def histogram_comparison(image1,image2,track_window):
    # this function finds the correlation in histogram between image1 and the
    # region of image2 specified by x,y,w,h ([x,y] top left corner, [x+w,y+h] bottom right)
    x,y,w,h = track_window
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_crop = image2[y:(y+h),x:(x+w)]
    image2_rgb = cv2.cvtColor(image2_crop, cv2.COLOR_BGR2RGB)
    hist1 = cv2.calcHist([image1_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1,hist1).flatten()
    hist2 = cv2.calcHist([image2_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2,hist2).flatten()

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation
