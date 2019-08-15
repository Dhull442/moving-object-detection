import numpy as np
import cv2
cap = cv2.VideoCapture('dataset/1.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
ret  = 1;
k2 = np.ones((4,4),np.uint8)
kernel = np.ones((8,8),np.uint8)
while(ret & cap.isOpened()):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    erosion = cv2.erode(fgmask,k2,iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    #erosion = cv2.erode(opening,k2,iterations = 1)
    cv2.imshow('frame',fgmask)
    cv2.imshow('opened', opening)
    cv2.imshow('eroded',erosion)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
