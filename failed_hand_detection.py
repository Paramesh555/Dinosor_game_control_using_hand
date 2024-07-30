import numpy as np
import cv2 as cv2
import math
import pyautogui

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorKNN()
if not cap.isOpened():
    print("cannot open camera")
    exit()

while(True):
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv2.rectangle(frame,(84,84),(510,500),(0,255,0),3)
    crop_image = frame[0:600, 0:600]
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    fgMask = backSub.apply(blur)
    im_floodfill = fgMask.copy()
    h, w = fgMask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
    im_out = fgMask | im_floodfill_inv   
    # Show the frames
    cv2.imshow('Frame', blur)
    cv2.imshow('Foreground',fgMask)
    cv2.imshow('another',im_out)

    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
