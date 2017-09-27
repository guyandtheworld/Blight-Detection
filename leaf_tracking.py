from scipy import ndimage
import numpy as np
import imutils
import cv2
 

def detect_contour(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts

def detect_leaf(frame, x, y, w, h, diseased):
    color = "green"
    hsv_color = (0, 255, 0)
    if diseased:
        color = "red"
        hsv_color = (0, 0, 255)
    w = w + w/5
    h = h + h/5
    cv2.rectangle(frame, (x, y), (x+w, y+h), hsv_color, 2)

def detect_disease(cnts_brown):
    if len(cnts_brown) > 0:
        c = max(cnts_brown, key=cv2.contourArea)
        return True
    return False

def draw_contours(cnts_green, cnts_brown, frame):
    c = max(cnts_green , key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    (x, y, w, h) = cv2.boundingRect(c)    
    diseased = True
    if radius > 10:
        diseased = detect_disease(cnts_brown)
        detect_leaf(frame, x, y, w, h, diseased)

def main():

    greenLower = (29, 86, 6)
    
    greenUpper = (64, 255, 255)

    brownLower = (10, 100, 20)

    brownUpper = (20, 255, 200)

    camera = cv2.VideoCapture(1)
    
    while True:
        (_, frame) = camera.read()

        frame = imutils.resize(frame, width=600)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv, greenLower, greenUpper)
        mask_brown = cv2.inRange(hsv, brownLower, brownUpper)

        mask_green = cv2.erode(mask_green, None, iterations=2)
        mask_brown = cv2.erode(mask_brown, None, iterations=2)
        mask_green = cv2.dilate(mask_green, None, iterations=2)
        mask_brown = cv2.dilate(mask_brown, None, iterations=2)

        cnts_green = detect_contour(mask_green.copy())
        cnts_brown = detect_contour(mask_brown.copy())

        if len(cnts_green) > 0:
            draw_contours(cnts_green, cnts_brown, frame)

        # Testing
        if len(cnts_green) > 0:
            mask = mask_green + mask_brown
            res = cv2.bitwise_and(frame, frame, mask=mask)
            
            cv2.imshow('res',res)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
     
        if key == ord("q"):
            break

main()
camera.release()
cv2.destroyAllWindows()