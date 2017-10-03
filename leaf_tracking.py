from scipy import ndimage
import numpy as np
import imutils
import cv2
 

def detect_contour(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts

def detect_leaf(frame, x, y, w, h, diseased):
    hsv_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = .5+float(w)/500*3
    cv2.rectangle(frame, (x, y), (x+5, y+5), (0, 0, 0), 2)
    if diseased:
        hsv_color = (0, 0, 255)
        cv2.putText( frame, 'Infected', (x,y-10), font, text_size, (255,255,255), 2, cv2.LINE_AA)
    else:
        cv2.putText( frame, 'Not-Infected', (x,y-10), font, text_size, (255,255,255), 2, cv2.LINE_AA)
    w = w + w/5
    h = h + h/5
    cv2.rectangle(frame, (x, y), (x+w, y+h), hsv_color, 2)

def detect_disease(cnts_brown):
    if len(cnts_brown) > 0:
        c = max(cnts_brown, key=cv2.contourArea)
        return True
    return False

def draw_contours(cnts_green, cnts_brown, frame, mask):
    res = cv2.bitwise_and(frame, frame, mask=mask)
    _, cnts, h = cv2.findContours(mask, cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE)            
    cv2.drawContours(frame, max(cnts_green , key=cv2.contourArea), -1, (0,255,0), 3)
    if len(cnts_brown) > 0:
        if(h[0][0][2]>0):
            cv2.drawContours(frame, max(cnts_brown , key=cv2.contourArea), -1, (0,0,255), 3)

    c = [c for c in cnts_green if cv2.arcLength(c, True) > 500]

    for c in c:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        (x, y, w, h) = cv2.boundingRect(c)    
        diseased = True
        if radius > 10:
            diseased = detect_disease(cnts_brown)
            detect_leaf(frame, x, y, w, h, diseased)


def main():

    sensitivity = 15

    greenLower = (60 - sensitivity, 100, 100)

    greenUpper = (60 + sensitivity, 255, 255)

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
            mask = mask_green + mask_brown
            draw_contours(cnts_green, cnts_brown, frame, mask)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
     
        if key == ord("q"):
            break

main()
camera.release()
cv2.destroyAllWindows()
