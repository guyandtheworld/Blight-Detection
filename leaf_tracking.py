import numpy as np
import imutils
import cv2
 

def detect_brown():
    pass

def detect_leaf(frame, x, y, w, h, diseased):
    color = "green"
    hsv_color = (0, 255, 0)
    if diseased:
        color = "red"
        hsv_color = (0, 0, 255)
    w = w + w/5
    h = h + h/5
    cv2.rectangle(frame, (x, y), (x+w, y+h), hsv_color, 2)

def detect_disease(image, brown):
    cv2.circle(image, brown, 5, (0, 0, 255), -1)

def draw_contours(cnts, frame):
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    (x, y, w, h) = cv2.boundingRect(c)
    M = cv2.moments(c)
    brown = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    diseased = True
    if radius > 10:
        detect_leaf(frame, x, y, w, h, False)
        if diseased:
            detect_disease(frame, brown)

def main():

    greenLower = (29, 86, 6)
    
    greenUpper = (64, 255, 255)
    
    camera = cv2.VideoCapture(0)
    
    while True:
        (_, frame) = camera.read()

        frame = imutils.resize(frame, width=600)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, greenLower, greenUpper)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            draw_contours(cnts, frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
     
        if key == ord("q"):
            break

main()
camera.release()
cv2.destroyAllWindows()