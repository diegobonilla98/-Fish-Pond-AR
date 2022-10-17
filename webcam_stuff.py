import cv2
import numpy as np


cam = cv2.VideoCapture(1)

WIDTH, HEIGHT = 1920 // 4, 1080 // 4
src_pts = np.float32([[121, 141], [553, 144], [556, 401], [114, 397]])
dst_pts = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

while True:
    ret, frame = cam.read()

    warped = cv2.warpPerspective(frame, M, (WIDTH, HEIGHT))
    lum = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[:, :, 0]
    _, mask = cv2.threshold(lum, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=lambda l: cv2.contourArea(l))
        mom = cv2.moments(contour)
        cX = int(mom["m10"] / mom["m00"])
        cY = int(mom["m01"] / mom["m00"])
        cv2.circle(warped, (cX, cY), 5, (255, 0, 0), -1)

    cv2.imshow("Result", frame)
    cv2.imshow("Warped", warped)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord(' '):
        cv2.imshow("Shoot", frame)
        cv2.imwrite('frame.png', warped)

cv2.destroyAllWindows()

