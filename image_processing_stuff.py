import cv2

frame = cv2.imread('frame.png')
frame = cv2.resize(frame, None, fx=0.25, fy=0.25)

lum = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
_, mask = cv2.threshold(lum, 200, 255, cv2.THRESH_BINARY)
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    contour = max(contours, key=lambda l: cv2.contourArea(l))
    mom = cv2.moments(contour)
    cX = int(mom["m10"] / mom["m00"])
    cY = int(mom["m01"] / mom["m00"])
    cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

cv2.imshow("Result", frame)
cv2.imshow("Mask", mask)
cv2.waitKey()
cv2.destroyAllWindows()
