import time
from Fish import Fish, cv2, np


# def mouse_callback(event, x, y, flags, param):
#     global mouse_pos, is_attraction_detected
#     if event == cv2.EVENT_LBUTTONDOWN:
#         is_attraction_detected = True
#     elif event == cv2.EVENT_MOUSEMOVE and is_attraction_detected:
#         mouse_pos = np.array([x, y])
#     elif event == cv2.EVENT_LBUTTONUP:
#         is_attraction_detected = False


BACKGROUND = (237, 242, 249)
WIDTH, HEIGHT = 1920, 1080
NUM_FISHES = 60

cam = cv2.VideoCapture(1)
CAM_DOWNSCALE_FAC = 4
CAM_WIDTH, CAM_HEIGHT = WIDTH // CAM_DOWNSCALE_FAC, HEIGHT // CAM_DOWNSCALE_FAC
src_pts = np.float32([[121, 141], [553, 144], [556, 401], [114, 397]])
dst_pts = np.float32([[0, 0], [CAM_WIDTH, 0], [CAM_WIDTH, CAM_HEIGHT], [0, CAM_HEIGHT]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

cv2.namedWindow('Canvas', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.setMouseCallback('Canvas', mouse_callback)

fishes = [Fish((WIDTH, HEIGHT), personality=None) for _ in range(NUM_FISHES)]

attraction_center = np.array([WIDTH / 2., HEIGHT / 2.])
is_attraction_detected = False

while True:
    t0 = time.time()
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)

    ret, frame = cam.read()
    warped = cv2.warpPerspective(frame, M, (WIDTH, HEIGHT))
    lum = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[:, :, 0]
    _, mask = cv2.threshold(lum, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_attraction_detected = False
    if len(contours) > 0:
        is_attraction_detected = True
        contour = max(contours, key=lambda l: cv2.contourArea(l))
        mom = cv2.moments(contour)
        cX = int(mom["m10"] / mom["m00"])
        cY = int(mom["m01"] / mom["m00"])
        attraction_center = np.array([cX, cY], np.float32) * CAM_DOWNSCALE_FAC
        # cv2.circle(warped, (cX, cY), 5, (255, 0, 0), -1)

    for fish in fishes:
        fish.update(attraction_center if is_attraction_detected else None)
        canvas = fish.display(canvas)

    cv2.imshow('Canvas', canvas)
    key = cv2.waitKey(1)
    print("FPS:", int(1. / (time.time() - t0)))
    if key == ord('q'):
        break


cv2.destroyAllWindows()
