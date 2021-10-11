import cv2

WINDOW_ORIGINAL = 'Original'
WINDOW_THRESH = 'Thresh'
WINDOW_CONTOURS = 'Contours'

cv2.namedWindow(WINDOW_ORIGINAL)
cv2.namedWindow(WINDOW_THRESH)
cv2.namedWindow(WINDOW_CONTOURS)

cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_THRESH, 400, 0)
cv2.moveWindow(WINDOW_CONTOURS, 800, 0)

CAM_INDEX = 1
capture = cv2.VideoCapture(CAM_INDEX)

if capture.isOpened():
    capture_val, frame = capture.read()
else:
    capture_val = False
    print("Cannot open the camera of index " + str(CAM_INDEX) + ".")

while capture_val:
    frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA) 
    canvas = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    cnt_board_move = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_board_move = cnt_board_move[0] if len(cnt_board_move) == 2 else cnt_board_move[1]

    for c in cnt_board_move:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) == 4:
            print("Current area is " + str(area) + ".")
            cv2.drawContours(canvas, approx, -1, (0, 255, 0), 3)

        # if 10000 < area < 30000:
        #     print("Area within constraints.")

    cv2.imshow(WINDOW_ORIGINAL, frame)
    cv2.imshow(WINDOW_THRESH, thresh)
    cv2.imshow(WINDOW_CONTOURS, canvas)

    capture_val, frame = capture.read()
    if cv2.waitKey(1) == 27: break

cv2.destroyAllWindows()
