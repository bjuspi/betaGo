import cv2
import goBoardImageProcessing as gbip

cap = cv2.VideoCapture(1)

win1 = "Original"
win2 = "Gray"
win3 = "Gray Blur"
win4 = "Canny Edge"
win5 = "Hough Line"
win6 = "Vertical Horizontal"

cv2.namedWindow(win1)
cv2.namedWindow(win2)
cv2.namedWindow(win3)
cv2.namedWindow(win4)
cv2.namedWindow(win5)
cv2.namedWindow(win6)

cv2.moveWindow(win1, 0, 0)
cv2.moveWindow(win2, 400, 0)
cv2.moveWindow(win3, 800, 0)
cv2.moveWindow(win4, 0, 300)
cv2.moveWindow(win5, 400, 300)
cv2.moveWindow(win6, 800, 300)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, src = cap.read()

    src = cv2.resize(src, (400, 300), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Convert to Black and White
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Convert to top down perspective transform
    top_down, board = gbip.getTopDownView(thresh, src)

    # Convert to grayscale
    if (top_down is not None):
        top_down_gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

        top_down_edges = gbip.cannyEdge(top_down_gray)

        # ver_hor_frame, board_frame = gbip.getBoardFrame(top_down_edges, top_down)

        cv2.imshow(win3, board)
        cv2.imshow(win4, top_down)

    cv2.imshow(win1, src)
    cv2.imshow(win2, thresh)
    # cv2.imshow(win5, top_down_edges)
    # cv2.imshow(win5, ver_hor_frame)
    # cv2.imshow(win6, board_frame)

    c = cv2.waitKey(1)
    if c == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
