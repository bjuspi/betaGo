import cv2
import goBoardImageProcessing as gbip

win1 = "win1"
win2 = "win2"
win3 = "win3"
win4 = "win4"
win5 = "win5"
win6 = "win6"

cv2.namedWindow(win1)
cv2.namedWindow(win2)
cv2.namedWindow(win3)
cv2.namedWindow(win4)
cv2.namedWindow(win5)
cv2.namedWindow(win6)

cv2.moveWindow(win1, 0, 0)
cv2.moveWindow(win2, 400, 0)
cv2.moveWindow(win3, 800, 0)
cv2.moveWindow(win4, 0, 330)
cv2.moveWindow(win5, 300, 330)
cv2.moveWindow(win6, 600, 330)

def main():
    image_path = "image/sample/from-cam/12.JPG"
    src = cv2.imread(image_path)
    src = cv2.resize(src, (400, 300), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Convert to Black and White
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Convert to top down perspective transform
    top_down, board = gbip.getTopDownView(thresh, src)

    # Convert to grayscale
    top_down_gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

    top_down_edges = gbip.cannyEdge(top_down_gray)

    ver_hor_frame, board_frame = gbip.getBoardFrame(top_down_edges, top_down)

    cv2.imshow(win1, board)
    cv2.imshow(win2, thresh)
    cv2.imshow(win3, top_down)
    cv2.imshow(win4, top_down_edges)
    cv2.imshow(win5, ver_hor_frame)
    cv2.imshow(win6, board_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()