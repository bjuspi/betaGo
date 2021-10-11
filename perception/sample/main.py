import cv2
from board_filter import *
from statistics import mean

cap = cv2.VideoCapture(0)

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
    ret, frame = cap.read()
    # frame = cv2.imread('4.jpg')

    hough_line_frame = cv2.resize(frame.copy(), (400, 300))
    # ver_hor_frame = cv2.resize(frame.copy(), (400, 300))
    # intersection_frame = cv2.resize(frame.copy(), (400, 300))

    frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))
    edges = canny_edge(gray_blur)
    lines = hough_line(edges)
    print(lines)
    if lines is not None:        
        for line in lines:
            rho,theta = line[0]
            if not np.isnan(rho) and not np.isnan(theta):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(hough_line_frame, (x1,y1), (x2,y2), (0,0,255), 2)
    h_lines, v_lines = h_v_lines(lines)
    if h_lines is not None and v_lines is not None:
        print("h_lines: " + str(h_lines))
        print("v_lines: " + str(v_lines))
        try:
            for h_line in h_lines:
                rho, theta = h_line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(ver_hor_frame, (x1,y1),(x2,y2),(255,0,0),2)
        except ValueError:
            print("h_line error")
            break
        try:
            for v_line in v_lines:
                rho, theta = v_line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(ver_hor_frame, (x1,y1),(x2,y2),(0,255,0),2)
        except ValueError:
            print("v_line error")
            break
    intersection_points = line_intersections(h_lines, v_lines)
    points = cluster_points(intersection_points)
    augmented_points = augment_points(points)
    for point in augmented_points:
        x, y = point
        cv2.circle(intersection_frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imshow(win1, frame)
    cv2.imshow(win2, gray_blur)
    cv2.imshow(win3, edges)
    cv2.imshow(win4, hough_line_frame)
    # cv2.imshow(win5, hough_line_frame)
    # cv2.imshow(win6, ver_hor_frame)

    c = cv2.waitKey(1)
    if c == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
