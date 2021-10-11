import cv2
import numpy as np
import matplotlib.pyplot as plt
import goBoardImageProcessing as gbip
import colorsys

cap = cv2.VideoCapture(1)

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

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def get_color(img, x, y):
    size = 3
    points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            try:
                points.append(img[y + i, x + j])
            except:
                print('\nThe coordinates are out of bounds')
                return
    norm = float(len(points))
    norm = float(norm*255)
    color = (sum(p[2] for p in points) / norm,
             sum(p[1] for p in points) / norm,
             sum(p[0] for p in points) / norm)
    print("x: " + str(x) + " y: " + str(y))
    print("color: " + str(color))
    hue, luma, saturation = colorsys.rgb_to_hls(*color)
    color = colorsys.hls_to_rgb(hue, 0.5, 1.)
    print("store color: " + str(color))
    return 

while True:
    # ret, frame = cap.read()
    frame = cv2.imread('image/sample/from-cam/4.jpg')

    frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    canvas = frame.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    print('\nThe corner points are ...\n')
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
    print(approx_corners)
    destination_corners, h, w = gbip.get_destination_points(approx_corners)
    un_warped = gbip.unwarp(frame, np.float32(approx_corners), destination_corners, w, h)
    cropped = un_warped[0:h, 0:w]
    cropped = cv2.resize(cropped, (300, 300))
    cropped = cropped[10:290, 10:290]
    # cv2.imwrite('image/sample/from-code/4.jpg', cropped)

    transformedGray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    transformedEdges = gbip.canny_edge(transformedGray)
    transformedLines = gbip.hough_line(transformedEdges)

    houghLine = cropped.copy()
    intersection_frame = cropped.copy()

    if transformedLines is not None:    
        h_lines, v_lines = gbip.h_v_lines(transformedLines)

        ver_hor_frame = cropped.copy()

        for line in transformedLines:
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
                cv2.line(houghLine, (x1,y1), (x2,y2), (0,0,255), 2)

        if h_lines is not None and v_lines is not None:
            intersection_points = gbip.line_intersections(h_lines, v_lines)
            points = gbip.cluster_points(intersection_points)
            
            augmented_points = gbip.augment_points(points)
            for index, point in enumerate(augmented_points):
                x, y = point
                print("index: " + str(index) + " x: " + str(x) + " y: " + str(y))
                
                color = get_color(cropped, int(x), int(y))
                
                cv2.circle(intersection_frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                if index == 42:
                    cv2.circle(intersection_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
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

    houghLine = cv2.resize(houghLine, (300, 300))
    intersection_frame = cv2.resize(intersection_frame, (300, 300))

    cv2.imshow(win1, canvas)
    cv2.imshow(win2, thresh)
    cv2.imshow(win3, cropped)
    cv2.imshow(win4, houghLine)
    cv2.imshow(win5, ver_hor_frame)
    cv2.imshow(win6, intersection_frame)

    c = cv2.waitKey(1)
    if c == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()