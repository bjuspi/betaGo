import cv2
import numpy as np
import goBoardImageProcessing as gbip
import draw

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

frame = cv2.imread('image/sample/from-cam/4.jpg')

frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)
canvas = frame.copy()

gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

cnt = gbip.findContours(thresh)
cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)

approx_corners = gbip.findApproxCorners(cnt)
cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
approx_corners = sorted(np.concatenate(approx_corners).tolist())
print('\nThe corner points are ...\n')
for index, c in enumerate(approx_corners):
    character = chr(65 + index)
    print(character, ':', c)
    cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# Rearranging the order of the corner points
approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]

destination_corners, h, w = gbip.getDestinationCorners(approx_corners)
un_warped = gbip.unwarp(frame, np.float32(approx_corners), destination_corners, w, h)
cropped = un_warped[0:h, 0:w]
cropped = cv2.resize(cropped, (300, 300))
cropped = cropped[10:290, 10:290]

cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

cropped_edges = gbip.cannyEdge(cropped_gray)
lines = gbip.houghLine(cropped_edges)

intersection_frame = cropped.copy()
ver_hor_frame = cropped.copy()
board_frame = cropped.copy()

if lines is not None:    
    h_lines, v_lines = gbip.horizontalVerticalLines(lines)

    if h_lines is not None and v_lines is not None:
        intersection_points = gbip.lineIntersections(h_lines, v_lines)
        points = gbip.clusterPoints(intersection_points)
        augmented_points = gbip.augmentPoints(points)

        for index, point in enumerate(augmented_points):
            x = int(point[1]) # The crop step requires integer, this could cause issues.
            y = int(point[0])
            color = gbip.getStoneColor(board_frame, x, y)
            # print(color)
            # cv2.waitKey(0)                
            cv2.circle(intersection_frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
        for h_line in h_lines:
            draw.drawLine(ver_hor_frame, h_line, (255, 0, 0))
            
        for v_line in v_lines:
            draw.drawLine(ver_hor_frame, v_line, (0, 255, 0))

cv2.imshow(win1, canvas)
cv2.imshow(win2, thresh)
cv2.imshow(win3, cropped)
cv2.imshow(win4, ver_hor_frame)
cv2.imshow(win5, intersection_frame)
cv2.imshow(win6, board_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()