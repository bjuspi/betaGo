import math
import cv2
import numpy as np
import goBoardImageProcessing as gbip
import draw

WINDOW_ORIGINAL = 'Original'
WINDOW_TRESH = 'Tresh'
WINDOW_PERSPECTIVE_TRANSFORM = 'Perspective Transform'
WINDOW_LINE_DETECTION = 'Line Detection'
WINDOW_STONE_RECOGNITION = 'Stone Recogntion'

cv2.namedWindow(WINDOW_ORIGINAL)
cv2.namedWindow(WINDOW_TRESH)
cv2.namedWindow(WINDOW_PERSPECTIVE_TRANSFORM)
cv2.namedWindow(WINDOW_LINE_DETECTION)
cv2.namedWindow(WINDOW_STONE_RECOGNITION)

cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_TRESH, 400, 0)
cv2.moveWindow(WINDOW_PERSPECTIVE_TRANSFORM, 800, 0)
cv2.moveWindow(WINDOW_LINE_DETECTION, 0, 300)
cv2.moveWindow(WINDOW_STONE_RECOGNITION, 400, 300)

# CAM_INDEX = 0
# capture = cv2.VideoCapture(CAM_INDEX)

# if capture.isOpened():
#     capture_val, frame = capture.read()
# else:
#     capture_val = False
#     print("Cannot open the camera of index " + str(CAM_INDEX) + ".")
capture_val = True # For testing with static image.
frame = cv2.imread('image/sample/from-cam/4.jpg')

stone_position_printed = False # For testing purpose.

point_position_recorded = False
'''
When the board is full of stones, the intersection recognition might fail. Thus we 
only record the intersection positions once at the start of the game, when the board 
is empty or nearly empty. This point position information is supposed to be unchanged 
and hence can be reused through the entire game.

'''

while capture_val:
    frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA) 
    # The resize propotion is huge, thus a proper interpolation is necessary.
    canvas = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    cnt = gbip.findContours(thresh)
    cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)

    approx_corners = gbip.findApproxCorners(cnt)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = np.concatenate(approx_corners).tolist()
    H, W = thresh.shape
    ref_corners = [[0, H], [0, 0], [W, 0], [W, H]]
    sorted_corners = []

    for ref_corner in ref_corners:
        x = [math.dist(ref_corner, corner) for corner in approx_corners]
        min_position = x.index(min(x))
        sorted_corners.append(approx_corners[min_position])

    # print('\nThe corner points are ...\n')
    # for index, c in enumerate(sorted_corners):
    #     character = chr(65 + index)
    #     print(character, ':', c)
    #     cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    destination_corners, h, w = gbip.getDestinationCorners(sorted_corners)
    un_warped = gbip.unwarp(frame, np.float32(sorted_corners), destination_corners, w, h)
    cropped = un_warped[0:h, 0:w]
    cropped = cv2.resize(cropped, (300, 300))
    cropped = cropped[10:290, 10:290]

    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    cropped_edges = gbip.cannyEdge(cropped_gray)
    lines = gbip.houghLine(cropped_edges)

    intersection_frame = cropped.copy()
    ver_hor_frame = cropped.copy()
    board_frame = cropped.copy()

    # Record point positions once at the start of the game.
    if point_position_recorded is False: 
        if lines is not None:
            h_lines, v_lines = gbip.horizontalVerticalLines(lines)

            if h_lines is not None and v_lines is not None:
                intersection_points = gbip.lineIntersections(h_lines, v_lines)
                points = gbip.clusterPoints(intersection_points)
                augmented_points = gbip.augmentPoints(points)

                point_position_recorded = True

                for h_line in h_lines:
                    draw.drawLine(ver_hor_frame, h_line, (255, 0, 0))
                for v_line in v_lines:
                    draw.drawLine(ver_hor_frame, v_line, (0, 255, 0))

    if point_position_recorded:
        black_stones = []
        white_stones = []
        available_points = []

        # Analyse the stone condition (black/white/empty) at each intersection.
        for index, point in enumerate(augmented_points):
            x = int(point[1]) # The crop step requires integer, this could cause issues.
            y = int(point[0])
            stone_color = gbip.getStoneColor(board_frame, x, y)

            if stone_color == 'black':
                black_stones.append(index)
            elif stone_color == 'white':
                white_stones.append(index)
            else:
                available_points.append(index)

    cv2.imshow(WINDOW_ORIGINAL, canvas)
    cv2.imshow(WINDOW_TRESH, thresh)
    cv2.imshow(WINDOW_PERSPECTIVE_TRANSFORM, cropped)
    cv2.imshow(WINDOW_LINE_DETECTION, ver_hor_frame)
    cv2.imshow(WINDOW_STONE_RECOGNITION, board_frame)

    # capture_val, frame = capture.read()

    if stone_position_printed is False:
        print('Black stones:', black_stones)
        print('White stones:', white_stones)

        stone_position_printed = True

    if cv2.waitKey(1) == 27: break

cv2.destroyAllWindows()
                    

