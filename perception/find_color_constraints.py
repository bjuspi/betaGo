import math
import cv2
import numpy as np
import goBoardImageProcessing as gbip
import draw

WINDOW_ORIGINAL = 'Original'
WINDOW_THRESH = 'Thresh'
WINDOW_PERSPECTIVE_TRANSFORM = 'Perspective Transform'
WINDOW_CANNY_EDGE = 'Canny Edge'
WINDOW_LINE_DETECTION = 'Line Detection'
WINDOW_STONE_RECOGNITION = 'Stone Recogntion'

cv2.namedWindow(WINDOW_ORIGINAL)
cv2.namedWindow(WINDOW_THRESH)
cv2.namedWindow(WINDOW_PERSPECTIVE_TRANSFORM)
cv2.namedWindow(WINDOW_CANNY_EDGE)
cv2.namedWindow(WINDOW_LINE_DETECTION)
cv2.namedWindow(WINDOW_STONE_RECOGNITION)

cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_THRESH, 400, 0)
cv2.moveWindow(WINDOW_PERSPECTIVE_TRANSFORM, 800, 0)
cv2.moveWindow(WINDOW_CANNY_EDGE, 0, 330)
cv2.moveWindow(WINDOW_LINE_DETECTION, 280, 330)
cv2.moveWindow(WINDOW_STONE_RECOGNITION, 560, 330)

CAM_INDEX = 1
capture = cv2.VideoCapture(CAM_INDEX)

if capture.isOpened():
    capture_val, frame = capture.read()
else:
    capture_val = False
    print("Cannot open the camera of index " + str(CAM_INDEX) + ".")

point_position_recorded = False
'''
When the board is full of stones, the intersection recognition might fail. Thus we 
only record the intersection positions once at the start of the game, when the board 
is empty or nearly empty. This point position information is supposed to be unchanged 
and hence can be reused through the entire game.

'''
previous_corners = []
area_correct = False

while capture_val:
    frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA) 
    # The resize propotion is huge, thus a proper interpolation is necessary.
    canvas = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    H, W = thresh.shape

    cv2.imshow(WINDOW_ORIGINAL, canvas)
    cv2.imshow(WINDOW_THRESH, thresh)

    cnt = gbip.findContours(thresh)
    approx_corners = gbip.findApproxCorners(cnt)

    cnt_board_move = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_board_move = cnt_board_move[0] if len(cnt_board_move) == 2 else cnt_board_move[1]

    if len(approx_corners) == 4:
        cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
        cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
        approx_corners = np.concatenate(approx_corners).tolist()
        ref_corners = [[0, H], [0, 0], [W, 0], [W, H]]
        sorted_corners = []

        for ref_corner in ref_corners:
            x = [math.dist(ref_corner, corner) for corner in approx_corners]
            min_position = x.index(min(x))
            sorted_corners.append(approx_corners[min_position])

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

        if np.array_equal(approx_corners, previous_corners) is False:
            is_moved = True
            point_position_recorded = False
        else:
            is_moved = False

        # Record point positions once at the start of the game.
        if not point_position_recorded and is_moved:
            if lines is not None:
                h_lines, v_lines = gbip.horizontalVerticalLines(lines)

                if h_lines is not None and v_lines is not None:
                    try:
                        intersection_points = gbip.lineIntersections(h_lines, v_lines)
                        points = gbip.clusterPoints(intersection_points)
                        augmented_points = gbip.augmentPoints(points)

                        if len(augmented_points) != 100:
                            augmented_points = previous_points.copy()

                        point_position_recorded = True

                        for h_line in h_lines:
                            draw.drawLine(ver_hor_frame, h_line, (255, 0, 0))
                        for v_line in v_lines:
                            draw.drawLine(ver_hor_frame, v_line, (0, 255, 0))
                    except:
                        point_position_recorded = False
                        print("Intersection points cannot be found.")

        if point_position_recorded and (len(augmented_points) == 100):
            black_stones = []
            white_stones = []
            empty = []

            ave_black = 0
            ave_white = 0
            ave_empty = 0

            # Analyse the stone condition (black/white/empty) at each intersection.
            for index, point in enumerate(augmented_points):
                x = int(point[1]) # The crop step requires integer, this could cause issues.
                y = int(point[0])
                
                analyse_area = board_frame[x - 5 : x + 5, y - 5 : y + 5]
    
                average_color_per_row = np.average(analyse_area, axis=0)
                average_color = np.average(average_color_per_row, axis=0)

                if (index == 0 or index == 9 or index == 90 or index == 99):
                    black_stones.append(average_color)
                if (index == 44 or index == 45 or index == 54 or index == 55):
                    black_stones.append(average_color)
                if (index == 22 or index == 27 or index == 72 or index == 77):
                    empty.append(average_color)

            if (len(black_stones) == 4):
                print("black stones' average color: ", black_stones)
                print("average of black: ", sum(black_stones) / 4)

            if (len(white_stones) == 4):
                print("white stones' average color: ", white_stones)
                print("average of white: ", sum(white_stones) / 4)

            if (len(empty) == 4):
                print("empty intersection's average color: ", empty)
                print("average of empty: ", sum(empty) / 4)

            if (len(black_stones) == 4 and len(empty) == 4):
                print("black_max: ", (sum(black_stones) / 4 + sum(empty) / 4)/2)

            if (len(white_stones) == 4 and len(empty) == 4):
                print("white_min: ", (sum(white_stones) / 4 + sum(empty) / 4)/2)

            previous_points = augmented_points.copy()
    else: 
        cropped = np.zeros((H, W, 3), np.uint8)
        cropped_edges = np.zeros((H, W, 3), np.uint8)
        ver_hor_frame = np.zeros((H, W, 3), np.uint8)
        board_frame = np.zeros((H, W, 3), np.uint8)
    
    cv2.imshow(WINDOW_PERSPECTIVE_TRANSFORM, cropped)
    cv2.imshow(WINDOW_CANNY_EDGE, cropped_edges)
    cv2.imshow(WINDOW_LINE_DETECTION, ver_hor_frame) # Not always working and not always required, consider not displaying this.
    cv2.imshow(WINDOW_STONE_RECOGNITION, board_frame)

    capture_val, frame = capture.read()
    previous_corners = approx_corners.copy()

    if cv2.waitKey(1) == 27: 
        break

cv2.destroyAllWindows()
                    

