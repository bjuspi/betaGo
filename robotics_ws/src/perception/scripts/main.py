#!/usr/bin/python3
# Python libs
import sys, time, math

# numpy and scipy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

import goBoardImageProcessing as gbip
import draw

VERBOSE=True

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

print(sys.version)
print("Opencv: " + cv2.__version__)

def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def listener():
    stone_position_printed = False # For testing purpose.
    point_position_recorded = False
    previous_corners = []
    previous_blacks = []
    area_correct = False

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    pub = rospy.Publisher('chatter', Float32MultiArray, queue_size=10)
    rospy.init_node('listener', anonymous=True)

    while not rospy.is_shutdown():
        ros_data = rospy.wait_for_message("/liveview/compressed", CompressedImage)
        
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        #Main Perception
        frame = cv2.resize(image_np, (400, 300), interpolation=cv2.INTER_AREA) 
        canvas = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        H, W = thresh.shape

        cnt = gbip.findContours(thresh)
        approx_corners = gbip.findApproxCorners(cnt)

        cnt_board_move = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_board_move = cnt_board_move[0] if len(cnt_board_move) == 2 else cnt_board_move[1]
        # for c in cnt_board_move:
        #     area = cv2.contourArea(c)
        #     if 10000 < area < 30000: # Constraints change if the board proportion in the image changes. Better set this after fix cam & board's relative position.
        #         area_correct = True
        area_correct = True

        if len(approx_corners) == 4 and area_correct:
            cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)

            approx_corners = np.concatenate(approx_corners).tolist()
            ref_corners = [[0, H], [0, 0], [W, 0], [W, H]]
            sorted_corners = []

            for ref_corner in ref_corners:
                x = [dist(ref_corner, corner) for corner in approx_corners]
                min_position = x.index(min(x))
                sorted_corners.append(approx_corners[min_position])

            destination_corners, h, w = gbip.getDestinationCorners(sorted_corners)
            un_warped = gbip.unwarp(frame, np.float32(sorted_corners), destination_corners, w, h)
            cropped = un_warped[0:h, 0:w]
            cropped = cv2.resize(cropped, (300, 300))
            cropped = cropped[15:-15, 15:-15]

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

                new_black = list(set(black_stones) - set(previous_blacks))
                if (len(new_black) > 0):
                    new_index = 99 - new_black[0]
                    print("New Black: ", new_index)
                    msg = Float32MultiArray()
                    msg.data = [new_index//10, new_index%10]
                    rospy.loginfo(msg)
                    pub.publish(msg)
                    previous_blacks = black_stones.copy()
                
                if stone_position_printed:
                    print('Black stones:', black_stones)
                    print('White stones:', white_stones)

                previous_points = augmented_points.copy()

        else: 
            cropped = np.zeros((H, W, 3), np.uint8)
            cropped_edges = np.zeros((H, W, 3), np.uint8)
            ver_hor_frame = np.zeros((H, W, 3), np.uint8)
            board_frame = np.zeros((H, W, 3), np.uint8)

        cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)

        cv2.imshow(WINDOW_ORIGINAL, canvas)
        cv2.imshow(WINDOW_THRESH, thresh)
        cv2.imshow(WINDOW_PERSPECTIVE_TRANSFORM, cropped)
        cv2.imshow(WINDOW_CANNY_EDGE, cropped_edges)
        cv2.imshow(WINDOW_LINE_DETECTION, ver_hor_frame)
        cv2.imshow(WINDOW_STONE_RECOGNITION, board_frame)

        previous_corners = approx_corners.copy()
        area_correct = False

        cv2.waitKey(2)

if __name__ == '__main__':
    # main(sys.argv)
    listener()