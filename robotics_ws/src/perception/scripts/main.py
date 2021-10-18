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
# cv2.namedWindow(WINDOW_CANNY_EDGE)
# cv2.namedWindow(WINDOW_LINE_DETECTION)
# cv2.namedWindow(WINDOW_STONE_RECOGNITION)

cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_THRESH, 400, 0)
cv2.moveWindow(WINDOW_PERSPECTIVE_TRANSFORM, 800, 0)
# cv2.moveWindow(WINDOW_CANNY_EDGE, 0, 330)
# cv2.moveWindow(WINDOW_LINE_DETECTION, 280, 330)
# cv2.moveWindow(WINDOW_STONE_RECOGNITION, 560, 330)

print(sys.version)
print("Opencv: " + cv2.__version__)

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        # self.image_pub = rospy.Publisher("/output/black_position", Float32MultiArray, queue_size=10)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/liveview/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print("subscribed to /liveview/compressed")


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print('received image of type: "%s"' % ros_data.format)

        ### direct conversion to CV2 ####
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
            # cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
            # cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)

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
            cropped = cropped[15:-15, 15:-15]

            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            cropped_edges = gbip.cannyEdge(cropped_gray)

            lines = gbip.houghLine(cropped_edges)

            print(sorted_corners)

        cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
        cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
        
        cv2.imshow(WINDOW_ORIGINAL, canvas)
        cv2.imshow(WINDOW_THRESH, thresh)
        cv2.imshow(WINDOW_PERSPECTIVE_TRANSFORM, cropped)
        # previous_corners = approx_corners.copy()

        cv2.waitKey(2)

        #### Create Float32MultiArray msg ####
        # msg = Float32MultiArray()
        # msg.data = [1.0, 1.0]
        # # # Publish new image
        # rospy.loginfo(msg)
        # self.image_pub.publish(msg)
        
        # self.subscriber.unregister()

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
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
            # cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
            # cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)

            approx_corners = np.concatenate(approx_corners).tolist()
            ref_corners = [[0, H], [0, 0], [W, 0], [W, H]]
            sorted_corners = []

            print(approx_corners)

            # for ref_corner in ref_corners:
            #     x = [math.dist(ref_corner, corner) for corner in approx_corners]
            #     min_position = x.index(min(x))
            #     sorted_corners.append(approx_corners[min_position])

            # destination_corners, h, w = gbip.getDestinationCorners(sorted_corners)
            # un_warped = gbip.unwarp(frame, np.float32(sorted_corners), destination_corners, w, h)
            # cropped = un_warped[0:h, 0:w]
            # cropped = cv2.resize(cropped, (300, 300))
            # cropped = cropped[15:-15, 15:-15]

            # cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # cropped_edges = gbip.cannyEdge(cropped_gray)

            # lines = gbip.houghLine(cropped_edges)

            # print(sorted_corners)
        else: 
            cropped = np.zeros((H, W, 3), np.uint8)

        cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
       # cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)

        cv2.imshow(WINDOW_ORIGINAL, canvas)
        cv2.imshow(WINDOW_THRESH, thresh)
        # cv2.imshow(WINDOW_PERSPECTIVE_TRANSFORM, cropped)

        cv2.waitKey(2)

if __name__ == '__main__':
    # main(sys.argv)
    listener()