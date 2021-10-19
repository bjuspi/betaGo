import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
import img_processing as ip


def listener():
    pub = rospy.Publisher('chatter', Float32MultiArray, queue_size=10)
    rospy.init_node('listener', anonymous=True)

    while not rospy.is_shutdown():
        ros_data = rospy.wait_for_message(
            "/liveview/compressed", CompressedImage)
        np_arr = np.fromstring(ros_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        global previous_cnrs, previous_intxns, previous_bks
        new_bks = []
        cropped, previous_cnrs, previous_intxns, new_bks = ip.imgProcessing(
            frame, previous_cnrs, previous_intxns, previous_bks)

        # Check wether there is a new black stone, if so, publish its coordinates.
        new_bk_idx = set(new_bks) - set(previous_bks)
        if (len(new_bk_idx) != set()):
            print("Index of the new black stone: ", new_bk_idx, + ".")
            previous_bks = new_bk_idx

            new_idx = 99 - new_bk_idx[0]
            msg = Float32MultiArray()
            msg.data = [new_idx//10, new_idx % 10]
            rospy.loginfo(msg)
            pub.publish(msg)

        # Display images.
        if (cropped == frame):
            cropped = cv2.resize(cropped, (400, 300),
                                 interpolation=cv2.INTER_AREA)
        else:
            cropped = cv2.resize(cropped, (300, 400),
                                 interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)
        cv2.imshow('Frame', frame)
        cv2.imshow('Perspective Transformed', cropped)

        # For calibration purpose.
        key = cv2.waitKey(3)
        if (key == 'a'):
            ip.findAreaConstraints(frame)
        elif (key == 'c'):
            ip.colorCalibration(cropped, previous_intxns)


WINDOW_ORIGINAL = 'Original'
WINDOW_PERSPECTIVE_TRANSFORMED = 'Perspective Transformed'
cv2.namedWindow(WINDOW_ORIGINAL)
cv2.namedWindow(WINDOW_PERSPECTIVE_TRANSFORMED)
cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_PERSPECTIVE_TRANSFORMED, 400, 0)

previous_cnrs, previous_intxns, previous_bks = ([],)*3

if __name__ == '__main__':
    listener()
