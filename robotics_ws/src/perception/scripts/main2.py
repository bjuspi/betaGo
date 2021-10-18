import sys
import numpy as np
import cv2
from perception.main import WINDOW_ORIGINAL, WINDOW_PERSPECTIVE_TRANSFORM
import rospy
from sensor_msgs.msg import CompressedImage
import img_processing as ip


class image_processing:
    def __init__(self):
        # ROS publisher:
        # self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage)
        # self.bridge = CvBridge()

        # ROS subscriber:
        self.subscriber = rospy.Subscriber(
            "/liveview/compressed", CompressedImage, self.callback,  queue_size=1)
        print("Subscribed to /liveview/compressed.")

    def callback(self, ros_data):
        print("Received image of type: '%s'." % ros_data.format)

        np_arr = np.fromstring(ros_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        global previous_cnrs, previous_intxns
        cropped, previous_cnrs, previous_intxns = ip.imgProcessing(
            frame, previous_cnrs, previous_intxns)

        if (cropped == frame):
            cropped = cv2.resize(cropped, (400, 300),
                                 interpolation=cv2.INTER_AREA)
        else:
            cropped = cv2.resize(cropped, (300, 400),
                                 interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)

        cv2.imshow('Frame', frame)
        cv2.imshow('Perspective Transformed', cropped)

        key = cv2.waitKey(3)
        if (key == 'a'):
            ip.findAreaConstraints(frame)
        elif (key == 'c'):
            ip.colorCalibration(cropped, previous_intxns)

        # Create CompressedIamge
        # msg = CompressedImage()
        # msg.header.stamp = rospy.Time.now()
        # msg.format = "jpeg"
        # msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # # Publish new image
        # self.image_pub.publish(msg)
        # self.subscriber.unregister()


def main(argv):
    img_processing = image_processing()
    rospy.init_node('image_processing', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module.")
    cv2.destroyAllWindows()


WINDOW_ORIGINAL = 'Original'
WINDOW_PERSPECTIVE_TRANSFORMED = 'Perspective Transformed'
cv2.namedWindow(WINDOW_ORIGINAL)
cv2.namedWindow(WINDOW_PERSPECTIVE_TRANSFORMED)
cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_PERSPECTIVE_TRANSFORMED, 400, 0)

previous_cnrs, previous_intxns = ([],)*2

if __name__ == '__main__':
    main(sys.argv)
