import sys
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
import img_processing as ip

class image_processing:
    def __init__(self):
        # ROS publisher:
        # self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage)
        # self.bridge = CvBridge()

        # ROS subscriber:
        self.subscriber = rospy.Subscriber("/liveview/compressed", CompressedImage, self.callback,  queue_size=1)
        print("Subscribed to /liveview/compressed.")

    def callback(self, ros_data):
        print("Received image of type: '%s'." % ros_data.format)
        
        np_arr = np.fromstring(ros_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imshow('Frame', frame)

        global previous_cnrs, previous_intxns
        previous_cnrs, previous_intxns = ip.imgProcessing(frame, previous_cnrs, previous_intxns)

        cv2.waitKey(2)

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

previous_cnrs, previous_intxns = ([],)*2

if __name__ == '__main__':
    main(sys.argv)