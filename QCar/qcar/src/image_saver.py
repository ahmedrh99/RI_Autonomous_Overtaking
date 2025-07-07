#!/usr/bin/env python3

import rospy
import os
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime

class ImageSaver:
    def __init__(self):
        rospy.init_node('image_saver_node', anonymous=True)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber('/qcar1/rgbd_color', Image, self.callback)
        self.save_dir = '/home/nvidia/qcar_images'
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_count = 0

        rospy.loginfo("📸 ImageSaver node started. Saving to %s", self.save_dir)

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.image_count += 1
            filename = f"{self.save_dir}/{self.image_count:06d}.png"
            
            cv2.imwrite(filename, cv_image)
            #self.image_count += 1
            rospy.loginfo("Saved image %d: %s", self.image_count, filename)
        except Exception as e:
            rospy.logerr("Failed to save image: %s", str(e))

if __name__ == '__main__':
    try:
        saver = ImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

