#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

import roslib
import rospy
import numpy as np
import cv2
from qcar.q_essential import Camera3D
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class RGBDNode(object):
    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()

        # Initialize publishers for RealSense camera topics
        self.color_pub = rospy.Publisher('/camera/color/image', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/camera/depth/image', Image, queue_size=10)
        self.color_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=10)
        self.depth_info_pub = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=10)

        self.color_compressed_pub = rospy.Publisher('/camera/color/image/compressed', CompressedImage, queue_size=10)
        self.depth_compressed_pub = rospy.Publisher('/camera/depth/image/compressed', CompressedImage, queue_size=10)
        
        # Initialize the camera with full resolution
        self.rgbd = Camera3D(mode='RGB&DEPTH', frame_width_RGB=1920, frame_height_RGB=1080, frame_width_depth=1280, frame_height_depth=720)

        # Placeholder calibration values
        self.fx = 600  # Focal length in x-axis
        self.fy = 600  # Focal length in y-axis
        self.cx = 960  # Optical center x
        self.cy = 540  # Optical center y

        # Get readings and publish
        while not rospy.is_shutdown():
            self.rgbd.read_RGB()
            self.rgbd.read_depth(dataMode='m')

            # Normalize depth data for display
            depth_display = self.normalize_depth(self.rgbd.image_buffer_depth_m)

            timestamp = rospy.Time.now()

            self.publish_image(self.color_pub, self.rgbd.image_buffer_RGB, "bgr8", 'camera_rgb', timestamp)
            self.publish_image(self.depth_pub, self.rgbd.image_buffer_depth_m, "32FC1", 'camera_depth', timestamp)
            self.publish_image(self.color_compressed_pub, self.rgbd.image_buffer_RGB, "jpeg", 'camera_rgb', compressed=True, timestamp=timestamp)
            self.publish_image(self.depth_compressed_pub, depth_display, "mono8", 'camera_depth', compressed=True, timestamp=timestamp)
            self.publish_camera_info(self.color_info_pub, 'camera_rgb', timestamp)
            self.publish_camera_info(self.depth_info_pub, 'camera_depth', timestamp)

        self.rgbd.terminate()

    def normalize_depth(self, depth_image):
        """Normalize depth image for display."""
        min_depth = np.min(depth_image[depth_image > 0])
        max_depth = np.max(depth_image)
        normalized_depth = (depth_image - min_depth) / (max_depth - min_depth)
        normalized_depth = (normalized_depth * 255).astype(np.uint8)
        return normalized_depth

    def publish_image(self, publisher, img_data, encoding, frame_id, timestamp, compressed=False):
        try:
            if compressed:
                pub_img = self.bridge.cv2_to_compressed_imgmsg(img_data, "png" if encoding == "mono8" else "jpeg")
                pub_img.format = "png" if encoding == "mono8" else "jpeg"
            else:
                pub_img = self.bridge.cv2_to_imgmsg(img_data, encoding)
            pub_img.header.stamp = timestamp
            pub_img.header.frame_id = frame_id
            publisher.publish(pub_img)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def publish_camera_info(self, publisher, frame_id, timestamp):
        cam_info = CameraInfo()
        cam_info.header.stamp = timestamp
        cam_info.header.frame_id = frame_id
        # Fill camera info message with actual values
        cam_info.width = 1920 if 'rgb' in frame_id else 1280
        cam_info.height = 1080 if 'rgb' in frame_id else 720
        cam_info.K = [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]
        cam_info.P = [self.fx, 0, self.cx, 0, 0, self.fy, self.cy, 0, 0, 0, 1, 0]
        publisher.publish(cam_info)

if __name__ == '__main__':
    rospy.init_node('rgbd_node')
    r = RGBDNode()
    rospy.spin()
