#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan

class LidarSectorNode:
    def __init__(self):
        rospy.init_node('lidar_sector_node')
        rospy.Subscriber('/qcar1/scan', LaserScan, self.lidar_callback)  # <-- Adjust topic if needed
        rospy.spin()

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Convert to Cartesian (x, y)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.vstack((xs, ys)).T

        # Remove inf/nan values
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]

        if points.shape[0] == 0:
            rospy.logwarn("No valid LiDAR points received.")
            return

        sector_distances = self.compute_sector_distances(points)
        rospy.loginfo("Sector distances: %s", np.round(sector_distances, 2))

    def compute_sector_distances(self, lidar_data):
        angles = np.arctan2(lidar_data[:, 1], lidar_data[:, 0])
        distances = np.linalg.norm(lidar_data[:, :2], axis=1)

        sector_masks = [
            (angles >= -np.pi/8) & (angles < np.pi/8),           # Front
            (angles >= np.pi/8) & (angles < 3*np.pi/8),          # Front-Right
            (angles >= 3*np.pi/8) & (angles < 5*np.pi/8),        # Right
            (angles >= 5*np.pi/8) & (angles < 7*np.pi/8),        # Rear-Right
            (angles >= 7*np.pi/8) | (angles < -7*np.pi/8),       # Rear
            (angles >= -7*np.pi/8) & (angles < -5*np.pi/8),      # Rear-Left
            (angles >= -5*np.pi/8) & (angles < -3*np.pi/8),      # Left
            (angles >= -3*np.pi/8) & (angles < -np.pi/8),        # Front-Left
        ]

        sector_distances = []
        for mask in sector_masks:
            if np.any(mask):
                min_dist = np.min(distances[mask])
                sector_distances.append(min(min_dist, 5.0))  # Cap at 5 meters
            else:
                sector_distances.append(5.0)
        return np.array(sector_distances)

if __name__ == '__main__':
    try:
        LidarSectorNode()
    except rospy.ROSInterruptException:
        pass

