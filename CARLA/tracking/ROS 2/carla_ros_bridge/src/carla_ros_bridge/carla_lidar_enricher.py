#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pcl2
from std_msgs.msg import Header

class CarlaLidarEnricher(Node):
    def __init__(self):
        super().__init__('carla_lidar_enricher')

        self.declare_parameter("input_topic", "/carla/ego_vehicle/lidar")
        self.declare_parameter("output_topic", "/points_raw")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.sub = self.create_subscription(PointCloud2, input_topic, self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

        self.get_logger().info(f"Subscribed to {input_topic}, publishing enriched data to {output_topic}")
        
    def callback(self, msg: PointCloud2):
        try:
            raw = list(pcl2.read_points(
                msg, field_names=("x","y","z","intensity"), skip_nans=True
            ))
            if not raw:
                return

            pts = np.array(raw, dtype=[
                ("x", np.float32), ("y", np.float32),
                ("z", np.float32), ("i", np.float32),
            ])
            x, y, z, i0 = pts["x"], pts["y"], pts["z"], pts["i"]

            # — for IRC we only need these fields —
            intensity   = np.clip(i0, 0, 1).astype(np.uint8) # UINT8 [0–255]
            return_type = np.zeros(len(x),    dtype=np.uint8) # UINT8
            channel     = np.zeros(len(x),    dtype=np.uint16) # UINT16

            # 4) build a 16-byte C-aligned dtype
            point_dt = np.dtype([
                ("x",          np.float32),  # 0–3
                ("y",          np.float32),  # 4–7
                ("z",          np.float32),  # 8–11
                ("intensity",  np.uint8),    # 12
                ("return_type",np.uint8),    # 13
                ("channel",    np.uint16),   # 14–15
            ], align=True)  # ensures offsets and total itemsize=16

            cloud = np.zeros(len(x), dtype=point_dt)
            cloud["x"], cloud["y"], cloud["z"]         = x, y, z
            cloud["intensity"], cloud["return_type"] = intensity, return_type
            cloud["channel"] = channel

            # 5) describe the fields with keyword args
            fields = [
                PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.UINT8,   count=1),
                PointField(name="return_type",offset=13, datatype=PointField.UINT8,   count=1),
                PointField(name="channel",    offset=14, datatype=PointField.UINT16,  count=1),
            ]

            enriched_msg = pcl2.create_cloud(msg.header, fields, cloud)
            enriched_msg.point_step = 16
            enriched_msg.is_dense   = False
            self.pub.publish(enriched_msg)

        except Exception as e:
            self.get_logger().error(f"Enricher failed: {e}")

    '''
    def callback(self, msg: PointCloud2):
        try:
            # 1) Read raw points (from CARLA)
            raw = list(pcl2.read_points(msg, field_names=("x","y","z","intensity"), skip_nans=True))
            if not raw:
                return

            # 2) Unpack into numpy arrays
            pts = np.array(raw, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("i", np.float32)])
            x, y, z, i0 = pts["x"], pts["y"], pts["z"], pts["i"]

            # 3) Compute Autoware-compatible fields
            intensity = np.clip(i0, 0, 1).astype(np.float32)
            return_type = np.zeros_like(intensity, dtype=np.uint8)  # UINT8
            ring = np.zeros_like(x, dtype=np.uint16)                # UINT16
            channel = np.zeros_like(x, dtype=np.uint8)              # UINT8 (required by Autoware)
            azimuth = np.arctan2(y, x).astype(np.float32)           # FLOAT32
            elevation = np.zeros_like(azimuth, dtype=np.float32)    # FLOAT32
            distance = np.linalg.norm(np.vstack((x,y,z)).T, axis=1).astype(np.float32)
            timestamp = (np.linspace(0, 1, len(x)) * 1e9).astype(np.uint32)  # UINT32 ns

            # 4) Structured array with preprocessor-friendly layout
            point_dt = np.dtype([
                ("x", np.float32),          # 0-3
                ("y", np.float32),          # 4-7
                ("z", np.float32),          # 8-11
                ("intensity", np.float32),  # 12-15
                ("return", np.uint8),  # 16
                ("ring", np.uint16),        # 18-19 (uint8 + padding + uint16)
                ("channel", np.uint8),     # 20
                ("azimuth", np.float32),   # 24-27 (4-byte aligned)
                ("elevation", np.float32), # 28-31
                ("distance", np.float32),  # 32-35
                ("time", np.uint32),  # 36-39
            ], align=True)  # Critical for Autoware!

            # 5) Populate the structured array
            enriched = np.zeros(len(x), dtype=point_dt)
            enriched["x"], enriched["y"], enriched["z"] = x, y, z
            enriched["intensity"], enriched["return"] = intensity, return_type
            enriched["ring"], enriched["channel"] = ring, channel
            enriched["azimuth"], enriched["elevation"] = azimuth, elevation
            enriched["distance"], enriched["time"] = distance, timestamp

            # 6) PointField definition (must match dtype EXACTLY)
            fields = [
                PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="return", offset=16, datatype=PointField.UINT8, count=1),
                PointField(name="ring", offset=18, datatype=PointField.UINT16, count=1),
                PointField(name="channel", offset=20, datatype=PointField.UINT8, count=1),
                PointField(name="azimuth", offset=24, datatype=PointField.FLOAT32, count=1),
                PointField(name="elevation", offset=28, datatype=PointField.FLOAT32, count=1),
                PointField(name="distance", offset=32, datatype=PointField.FLOAT32, count=1),
                PointField(name="time", offset=36, datatype=PointField.UINT32, count=1),
            ]

            # 7) Create and publish the cloud
            enriched_msg = pcl2.create_cloud(msg.header, fields, enriched)
            enriched_msg.point_step = 40  # Must match dtype.itemsize
            enriched_msg.is_dense = False # Required for Autoware
            self.pub.publish(enriched_msg)

        except Exception as e:
            self.get_logger().error(f"Enricher failed: {str(e)}")
    '''        
    
    
def main(args=None):
    rclpy.init(args=args)
    node = CarlaLidarEnricher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
