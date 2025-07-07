#!/usr/bin/env python3
'''
import rclpy
from rclpy.node import Node

import carla
from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.srv import SpawnObject, DestroyObject
from geometry_msgs.msg import Pose, Point, Quaternion
from diagnostic_msgs.msg import KeyValue


class LidarSpawner(Node):
    def __init__(self):
        super().__init__('spawn_lidar')
        # parameters (declare or get)
        self.declare_parameter('host', 'host.docker.internal')
        self.declare_parameter('port', 2000)
        self.declare_parameter('ego_vehicle_role_name', 'ego_vehicle')
        self.declare_parameter('lidar_range', 50.0)
        self.declare_parameter('lidar_rotation_frequency', 40.0)
        self.declare_parameter('lidar_channels', 64)
        self.declare_parameter('lidar_points_per_second', 2000000)
        self.declare_parameter('lidar_upper_fov', 10.0)
        self.declare_parameter('lidar_lower_fov', -30.0)
        self.declare_parameter('lidar_z', 1.7)

        self.spawn_cli = self.create_client(SpawnObject, '/carla/spawn_object')
        self.destroy_cli = self.create_client(DestroyObject, '/carla/destroy_object')
        self.current_lidar_id = None
        self.current_vehicle_id = None
        
        

        # wait for service availability
        while not self.spawn_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /carla/spawn_object service...')
        while not self.destroy_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /carla/destroy_object service...')

        # subscribe to ego info
        self.create_subscription(
            CarlaEgoVehicleInfo,
            '/carla/ego_vehicle/vehicle_info',
            self.ego_info_cb,
            1)

        self.get_logger().info('SpawnLidar node started, waiting for ego vehicle info.')

    def ego_info_cb(self, msg: CarlaEgoVehicleInfo):
        vid = msg.id
        # if a new ego vehicle appears (or changes),
        # spawn a fresh LiDAR attached to it
        if vid != self.current_vehicle_id:
            self.current_vehicle_id = vid
            self.spawn_lidar(vid)

    def spawn_lidar(self, vehicle_id: int):
        # destroy previous LiDAR if any
        if self.current_lidar_id is not None:
            req = DestroyObject.Request()
            req.id = self.current_lidar_id
            self.destroy_cli.call_async(req)
            self.get_logger().info(f'Destroyed old LiDAR id={self.current_lidar_id}')

        # build spawn request
        req = SpawnObject.Request()
        req.type = 'sensor.lidar.ray_cast'
        req.transform = Pose(
            position=Point(x=0.0, y=0.0, z=self.get_parameter('lidar_z').value),
            orientation=Quaternion(w=1.0))
        req.attach_to = vehicle_id
        req.attributes = [
            KeyValue(key='role_name', value='lidar'),
            KeyValue(key='range', value=str(self.get_parameter('lidar_range').value)),
            KeyValue(key='rotation_frequency', value=str(self.get_parameter('lidar_rotation_frequency').value)),
            KeyValue(key='channels', value=str(self.get_parameter('lidar_channels').value)),
            KeyValue(key='points_per_second', value=str(self.get_parameter('lidar_points_per_second').value)),
            KeyValue(key='sensor_tick', value='0.025'),

        ]

        # call spawn service
        future = self.spawn_cli.call_async(req)
        future.add_done_callback(self.spawn_response)

    def spawn_response(self, future):
        resp = future.result()
        if resp.id < 0:
            self.get_logger().error(f'LiDAR spawn failed: {resp.error_string}')
        else:
            self.current_lidar_id = resp.id
            self.get_logger().info(f'Spawned LiDAR id={resp.id} on ego={self.current_vehicle_id}')
            
            
            try:
                # Reconnect to CARLA manually to query the spawned actor
                client = carla.Client(
                    self.get_parameter('host').get_parameter_value().string_value,
                    self.get_parameter('port').get_parameter_value().integer_value
                )
                client.set_timeout(2.0)
                world = client.get_world()

                sensor = world.get_actor(resp.id)
                self.get_logger().info('✅ Confirmed Sensor Attributes from CARLA:')
                for attr in sensor.attributes:
                    self.get_logger().info(f'  {attr}: {sensor.attributes[attr]}')
            except Exception as e:
                self.get_logger().warn(f'⚠ Could not read attributes from CARLA: {e}')
            
        


def main(args=None):
    rclpy.init(args=args)
    node = LidarSpawner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
'''
#!/usr/bin/env python3
"""
This ROS2 node for CARLA:
  1) Spawns a LiDAR sensor on the ego vehicle
  2) Spawns a depth camera on the ego vehicle
  3) Runs YOLO on each depth image to detect obstacles
  4) Publishes the YOLO detections as vision_msgs/Detection2DArray
  5) Publishes static TF transforms for base_link→lidar_link and base_link→camera_link

You need to have:
 - CARLA ROS2 bridge running
 - YOLOv8 installed (pip install ultralytics)
 - cv_bridge installed

Usage:
  ros2 run <package> sensor_fusion_and_detection.py

"""

'''
import rclpy
from rclpy.node import Node

# CARLA bridge messages & services
from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.srv import SpawnObject, DestroyObject
from geometry_msgs.msg import Pose, Point, Quaternion
from diagnostic_msgs.msg import KeyValue

# ROS message types
from sensor_msgs.msg import Image as RosImage
#from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesis, Pose2D
#from vision_msgs.msg import Point2D  # Add this import at top of file
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    BoundingBox2D,
    ObjectHypothesisWithPose,
    Pose2D,
    Point2D
)
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion


#from vision_msgs.msg import Pose2D as InternalPose2D
#from vision_msgs.msg._pose2_d import Pose2D as VisionPose2D

# CV bridge & YOLO
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image as PILImage

#from geometry_msgs.msg import Pose2D
#from vision_msgs.msg     import BoundingBox2D


# TF2 for static transforms
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler

class SensorFusionAndDetection(Node):
    def __init__(self):
        super().__init__('sensor_fusion_and_detection')

        # --- 1) Declare parameters ------------------------------------
        self.declare_parameter('host', 'host.docker.internal')
        self.declare_parameter('port', 2000)
        self.declare_parameter('ego_role', 'ego_vehicle')
        # LiDAR params
        self.declare_parameter('lidar_z',     1.7)
        self.declare_parameter('lidar_rotation_frequency', 40.0) #rotation was 40
        self.declare_parameter('lidar_channels', 64)
        self.declare_parameter('lidar_points_per_second', 300000)
        self.declare_parameter('lidar_range', 50.0)
        self.declare_parameter('lidar_sensor_tick', 0.025) #was 0.025
        # Depth camera params (offset from base_link)
        self.declare_parameter('cam_offset_x', 2.4)
        self.declare_parameter('cam_offset_y', 0.0)
        self.declare_parameter('cam_offset_z', 1.5)
        self.declare_parameter('cam_image_width', 640)
        self.declare_parameter('cam_image_height', 480)
        self.declare_parameter('cam_fov', 90.0)
        self.declare_parameter('cam_sensor_tick', 0.025) #was 0.025
        # Output topic
        self.declare_parameter('vision_topic', '/vision_detections')

        # --- 2) Internal state & YOLO model ----------------------------
        self.current_vehicle_id = None
        self.current_lidar_id = None
        self.current_cam_id   = None

        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8m.pt')  # load pretrained YOLOv8-nano
        self.yolo_labels = self.yolo.names

        # --- 3) CARLA spawn service clients ----------------------------
        self.spawn_cli_lidar = self.create_client(SpawnObject, '/carla/spawn_object')
        self.spawn_cli_cam   = self.create_client(SpawnObject, '/carla/spawn_object')
        self.destroy_cli     = self.create_client(DestroyObject, '/carla/destroy_object')
        for cli in (self.spawn_cli_lidar, self.spawn_cli_cam, self.destroy_cli):
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for service {cli.srv_name}...')

        # --- 4) TF broadcaster ------------------------------------------
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transforms()

        # --- 5) ROS 2 Subscriptions & Publisher ------------------------
        self.create_subscription(
            CarlaEgoVehicleInfo,
            '/carla/ego_vehicle/vehicle_info',
            self.ego_info_cb,
            1)

        self.depth_sub = None
        self.vision_pub = self.create_publisher(
            Detection2DArray,
            self.get_parameter('vision_topic').value,
            10)

        self.get_logger().info('SensorFusionAndDetection node initialized.')

    def publish_static_transforms(self):
        """Publish static TFs: base_link→lidar_link and base_link→camera_link"""
        transforms = []
        # base_link -> lidar_link
        t_lidar = TransformStamped()
        t_lidar.header.stamp = self.get_clock().now().to_msg()
        t_lidar.header.frame_id = 'base_link'
        t_lidar.child_frame_id  = 'lidar_link' #ego_vehicle/lidar
        t_lidar.transform.translation.x = 0.0
        t_lidar.transform.translation.y = 0.0
        t_lidar.transform.translation.z = self.get_parameter('lidar_z').value
        q = quaternion_from_euler(0, 0, 0)
        t_lidar.transform.rotation.x = q[0]
        t_lidar.transform.rotation.y = q[1]
        t_lidar.transform.rotation.z = q[2]
        t_lidar.transform.rotation.w = q[3]
        transforms.append(t_lidar)

        # base_link -> camera_link
        t_cam = TransformStamped()
        t_cam.header.stamp = self.get_clock().now().to_msg()
        t_cam.header.frame_id = 'base_link'
        t_cam.child_frame_id  = 'camera_link'
        t_cam.transform.translation.x = self.get_parameter('cam_offset_x').value
        t_cam.transform.translation.y = self.get_parameter('cam_offset_y').value
        t_cam.transform.translation.z = self.get_parameter('cam_offset_z').value
        q2 = quaternion_from_euler(0, 0, 0)
        t_cam.transform.rotation.x = q2[0]
        t_cam.transform.rotation.y = q2[1]
        t_cam.transform.rotation.z = q2[2]
        t_cam.transform.rotation.w = q2[3]
        transforms.append(t_cam)

        self.tf_broadcaster.sendTransform(transforms)
        self.get_logger().info('Published static TF transforms: base_link->lidar_link, base_link->camera_link')

    def ego_info_cb(self, msg: CarlaEgoVehicleInfo):
        vid = msg.id
        if vid != self.current_vehicle_id:
            self.current_vehicle_id = vid
            self.spawn_lidar(vid)
            self.spawn_depth_camera(vid)

    def spawn_lidar(self, vehicle_id: int):
        if self.current_lidar_id is not None:
            req = DestroyObject.Request(); req.id = self.current_lidar_id
            self.destroy_cli.call_async(req)
        req = SpawnObject.Request()
        req.type = 'sensor.lidar.ray_cast'
        req.attach_to = vehicle_id
        req.transform = Pose(
            position=Point(x=0.0, y=0.0, z=1.7),
            orientation=Quaternion(w=1.0)
        )
        attrs = [
            KeyValue(key='role_name', value='lidar'),
            KeyValue(key='range', value=str(self.get_parameter('lidar_range').value)),
            KeyValue(key='rotation_frequency', value=str(self.get_parameter('lidar_rotation_frequency').value)),
            KeyValue(key='channels', value=str(self.get_parameter('lidar_channels').value)),
            KeyValue(key='points_per_second', value=str(self.get_parameter('lidar_points_per_second').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('lidar_sensor_tick').value)),
        ]
        req.attributes = attrs
        self.spawn_cli_lidar.call_async(req).add_done_callback(self.lidar_spawn_resp)

    def lidar_spawn_resp(self, future):
        resp = future.result()
        if resp.id < 0:
            self.get_logger().error('LiDAR spawn failed: ' + resp.error_string)
        else:
            self.current_lidar_id = resp.id
            self.get_logger().info(f'Spawned LiDAR id={resp.id}')

    def spawn_depth_camera(self, vehicle_id: int):
        if self.current_cam_id is not None:
            req = DestroyObject.Request(); req.id = self.current_cam_id
            self.destroy_cli.call_async(req)
        req = SpawnObject.Request()
        req.type = 'sensor.camera.depth'
        req.attach_to = vehicle_id
        req.transform = Pose(
            position=Point(
                x=self.get_parameter('cam_offset_x').value,
                y=self.get_parameter('cam_offset_y').value,
                z=self.get_parameter('cam_offset_z').value
            ), orientation=Quaternion(w=1.0)
        )
        req.attributes = [
            KeyValue(key='role_name', value='depth_camera'),
            KeyValue(key='image_size_x', value=str(self.get_parameter('cam_image_width').value)),
            KeyValue(key='image_size_y', value=str(self.get_parameter('cam_image_height').value)),
            KeyValue(key='fov', value=str(self.get_parameter('cam_fov').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('cam_sensor_tick').value)),
        ]
        self.spawn_cli_cam.call_async(req).add_done_callback(self.cam_spawn_resp)

    def cam_spawn_resp(self, future):
        resp = future.result()
        if resp.id < 0:
            self.get_logger().error('Depth camera spawn failed: ' + resp.error_string)
        else:
            self.current_cam_id = resp.id
            self.get_logger().info(f'Spawned Depth Camera id={resp.id}')
            topic = f'/carla/ego_vehicle/depth_camera/image'
            self.depth_sub = self.create_subscription(
                RosImage, topic, self.depth_image_cb, 10)
            self.get_logger().info('Subscribed to ' + topic)

    def depth_image_cb(self, msg: RosImage):
        try:
            # 1) Convert to CV image
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 2) To float32 and remove NaNs
            img = np.nan_to_num(cv_img.astype('float32'))

            # 3) Normalize to 0–255 uint8
            img_max = np.nanmax(img)
            if img_max == 0:
                self.get_logger().warn("Depth image max is 0. Skipping frame.")
                return
            img_norm = (img / img_max * 255).clip(0,255).astype(np.uint8)

            # ─── From here on, use the img_norm you just defined ─────────

            # 4) Make it contiguous uint8
            gray = np.ascontiguousarray(img_norm).astype(np.uint8)

            # 5) Stack into H×W×3 and enforce contiguity
            rgb_np = np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))

            # 6) Run YOLO on the NumPy array
            results = self.yolo.predict(source=rgb_np, verbose=False)

            # 7) Unpack and publish
            det2d_arr = Detection2DArray(header=msg.header)
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    conf = float(box.conf.cpu().numpy())
                    cls  = int(box.cls.cpu().numpy())
                    label = self.yolo_labels[cls]
                    if label not in ('car', 'truck'):
                        continue

                    d = Detection2D()

                    # Create position point first
                    position = Point2D()
                    position.x = float((xyxy[0] + xyxy[2]) / 2.0)
                    position.y = float((xyxy[1] + xyxy[3]) / 2.0)

                    # Create pose
                    pose = Pose2D()
                    pose.position = position
                    pose.theta = 0.0

                    # Create bounding box
                    bb = BoundingBox2D()
                    bb.center = pose
                    bb.size_x = float(xyxy[2] - xyxy[0])
                    bb.size_y = float(xyxy[3] - xyxy[1])

                    # Create hypothesis with pose
                    hyp_with_pose = ObjectHypothesisWithPose()
                    
                    # Set basic hypothesis
                    hyp_with_pose.hypothesis.class_id = label
                    hyp_with_pose.hypothesis.score = conf
                    
                    # Create full pose with covariance
                    pose_with_cov = PoseWithCovariance()
                    pose_with_cov.pose.position.x = position.x
                    pose_with_cov.pose.position.y = position.y
                    pose_with_cov.pose.position.z = 0.0  # 2D detection
                    pose_with_cov.pose.orientation.w = 1.0  # No rotation
                    pose_with_cov.covariance = [0.0] * 36  # Initialize covariance
                    
                    hyp_with_pose.pose = pose_with_cov

                    # Create detection
                    d = Detection2D()
                    d.bbox = bb
                    d.results.append(hyp_with_pose)  # Note: using hyp_with_pose now

                    det2d_arr.detections.append(d)
            self.vision_pub.publish(det2d_arr)

        except Exception as e:
            self.get_logger().error(f"Error in depth_image_cb: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionAndDetection()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
'''
'''
import rclpy
from rclpy.node import Node

from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.srv import SpawnObject, DestroyObject
from geometry_msgs.msg import Pose, Point, Quaternion
from diagnostic_msgs.msg import KeyValue

from sensor_msgs.msg import Image as RosImage
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    BoundingBox2D,
    ObjectHypothesisWithPose,
    Pose2D,
    Point2D
)
from geometry_msgs.msg import PoseWithCovariance

from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image as PILImage

import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler

class SensorFusionAndDetection(Node):
    def __init__(self):
        super().__init__('sensor_fusion_and_detection')

        self.declare_parameter('host', 'host.docker.internal')
        self.declare_parameter('port', 2000)
        self.declare_parameter('ego_role', 'ego_vehicle')
        self.declare_parameter('lidar_z', 1.7)
        self.declare_parameter('lidar_rotation_frequency', 40.0)
        self.declare_parameter('lidar_channels', 64)
        self.declare_parameter('lidar_points_per_second', 300000)
        self.declare_parameter('lidar_range', 50.0)
        self.declare_parameter('lidar_sensor_tick', 0.025)
        self.declare_parameter('cam_offset_x', 2.4)
        self.declare_parameter('cam_offset_y', 0.0)
        self.declare_parameter('cam_offset_z', 1.5)
        self.declare_parameter('cam_image_width', 640)
        self.declare_parameter('cam_image_height', 480)
        self.declare_parameter('cam_fov', 90.0)
        self.declare_parameter('cam_sensor_tick', 0.025)
        self.declare_parameter('vision_topic', '/vision_detections')

        self.current_vehicle_id = None
        self.current_lidar_id = None
        self.current_cam_id = None
        self.current_rgb_id = None

        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8m.pt')
        self.yolo_labels = self.yolo.names

        self.spawn_cli_lidar = self.create_client(SpawnObject, '/carla/spawn_object')
        self.spawn_cli_cam = self.create_client(SpawnObject, '/carla/spawn_object')
        self.destroy_cli = self.create_client(DestroyObject, '/carla/destroy_object')
        for cli in (self.spawn_cli_lidar, self.spawn_cli_cam, self.destroy_cli):
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for service {cli.srv_name}...')

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transforms()

        self.create_subscription(
            CarlaEgoVehicleInfo,
            '/carla/ego_vehicle/vehicle_info',
            self.ego_info_cb,
            1)

        self.depth_sub = None
        self.rgb_sub = None
        self.latest_depth = None
        self.latest_rgb = None

        self.vision_pub = self.create_publisher(
            Detection2DArray,
            self.get_parameter('vision_topic').value,
            10)

        self.get_logger().info('SensorFusionAndDetection node initialized.')

    def publish_static_transforms(self):
        transforms = []
        t_lidar = TransformStamped()
        t_lidar.header.stamp = self.get_clock().now().to_msg()
        t_lidar.header.frame_id = 'base_link'
        t_lidar.child_frame_id = 'lidar_link'
        t_lidar.transform.translation.z = self.get_parameter('lidar_z').value
        q = quaternion_from_euler(0, 0, 0)
        t_lidar.transform.rotation.x, t_lidar.transform.rotation.y, t_lidar.transform.rotation.z, t_lidar.transform.rotation.w = q
        transforms.append(t_lidar)

        t_cam = TransformStamped()
        t_cam.header.stamp = self.get_clock().now().to_msg()
        t_cam.header.frame_id = 'base_link'
        t_cam.child_frame_id = 'camera_link'
        t_cam.transform.translation.x = self.get_parameter('cam_offset_x').value
        t_cam.transform.translation.y = self.get_parameter('cam_offset_y').value
        t_cam.transform.translation.z = self.get_parameter('cam_offset_z').value
        q2 = quaternion_from_euler(0, 0, 0)
        t_cam.transform.rotation.x, t_cam.transform.rotation.y, t_cam.transform.rotation.z, t_cam.transform.rotation.w = q2
        transforms.append(t_cam)

        self.tf_broadcaster.sendTransform(transforms)

    def ego_info_cb(self, msg):
        vid = msg.id
        if vid != self.current_vehicle_id:
            self.current_vehicle_id = vid
            self.spawn_lidar(vid)
            self.spawn_depth_camera(vid)
            self.spawn_rgb_camera(vid)

    def spawn_lidar(self, vehicle_id):
        if self.current_lidar_id:
            req = DestroyObject.Request(); req.id = self.current_lidar_id
            self.destroy_cli.call_async(req)
        req = SpawnObject.Request()
        req.type = 'sensor.lidar.ray_cast'
        req.attach_to = vehicle_id
        req.transform = Pose(position=Point(z=1.7), orientation=Quaternion(w=1.0))
        req.attributes = [
            KeyValue(key='role_name', value='lidar'),
            KeyValue(key='range', value=str(self.get_parameter('lidar_range').value)),
            KeyValue(key='rotation_frequency', value=str(self.get_parameter('lidar_rotation_frequency').value)),
            KeyValue(key='channels', value=str(self.get_parameter('lidar_channels').value)),
            KeyValue(key='points_per_second', value=str(self.get_parameter('lidar_points_per_second').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('lidar_sensor_tick').value))
        ]
        self.spawn_cli_lidar.call_async(req).add_done_callback(self.lidar_spawn_resp)

    def spawn_depth_camera(self, vehicle_id):
        if self.current_cam_id:
            req = DestroyObject.Request(); req.id = self.current_cam_id
            self.destroy_cli.call_async(req)
        req = SpawnObject.Request()
        req.type = 'sensor.camera.depth'
        req.attach_to = vehicle_id
        req.transform = Pose(position=Point(
            x=self.get_parameter('cam_offset_x').value,
            y=self.get_parameter('cam_offset_y').value,
            z=self.get_parameter('cam_offset_z').value),
            orientation=Quaternion(w=1.0))
        req.attributes = [
            KeyValue(key='role_name', value='depth_camera'),
            KeyValue(key='image_size_x', value=str(self.get_parameter('cam_image_width').value)),
            KeyValue(key='image_size_y', value=str(self.get_parameter('cam_image_height').value)),
            KeyValue(key='fov', value=str(self.get_parameter('cam_fov').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('cam_sensor_tick').value))
        ]
        self.spawn_cli_cam.call_async(req).add_done_callback(self.cam_spawn_resp)

    def spawn_rgb_camera(self, vehicle_id):
        if self.current_rgb_id:
            req = DestroyObject.Request(); req.id = self.current_rgb_id
            self.destroy_cli.call_async(req)
        req = SpawnObject.Request()
        req.type = 'sensor.camera.rgb'
        req.attach_to = vehicle_id
        req.transform = Pose(position=Point(
            x=self.get_parameter('cam_offset_x').value,
            y=self.get_parameter('cam_offset_y').value,
            z=self.get_parameter('cam_offset_z').value),
            orientation=Quaternion(w=1.0))
        req.attributes = [
            KeyValue(key='role_name', value='rgb_camera'),
            KeyValue(key='image_size_x', value=str(self.get_parameter('cam_image_width').value)),
            KeyValue(key='image_size_y', value=str(self.get_parameter('cam_image_height').value)),
            KeyValue(key='fov', value=str(self.get_parameter('cam_fov').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('cam_sensor_tick').value))
        ]
        self.spawn_cli_cam.call_async(req).add_done_callback(self.rgb_cam_resp)

    def lidar_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_lidar_id = resp.id

    def cam_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_cam_id = resp.id
            topic = '/carla/ego_vehicle/depth_camera/image'
            self.depth_sub = self.create_subscription(RosImage, topic, self.depth_cb, 10)

    def rgb_cam_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_rgb_id = resp.id
            topic = '/carla/ego_vehicle/rgb_camera/image'
            self.rgb_sub = self.create_subscription(RosImage, topic, self.rgb_cb, 10)

    def depth_cb(self, msg):
        self.latest_depth = msg
        self.try_infer()

    def rgb_cb(self, msg):
        self.latest_rgb = msg
        self.try_infer()

    def try_infer(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding='rgb8')
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
            img = PILImage.fromarray(rgb)
            results = self.yolo(img)

            det2d_arr = Detection2DArray(header=self.latest_rgb.header)
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    conf = float(box.conf.cpu().numpy())
                    cls = int(box.cls.cpu().numpy())
                    label = self.yolo_labels[cls]
                    if label not in ('car', 'truck'):
                        continue

                    cx = int((xyxy[0] + xyxy[2]) / 2.0)
                    cy = int((xyxy[1] + xyxy[3]) / 2.0)
                    z_val = float(depth[cy, cx]) if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1] else 0.0

                    position = Point2D()
                    position.x = float(cx)
                    position.y = float(cy)

                    pose2d = Pose2D()
                    pose2d.position = position
                    pose2d.theta = 0.0

                    bb = BoundingBox2D()
                    bb.center = pose2d
                    bb.size_x = float(xyxy[2] - xyxy[0])
                    bb.size_y = float(xyxy[3] - xyxy[1])

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = label
                    hyp.hypothesis.score = conf

                    pose_cov = PoseWithCovariance()
                    pose_cov.pose.position.x = position.x
                    pose_cov.pose.position.y = position.y
                    pose_cov.pose.position.z = z_val
                    pose_cov.pose.orientation.w = 1.0
                    pose_cov.covariance = [0.0] * 36
                    hyp.pose = pose_cov

                    d = Detection2D()
                    d.bbox = bb
                    d.results.append(hyp)
                    det2d_arr.detections.append(d)

            self.vision_pub.publish(det2d_arr)
        except Exception as e:
            self.get_logger().error(f"Error in try_infer: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionAndDetection()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
'''


import rclpy
from rclpy.node import Node

from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.srv import SpawnObject, DestroyObject
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped
from diagnostic_msgs.msg import KeyValue

from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    BoundingBox2D,
    ObjectHypothesisWithPose,
    Pose2D,
    Point2D
)
from geometry_msgs.msg import PoseWithCovariance
#from rclpy.duration import Duration

from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image as PILImage

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler


from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import math
from tf_transformations import quaternion_from_euler


class SensorFusionAndDetection(Node):
    def __init__(self):
        super().__init__('sensor_fusion_and_detection')

        self.declare_parameter('host', 'host.docker.internal')
        self.declare_parameter('port', 2000)
        self.declare_parameter('ego_role', 'ego_vehicle')
        self.declare_parameter('lidar_z', 1.7)
        self.declare_parameter('lidar_rotation_frequency', 40.0)
        self.declare_parameter('lidar_channels', 64)
        self.declare_parameter('lidar_points_per_second', 700000)
        self.declare_parameter('lidar_range', 50.0)
        self.declare_parameter('lidar_sensor_tick', 0.025)
        self.declare_parameter('cam_offset_x', 2.4)
        self.declare_parameter('cam_offset_y', 0.0)
        self.declare_parameter('cam_offset_z', 1.5)
        self.declare_parameter('cam_image_width', 640)
        self.declare_parameter('cam_image_height', 480)
        self.declare_parameter('cam_fov', 120.0)
        self.declare_parameter('cam_sensor_tick', 0.025)
        self.declare_parameter('vision_topic', '/vision_detections')

        self.current_vehicle_id = None
        self.current_lidar_id = None
        self.current_cam_id = None
        self.current_rgb_id = None
        
        self.current_odom_id = None
        
        
        
        self.destroying_lidar = False
        self.destroying_depth = False
        self.destroying_rgb = False


        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8m.pt')
        self.yolo_labels = self.yolo.names

        self.spawn_cli_lidar = self.create_client(SpawnObject, '/carla/spawn_object')
        self.spawn_cli_cam = self.create_client(SpawnObject, '/carla/spawn_object')
        self.destroy_cli = self.create_client(DestroyObject, '/carla/destroy_object')
        #self.spawn_cli_odom  = self.create_client(SpawnObject, '/carla/spawn_object')  # <-- add this

        for cli in (self.spawn_cli_lidar, self.spawn_cli_cam,  self.destroy_cli):
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for service {cli.srv_name}...')

        #self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0)) #tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        #self.publish_static_transforms()
        
        # broadcaster for dynamic map→base_link

        self.tf_dyn      = TransformBroadcaster(self)
        self.tf_stat     = StaticTransformBroadcaster(self)
        #self._publish_static_mounts()
        #self.create_timer(0.02, self._broadcast_base_link)

        # Subscribers
        #ego_topic = f'/carla/{self.get_parameter("ego_role").value}/vehicle_info'
        #self.create_subscription(CarlaEgoVehicleInfo, ego_topic, self.ego_cb, 1)
        

        self.create_subscription(
            CarlaEgoVehicleInfo,
            '/carla/ego_vehicle/vehicle_info',
            self.ego_info_cb,
            1)

        self.depth_sub = None
        self.rgb_sub = None
        self.latest_depth = None
        self.latest_rgb = None

        self.vision_pub = self.create_publisher(
            Detection2DArray,
            self.get_parameter('vision_topic').value,
            10)

        self.get_logger().info('SensorFusionAndDetection node initialized.')

    ''' ###
        now = self.get_clock().now().to_msg()

        # 1) base_link -> ego_vehicle/lidar
        t0 = TransformStamped()
        t0.header.stamp    = now
        t0.header.frame_id = 'base_link'
        t0.child_frame_id  = 'ego_vehicle/lidar'
        t0.transform.translation.x = 0.0
        t0.transform.translation.y = 0.0
        t0.transform.translation.z = self.get_parameter('lidar_z').value
        t0.transform.rotation.w    = 1.0

        # 2) base_link -> ego_vehicle/rgb_camera
        t1 = TransformStamped()
        t1.header.stamp    = now
        t1.header.frame_id = 'base_link'
        t1.child_frame_id  = 'ego_vehicle/rgb_camera'
        t1.transform.translation.x = self.get_parameter('cam_offset_x').value
        t1.transform.translation.y = self.get_parameter('cam_offset_y').value
        t1.transform.translation.z = self.get_parameter('cam_offset_z').value
        t1.transform.rotation.w    = 1.0

        self.tf_stat.sendTransform((t0, t1))
    ''' ###
    ''' ####

    def publish_static_transforms(self):
        transforms = []

        
        # 1. map -> ego_vehicle (static root)
        t_map = TransformStamped()
        t_map.header.stamp = self.get_clock().now().to_msg()
        t_map.header.frame_id = 'map'
        t_map.child_frame_id = 'ego_vehicle'
        t_map.transform.translation.x = 0.0
        t_map.transform.translation.y = 0.0
        t_map.transform.translation.z = 0.0
        t_map.transform.rotation.w = 1.0
        transforms.append(t_map)

        # 2. ego_vehicle -> ego_vehicle/lidar
        t_lidar = TransformStamped()
        t_lidar.header.stamp = self.get_clock().now().to_msg()
        t_lidar.header.frame_id = 'ego_vehicle'
        t_lidar.child_frame_id = 'ego_vehicle/lidar'
        t_lidar.transform.translation.x = 0.0
        t_lidar.transform.translation.y = 0.0
        t_lidar.transform.translation.z = self.get_parameter('lidar_z').value
        q = quaternion_from_euler(0, 0, 0)
        t_lidar.transform.rotation.x, t_lidar.transform.rotation.y, t_lidar.transform.rotation.z, t_lidar.transform.rotation.w = q
        transforms.append(t_lidar)
        
        # 3. ego_vehicle -> ego_vehicle/rgb_camera
        t_rgb = TransformStamped()
        t_rgb.header.stamp = self.get_clock().now().to_msg()
        t_rgb.header.frame_id = 'ego_vehicle/lidar'
        t_rgb.child_frame_id = 'ego_vehicle/rgb_camera'
        t_rgb.transform.translation.x = self.get_parameter('cam_offset_x').value
        t_rgb.transform.translation.y = self.get_parameter('cam_offset_y').value
        t_rgb.transform.translation.z = self.get_parameter('cam_offset_z').value
        t_rgb.transform.rotation.x, t_rgb.transform.rotation.y, t_rgb.transform.rotation.z, t_rgb.transform.rotation.w = quaternion_from_euler(0, 0, 0)
        transforms.append(t_rgb)

        # 4. ego_vehicle -> ego_vehicle/depth_camera
        t_depth = TransformStamped()
        t_depth.header.stamp = self.get_clock().now().to_msg()
        t_depth.header.frame_id = 'ego_vehicle/lidar'
        t_depth.child_frame_id = 'ego_vehicle/depth_camera'
        t_depth.transform.translation.x = self.get_parameter('cam_offset_x').value
        t_depth.transform.translation.y = self.get_parameter('cam_offset_y').value
        t_depth.transform.translation.z = self.get_parameter('cam_offset_z').value
        t_depth.transform.rotation.x, t_depth.transform.rotation.y, t_depth.transform.rotation.z, t_depth.transform.rotation.w = quaternion_from_euler(0, 0, 0)
        transforms.append(t_depth)

        self.tf_broadcaster.sendTransform(transforms)
    ''' ######
    ## this new
 
    def _broadcast_base_link(self):
        try:
            t_l = self.tf_buffer.lookup_transform('map', 'ego_vehicle/lidar', rclpy.time.Time())
            t = TransformStamped()
            t.header.stamp    = t_l.header.stamp
            t.header.frame_id = 'map'
            t.child_frame_id  = 'base_link'
            t.transform       = t_l.transform
            self.tf_dyn.sendTransform(t)
        except Exception as e:
            self.get_logger().warn(f'base_link broadcast failed: {e}')
            
    def ego_cb(self, msg: CarlaEgoVehicleInfo):
        vid = msg.id
        if vid != self.current_vehicle_id:
            self.current_vehicle_id = vid
            # spawn lidar and cameras
            self._spawn_sensor('sensor.lidar.ray_cast', 'lidar', Point(z=self.get_parameter('lidar_z').value))
            self._spawn_sensor('sensor.camera.rgb', 'rgb_camera',
                               Point(x=self.get_parameter('cam_offset_x').value,
                                     y=self.get_parameter('cam_offset_y').value,
                                     z=self.get_parameter('cam_offset_z').value)) 
            self.spawn_sensor('sensor.pseudo.odom', 'odometry', point(z=self.get_parameter('lidar_z').value) )

    ## until here/ uncomment next part
    def ego_info_cb(self, msg):   
        vid = msg.id
        if vid != self.current_vehicle_id:
            self.current_vehicle_id = vid
            self.spawn_lidar(vid)
            self.spawn_depth_camera(vid)
            self.spawn_rgb_camera(vid)
            #self.spawn_odometry(vid) 

    def spawn_lidar(self, vehicle_id):
        
        if self.current_lidar_id:
            req = DestroyObject.Request(); req.id = self.current_lidar_id
            self.current_lidar_id = None
            self.destroy_cli.call_async(req).add_done_callback(lambda future: self.actually_spawn_lidar(vehicle_id))
        else:
            self.actually_spawn_lidar(vehicle_id)
    
    '''      
            
    def spawn_odometry(self, vehicle_id):
        
        if self.current_odom_id:
            req = DestroyObject.Request(); req.id = self.current_odom_id
            self.current_odom_id = None
            self.destroy_cli.call_async(req).add_done_callback(lambda future: self.actually_spawn_odometry(vehicle_id))
        else:
            self.actually_spawn_odometry(vehicle_id)
          
    def actually_spawn_odometry(self, vehicle_id):
        #req = SpawnObject.Request()
        req = SpawnObject.Request()
        req.id        = 'odometry'
        req.type      = 'sensor.pseudo.odom'
        
        req.attach_to = vehicle_id
        # no transform needed for odom pseudo-sensor
        req.transform = Pose(position=Point(z=1.7), orientation=Quaternion(w=1.0))

        #req.transform.orientation.w = 1.0
        req.attributes = [
            KeyValue(key='role_name', value='odometry')
        ]
        self.spawn_cli_odom.call_async(req).add_done_callback(self.odom_spawn_resp)
        
    def odom_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_odom_id = resp.id
    '''




    '''#####
        if self.current_lidar_id and not self.destroying_lidar:
            self.destroying_lidar = True
            req = DestroyObject.Request()
            req.id = self.current_lidar_id
            self.current_lidar_id = None
            self.destroy_cli.call_async(req).add_done_callback(
                lambda future: self.after_lidar_destroyed(vehicle_id)
            )
        elif not self.destroying_lidar:
            self.actually_spawn_lidar(vehicle_id)
        
    def after_lidar_destroyed(self, vehicle_id):
        self.destroying_lidar = False
        self.actually_spawn_lidar(vehicle_id)
 
    ''' ####

    def actually_spawn_lidar(self, vehicle_id):
        req = SpawnObject.Request()
        req.type = 'sensor.lidar.ray_cast'
        req.attach_to = vehicle_id
        req.transform = Pose(position=Point(z=1.7), orientation=Quaternion(w=1.0))
        req.attributes = [
            KeyValue(key='role_name', value='lidar'),
            KeyValue(key='range', value=str(self.get_parameter('lidar_range').value)),
            KeyValue(key='rotation_frequency', value=str(self.get_parameter('lidar_rotation_frequency').value)),
            KeyValue(key='channels', value=str(self.get_parameter('lidar_channels').value)),
            KeyValue(key='points_per_second', value=str(self.get_parameter('lidar_points_per_second').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('lidar_sensor_tick').value))
        ]
        self.spawn_cli_lidar.call_async(req).add_done_callback(self.lidar_spawn_resp)

    def spawn_depth_camera(self, vehicle_id):
        if self.current_cam_id:
            req = DestroyObject.Request(); req.id = self.current_cam_id
            self.current_cam_id = None
            self.destroy_cli.call_async(req).add_done_callback(lambda future: self.actually_spawn_depth_camera(vehicle_id))
        else:
            self.actually_spawn_depth_camera(vehicle_id)
         
    def actually_spawn_depth_camera(self, vehicle_id):     
        self.latest_depth = None    
       
        req = SpawnObject.Request()
        req.type = 'sensor.camera.depth'
        req.attach_to = vehicle_id
        req.transform = Pose(position=Point(
            x=self.get_parameter('cam_offset_x').value,
            y=self.get_parameter('cam_offset_y').value,
            z=self.get_parameter('cam_offset_z').value),
            orientation=Quaternion(w=1.0))
        req.attributes = [
            KeyValue(key='role_name', value='depth_camera'),
            KeyValue(key='image_size_x', value=str(self.get_parameter('cam_image_width').value)),
            KeyValue(key='image_size_y', value=str(self.get_parameter('cam_image_height').value)),
            KeyValue(key='fov', value=str(self.get_parameter('cam_fov').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('cam_sensor_tick').value))
        ]
        self.spawn_cli_cam.call_async(req).add_done_callback(self.cam_spawn_resp)

    def spawn_rgb_camera(self, vehicle_id):
        if self.current_rgb_id:
            req = DestroyObject.Request(); req.id = self.current_rgb_id
            
            self.current_rgb_id = None
            self.destroy_cli.call_async(req).add_done_callback(lambda future: self.actually_spawn_rgb_camera(vehicle_id))
        else:
            self.actually_spawn_rgb_camera(vehicle_id)
        
    def actually_spawn_rgb_camera(self, vehicle_id):
        self.latest_rgb = None
        req = SpawnObject.Request()
        req.type = 'sensor.camera.rgb'
        req.attach_to = vehicle_id
        req.transform = Pose(position=Point(
            x=self.get_parameter('cam_offset_x').value,
            y=self.get_parameter('cam_offset_y').value,
            z=self.get_parameter('cam_offset_z').value),
            orientation=Quaternion(w=1.0))
        req.attributes = [
            KeyValue(key='role_name', value='rgb_camera'),
            KeyValue(key='image_size_x', value=str(self.get_parameter('cam_image_width').value)),
            KeyValue(key='image_size_y', value=str(self.get_parameter('cam_image_height').value)),
            KeyValue(key='fov', value=str(self.get_parameter('cam_fov').value)),
            KeyValue(key='sensor_tick', value=str(self.get_parameter('cam_sensor_tick').value))
        ]
        self.spawn_cli_cam.call_async(req).add_done_callback(self.rgb_cam_resp)

    def lidar_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_lidar_id = resp.id

    def cam_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_cam_id = resp.id
            topic = '/carla/ego_vehicle/depth_camera/image'
            self.depth_sub = self.create_subscription(RosImage, topic, self.depth_cb, 10)

    def rgb_cam_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_rgb_id = resp.id
            topic = '/carla/ego_vehicle/rgb_camera/image'
            self.rgb_sub = self.create_subscription(RosImage, topic, self.rgb_cb, 10)

    def depth_cb(self, msg):
        self.latest_depth = msg
        self.try_infer()

    def rgb_cb(self, msg):
        self.latest_rgb = msg
        self.try_infer()

    '''####
    def camera_to_base_link(self, u, v, depth, width, height, fov, stamp):
        fx = fy = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        cx = width / 2.0
        cy = height / 2.0

        x_c = (u - cx) * depth / fx
        y_c = (v - cy) * depth / fy
        z_c = depth

        pt_cam = PointStamped()
        pt_cam.header.stamp = self.get_clock().now().to_msg()
        pt_cam.header.frame_id = 'ego_vehicle/rgb_camera'
        pt_cam.point.x = x_c
        pt_cam.point.y = y_c
        pt_cam.point.z = z_c

        try:
            #pt_base = self.tf_buffer.transform(pt_cam, 'base_link', timeout=rclpy.duration.Duration(seconds=0.5))
            transform = self.tf_buffer.lookup_transform('ego_vehicle/lidar', pt_cam.header.frame_id, rclpy.time.Time()) #stamp, timeout=rclpy.duration.Duration(seconds=0.5))# rclpy.time.Time())
            pt_base = do_transform_point(pt_cam, transform)

            return pt_base.point
        except Exception as e:
            self.get_logger().error(f"TF transform failed (camera): {e}")
            return None
    '''####
 

    def camera_to_base_link(self, u, v, depth, width, height, fov, stamp):
        # 1) project pixel to camera frame
        fx = fy = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        cx, cy = width / 2.0, height / 2.0
        x_c = (u - cx) * depth / fx
        y_c = (v - cy) * depth / fy
        z_c = depth

        pt_cam = PointStamped()
        pt_cam.header.frame_id = 'ego_vehicle/rgb_camera'
        pt_cam.point.x = x_c
        pt_cam.point.y = y_c
        pt_cam.point.z = z_c

        # 2) try exact‐time transform

        '''####
        try:
            pt_cam.header.stamp = stamp
            transform = self.tf_buffer.lookup_transform(
                'ego_vehicle/lidar',           # target
                pt_cam.header.frame_id,        # source
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.1))
        except (tf2_ros.ExtrapolationException, tf2_ros.LookupException):
            # fallback: use the latest
            self.get_logger().warn(
                f"TF exact lookup failed at {stamp.sec}.{stamp.nanosec}, falling back to latest")
            # make the point stamp match “now”
            now = self.get_clock().now().to_msg()
            pt_cam.header.stamp = now
            transform = self.tf_buffer.lookup_transform(
                'ego_vehicle/lidar',
                pt_cam.header.frame_id,
                rclpy.time.Time(),      # latest available
                timeout=rclpy.duration.Duration(seconds=0.1))
        '''#####
        '''#####
        try:
            if self.tf_buffer.can_transform('ego_vehicle/lidar', pt_cam.header.frame_id, stamp, timeout=rclpy.duration.Duration(seconds=0.1)):
                pt_cam.header.stamp = stamp
                transform = self.tf_buffer.lookup_transform('ego_vehicle/lidar', pt_cam.header.frame_id, stamp)
            else:
                self.get_logger().warn(f"No transform at {stamp.sec}.{stamp.nanosec}, using latest")
                pt_cam.header.stamp = rclpy.time.Time().to_msg()
                transform = self.tf_buffer.lookup_transform('ego_vehicle/lidar', pt_cam.header.frame_id, rclpy.time.Time())

            pt_base = do_transform_point(pt_cam, transform)
            return pt_base.point

        except Exception as e:
            self.get_logger().error(f"TF transform failed (camera): {e}")
            return None
        '''####
      

        try:
            pt_cam.header.stamp = stamp  # original camera image stamp

            # Try to get transform at image time
            if self.tf_buffer.can_transform('ego_vehicle/lidar', pt_cam.header.frame_id, stamp, timeout=rclpy.duration.Duration(seconds=0.1)):
                transform_time = stamp
            else:
                self.get_logger().warn(f"No transform at {stamp.sec}.{stamp.nanosec}, using latest")
                

                transform_time = rclpy.time.Time().to_msg()

            # Actually get transform
            transform = self.tf_buffer.lookup_transform('ego_vehicle/lidar', pt_cam.header.frame_id, transform_time)

            # Set the point's stamp to match the transform time
            pt_cam.header.stamp = transform_time

            # Transform point
            pt_base = do_transform_point(pt_cam, transform)
            return pt_base.point

        except Exception as e:
            self.get_logger().error(f"TF transform failed (camera): {e}")
            return None

        # 3) actually transform the point
        #pt_base = do_transform_point(pt_cam, transform)
        #return pt_base.point




    
    def try_infer(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding='rgb8')
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
            img = PILImage.fromarray(rgb)
            results = self.yolo(img)

            width = rgb.shape[1]
            height = rgb.shape[0]
            fov = self.get_parameter('cam_fov').value

            # Use current ROS time for detections to match TF
            now_msg = self.get_clock().now().to_msg()

            det2d_arr = Detection2DArray()
            det2d_arr.header.stamp = now_msg
            det2d_arr.header.frame_id = self.latest_rgb.header.frame_id

            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    conf = float(box.conf.cpu().numpy())
                    cls = int(box.cls.cpu().numpy())
                    label = self.yolo_labels[cls]
                    if label not in ('car', 'truck'):
                        continue

                    cx = int((xyxy[0] + xyxy[2]) / 2.0)
                    cy = int((xyxy[1] + xyxy[3]) / 2.0)
                    z_val = float(depth[cy, cx]) if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1] else 0.0

                    now_msg = self.get_clock().now().to_msg()
                    # Use now_msg for TF lookup instead of stale RGB timestamp
                    pt3d = self.camera_to_base_link(cx, cy, z_val, width, height, fov, now_msg)
                    if pt3d is None:
                        continue

                    position = Point2D(x=float(cx), y=float(cy))

                    pose2d = Pose2D()
                    pose2d.position = position
                    pose2d.theta = 0.0

                    bb = BoundingBox2D()
                    bb.center = pose2d
                    bb.size_x = float(xyxy[2] - xyxy[0])
                    bb.size_y = float(xyxy[3] - xyxy[1])

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = label
                    hyp.hypothesis.score = conf

                    pose_cov = PoseWithCovariance()
                    pose_cov.pose.position.x = pt3d.x
                    pose_cov.pose.position.y = pt3d.y
                    pose_cov.pose.position.z = pt3d.z
                    pose_cov.pose.orientation.w = 1.0
                    pose_cov.covariance = [0.0] * 36
                    hyp.pose = pose_cov

                    d = Detection2D()
                    d.bbox = bb
                    d.results.append(hyp)
                    det2d_arr.detections.append(d)

            self.vision_pub.publish(det2d_arr)
            self.get_logger().debug(f"Published {len(det2d_arr.detections)} detections at {now_msg.sec}.{now_msg.nanosec}")

        except Exception as e:
            self.get_logger().error(f"[try_infer] Failed inference or conversion: {e}")
    

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionAndDetection()
    try:    
        rclpy.spin(node)
    finally:    
        node.destroy_node()
        rclpy.shutdown()











'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.srv import SpawnObject, DestroyObject
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped, PointStamped
from diagnostic_msgs.msg import KeyValue

from sensor_msgs.msg import Image as RosImage
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Pose2D, Point2D
from geometry_msgs.msg import PoseWithCovariance

from cv_bridge import CvBridge
from ultralytics import YOLO
from PIL import Image as PILImage

import numpy as np
import math

import tf2_ros
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Vector3


class SensorFusionAndDetection(Node):
    def __init__(self):
        super().__init__('sensor_fusion_and_detection')

        # Parameters
        self.declare_parameter('ego_role', 'ego_vehicle')
        self.declare_parameter('lidar_z', 1.7)
        self.declare_parameter('cam_offset_x', 2.4)
        self.declare_parameter('cam_offset_y', 0.0)
        self.declare_parameter('cam_offset_z', 1.5)
        self.declare_parameter('cam_fov', 90.0)
        self.declare_parameter('vision_topic', '/vision_detections')

        # YOLO model
        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8m.pt')
        self.labels = self.yolo.names

        # IDs for spawned sensors
        self.current_vehicle_id = None
        self.current_lidar_id = None
        self.current_rgb_id = None
        self.current_cam_id = None
        self.current_odom_id = None
        self.current_tf_id      = None  # <--- initialize TF sensor ID

        self.pseudo_spawned = False

        # TF setup
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_dyn      = TransformBroadcaster(self)
        self.tf_stat     = StaticTransformBroadcaster(self)

        # Service clients
        self.spawn_cli   = self.create_client(SpawnObject, '/carla/spawn_object')
        self.destroy_cli = self.create_client(DestroyObject, '/carla/destroy_object')
        for cli in [self.spawn_cli, self.destroy_cli]:
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for service {cli.srv_name}...')

        # Subscribe to CARLA vehicle_info to trigger sensor spawns
        self.create_subscription(
            CarlaEgoVehicleInfo,
            f'/carla/{self.get_parameter("ego_role").value}/vehicle_info',
            self.ego_cb,
            1)

        # Vision publisher
        self.vision_pub = self.create_publisher(
            Detection2DArray,
            self.get_parameter('vision_topic').value,
            10)

        # Buffers for image callbacks
        self.latest_rgb   = None
        self.latest_depth = None

        self.get_logger().info('SensorFusionAndDetection node initialized.')

    def ego_cb(self, msg: CarlaEgoVehicleInfo):
        """Spawn sensors when the ego vehicle first appears (or respawns)."""
        vid = msg.id
        if vid != self.current_vehicle_id:
            self._destroy_old_sensors()
            self.current_vehicle_id = vid

            # Real sensors
            self._spawn_sensor('sensor.lidar.ray_cast', 'lidar',
                Point(z=self.get_parameter('lidar_z').value))
            self._spawn_sensor('sensor.camera.rgb',   'rgb_camera',
                Point(x=self.get_parameter('cam_offset_x').value,
                      y=self.get_parameter('cam_offset_y').value,
                      z=self.get_parameter('cam_offset_z').value))
            self._spawn_sensor('sensor.camera.depth', 'depth_camera',
                Point(x=self.get_parameter('cam_offset_x').value,
                      y=self.get_parameter('cam_offset_y').value,
                      z=self.get_parameter('cam_offset_z').value))

            # Pseudo-sensors for Odometry and TF
            self._spawn_sensor('sensor.pseudo.odom', 'odometry', Point())
            self._spawn_sensor('sensor.pseudo.tf',   'base_link',       Point())
            self.pseudo_spawned = False


    
    def _destroy_old_sensors(self):
        # destroy and clear any previously spawned sensors
        for sid in [self.current_lidar_id, self.current_rgb_id, self.current_cam_id,
                    self.current_odom_id, self.current_tf_id]:
            if sid:
                req = DestroyObject.Request()
                req.id = sid
                self.destroy_cli.call_async(req)
        self.current_lidar_id = None
        self.current_rgb_id   = None
        self.current_cam_id   = None
        self.current_odom_id  = None
        self.current_tf_id    = None
        self.pseudo_spawned   = False

    
    
    
    def _spawn_sensor(self, type_, role, position: Point):
        req = SpawnObject.Request()
        req.id        = role
        req.type      = type_
        
        req.attach_to = self.current_vehicle_id
        req.transform = Pose(position=position, orientation=Quaternion(w=1.0))
        req.attributes = [KeyValue(key='role_name', value=role)]
        if type_.startswith('sensor.camera'):
            req.attributes.append(
                KeyValue(key='fov', value=str(self.get_parameter('cam_fov').value)))
        self.spawn_cli.call_async(req).add_done_callback(
            lambda future, r=role: self._resp_cb(future, r))

    def _resp_cb(self, future, role):
        resp = future.result()
        if resp.id < 0:
            self.get_logger().error(f"Failed to spawn {role}")
            return
        if role == 'lidar':
            self.current_lidar_id = resp.id
            self.get_logger().info('LiDAR spawned.')
        elif role == 'rgb_camera':
            self.current_rgb_id = resp.id
            topic = f'/carla/{self.get_parameter("ego_role").value}/rgb_camera/image'
            self.create_subscription(RosImage, topic, self.rgb_cb, 10)
        elif role == 'depth_camera':
            self.current_cam_id = resp.id
            topic = f'/carla/{self.get_parameter("ego_role").value}/depth_camera/image'
            self.create_subscription(RosImage, topic, self.depth_cb, 10)
        # After LiDAR+RGB+Depth, publish static mounts + pseudo-sensors
        if (self.current_lidar_id and self.current_rgb_id and
            self.current_cam_id and not self.pseudo_spawned):
            self.get_logger().info('Publishing static TF mounts and spawning pseudo-sensors.')
            self.publish_static_mounts()
            # spawn odometry and tf after static mounts
            self._spawn_sensor('sensor.pseudo.odom', 'odometry', Point())
            self._spawn_sensor('sensor.pseudo.tf',   'base_link', Point())
            self.pseudo_spawned = True
        elif role == 'odometry':
            self.current_odom_id = resp.id
            self.get_logger().info('Odometry spawned; subscribing.')
            topic = f'/carla/{self.get_parameter("ego_role").value}/odometry'
            self.create_subscription(Odometry, topic, self.odometry_cb, 10)
        elif role == 'base_link':
            self.current_tf_id = resp.id
            self.get_logger().info('TF sensor spawned; listening on /tf_static.')
            
    def odometry_cb(self, msg: Odometry):
        """Convert incoming Odometry → map→base_link TF."""
        tf = TransformStamped()
        tf.header.stamp    = msg.header.stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id  = 'base_link'
        #tf.transform.translation = msg.pose.pose.position
        tf.transform.translation = Vector3(
        x=msg.pose.pose.position.x,
        y=msg.pose.pose.position.y,
        z=msg.pose.pose.position.z)

        tf.transform.rotation    = msg.pose.pose.orientation
        self.tf_dyn.sendTransform(tf)

    def publish_static_mounts(self):
        """Static transforms from base_link → {lidar, rgb, depth}."""
        now = self.get_clock().now().to_msg()

        # base_link → my_lidar
        t0 = TransformStamped()
        t0.header.stamp    = now
        t0.header.frame_id = 'base_link'
        t0.child_frame_id  = 'my_lidar'
        t0.transform.translation.z = self.get_parameter('lidar_z').value
        t0.transform.rotation.w    = 1.0

        # base_link → my_rgb_camera
        t1 = TransformStamped()
        t1.header.stamp    = now
        t1.header.frame_id = 'base_link'
        t1.child_frame_id  = 'my_rgb_camera'
        t1.transform.translation.x = self.get_parameter('cam_offset_x').value
        t1.transform.translation.y = self.get_parameter('cam_offset_y').value
        t1.transform.translation.z = self.get_parameter('cam_offset_z').value
        t1.transform.rotation.w    = 1.0

        # base_link → my_depth_camera
        t2 = TransformStamped()
        t2.header.stamp    = now
        t2.header.frame_id = 'base_link'
        t2.child_frame_id  = 'my_depth_camera'
        t2.transform.translation.x = self.get_parameter('cam_offset_x').value
        t2.transform.translation.y = self.get_parameter('cam_offset_y').value
        t2.transform.translation.z = self.get_parameter('cam_offset_z').value
        t2.transform.rotation.w    = 1.0

        self.tf_stat.sendTransform([t0, t1, t2])

    def rgb_cb(self, msg: RosImage):
        self.latest_rgb = msg
        self.try_infer()

    def depth_cb(self, msg: RosImage):
        self.latest_depth = msg
        self.try_infer()

    def camera_to_base_link(self, u, v, depth, width, height, fov):
        fx = fy = (width/2) / math.tan(math.radians(fov/2))
        cx, cy = width/2, height/2
        pt = PointStamped()
        pt.header.stamp    = self.get_clock().now().to_msg()
        pt.header.frame_id = 'my_rgb_camera'
        pt.point.x = (u - cx) * depth / fx
        pt.point.y = (v - cy) * depth / fy
        pt.point.z = depth
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', pt.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            return do_transform_point(pt, tf).point
        except Exception as e:
            self.get_logger().error(f'TF transform error: {e}')
            return None
'''
'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleInfo
from tf2_msgs.msg import TFMessage  # ── CHANGES ── need to import TFMessage

from carla_msgs.srv import SpawnObject, DestroyObject
from geometry_msgs.msg import (
    Pose, Point, Quaternion,
    TransformStamped, PointStamped, Vector3
)
from diagnostic_msgs.msg import KeyValue
from sensor_msgs.msg import Image as RosImage
from vision_msgs.msg import (
    Detection2DArray, Detection2D, ObjectHypothesisWithPose,
    Pose2D, Point2D
)
from geometry_msgs.msg import PoseWithCovariance
from cv_bridge import CvBridge
from ultralytics import YOLO
from PIL import Image as PILImage
import math
import tf2_ros
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

class SensorFusionAndDetection(Node):
    def __init__(self):
        super().__init__('sensor_fusion_and_detection')
        p = self.declare_parameter
        # parameters
        self.ego_role = p('ego_role', 'ego_vehicle').value
        self.lidar_z = p('lidar_z', 1.7).value
        self.lidar_range = p('lidar_range', 50.0).value
        self.lidar_rotation_frequency = p('lidar_rotation_frequency', 40.0).value
        self.lidar_channels = p('lidar_channels', 64).value
        self.lidar_points_per_second = p('lidar_points_per_second', 700000).value
        self.lidar_sensor_tick = p('lidar_sensor_tick', 0.025).value
        self.cam_offset_x = p('cam_offset_x', 2.4).value
        self.cam_offset_y = p('cam_offset_y', 0.0).value
        self.cam_offset_z = p('cam_offset_z', 1.5).value
        self.cam_fov = p('cam_fov', 90.0).value
        self.vision_topic = p('vision_topic', '/vision_detections').value

        # detection model
        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8m.pt')
        self.labels = self.yolo.names

        # track actor IDs
        self.current_vehicle_id = None
        self.current_lidar_id = None
        self.current_dep_id = None
        self.current_rgb_id = None
        self.current_odom_id = None
        # track destroy-in-progress flags
        self.destroying_lidar = False
        self.destroying_depth = False
        self.destroying_rgb = False
        self.destroying_odom = False
        # ready flags
        self.lidar_ready = False
        self.rgb_ready = False
        self.depth_ready = False
        self.pseudo_spawned = False
        
        self.last_odom_stamp = None


        # TF
        self.tf_buffer = Buffer()
        #TransformListener(self.tf_buffer, self)
        self.tf_dyn = TransformBroadcaster(self)
        self.tf_stat = StaticTransformBroadcaster(self)
        
        self.tf_sub = self.create_subscription(
            TFMessage, '/tf', self._tf_filter_cb, 10)
        self.tf_static_sub = self.create_subscription(
            TFMessage, '/tf_static', self._tf_static_filter_cb, 10)

        # service clients
        self.spawn_cli = self.create_client(SpawnObject, '/carla/spawn_object')
        self.destroy_cli = self.create_client(DestroyObject, '/carla/destroy_object')
        for cli in (self.spawn_cli, self.destroy_cli):
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {cli.srv_name}...')

        # subscribe to ego info
        self.create_subscription(CarlaEgoVehicleInfo,
                                 f'/carla/{self.ego_role}/vehicle_info',
                                 self.ego_info_cb, 1)
        # publisher for vision detections
        self.vision_pub = self.create_publisher(Detection2DArray, self.vision_topic, 10)
        self.latest_rgb = None
        self.latest_depth = None

        self.get_logger().info('SensorFusionAndDetection node initialized.')

    # ------------------------ Destruction Helpers ------------------------
    
    
    
    
    def _tf_filter_cb(self, msg: TFMessage):
        for t in msg.transforms:
            if not (t.header.frame_id == 'map'
                    and t.child_frame_id.startswith(f'{self.ego_role}/')):
                # keep it
                #self.tf_buffer.set_transform(t, 'bridge', is_static=False)
                self.tf_buffer.set_transform(t, 'bridge')

    def _tf_static_filter_cb(self, msg: TFMessage):
        for t in msg.transforms:
            if not (t.header.frame_id == 'map'
                    and t.child_frame_id.startswith(f'{self.ego_role}/')):
                #self.tf_buffer.set_transform(t, 'bridge', is_static=True)
                self.tf_buffer.set_transform(t, 'bridge')
                
    
    
    
    def destroy_lidar(self):
        if self.current_lidar_id and not self.destroying_lidar:
            self.destroying_lidar = True
            req = DestroyObject.Request()
            req.id = self.current_lidar_id
            self.current_lidar_id = None
            self.destroy_cli.call_async(req).add_done_callback(
                lambda _: setattr(self, 'destroying_lidar', False)
            )

    def destroy_depth(self):
        if self.current_dep_id and not self.destroying_depth:
            self.destroying_depth = True
            req = DestroyObject.Request()
            req.id = self.current_dep_id
            self.current_dep_id = None
            self.destroy_cli.call_async(req).add_done_callback(
                lambda _: setattr(self, 'destroying_depth', False)
            )

    def destroy_rgb(self):
        if self.current_rgb_id and not self.destroying_rgb:
            self.destroying_rgb = True
            req = DestroyObject.Request()
            req.id = self.current_rgb_id
            self.current_rgb_id = None
            self.destroy_cli.call_async(req).add_done_callback(
                lambda _: setattr(self, 'destroying_rgb', False)
            )

    def destroy_odom(self):
        if self.current_odom_id and not self.destroying_odom:
            self.destroying_odom = True
            req = DestroyObject.Request()
            req.id = self.current_odom_id
            self.current_odom_id = None
            self.destroy_cli.call_async(req).add_done_callback(
                lambda _: setattr(self, 'destroying_odom', False)
            )

    def destroy_all_sensors(self):
        # destroy real and pseudo sensors
        self.destroy_lidar()
        self.destroy_depth()
        self.destroy_rgb()
        self.destroy_odom()

    # ------------------------ Ego Info Callback ------------------------
    def ego_info_cb(self, msg: CarlaEgoVehicleInfo):
        vid = msg.id
        if vid != self.current_vehicle_id:
            self.get_logger().info(f'Ego changed: old {self.current_vehicle_id}, new {vid}. Cleaning up...')
            self.current_vehicle_id = vid
            # destroy all previous
            self.destroy_all_sensors()
            # reset flags
            self.lidar_ready = self.rgb_ready = self.depth_ready = False
            self.pseudo_spawned = False
            # spawn new sensors
            self.spawn_lidar(vid)
            self.spawn_depth_camera(vid)
            self.spawn_rgb_camera(vid)

    # ------------------------ Spawn LiDAR ------------------------
    def spawn_lidar(self, vid):
        if not self.destroying_lidar and not self.current_lidar_id:
            req = SpawnObject.Request()
            req.id = 'lidar'
            req.type = 'sensor.lidar.ray_cast'
            req.attach_to = vid
            req.transform = Pose(position=Point(z=self.lidar_z), orientation=Quaternion(w=1.0))
            req.attributes = [
                KeyValue(key='role_name', value='my_lidar'),
                KeyValue(key='range', value=str(self.lidar_range)),
                KeyValue(key='rotation_frequency', value=str(self.lidar_rotation_frequency)),
                KeyValue(key='channels', value=str(self.lidar_channels)),
                KeyValue(key='points_per_second', value=str(self.lidar_points_per_second)),
                KeyValue(key='sensor_tick', value=str(self.lidar_sensor_tick)),
            ]
            self.spawn_cli.call_async(req).add_done_callback(self.lidar_spawn_resp)

    def lidar_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_lidar_id = resp.id
            self.lidar_ready = True
            self.get_logger().info(f'LiDAR spawned id={resp.id}')
            self._maybe_configure_mounts()

    # ------------------------ Spawn Depth ------------------------
    def spawn_depth_camera(self, vid):
        if not self.destroying_depth and not self.current_dep_id:
            req = SpawnObject.Request()
            req.id = 'depth_camera'
            req.type = 'sensor.camera.depth'
            req.attach_to = vid
            req.transform = Pose(position=Point(x=self.cam_offset_x, y=self.cam_offset_y, z=self.cam_offset_z),
                                  orientation=Quaternion(w=1.0))
            req.attributes = [
                KeyValue(key='role_name', value='my_depth_camera'),
                KeyValue(key='fov', value=str(self.cam_fov)),
                KeyValue(key='sensor_tick', value=str(0.025)),
            ]
            self.spawn_cli.call_async(req).add_done_callback(self.depth_spawn_resp)

    def depth_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_dep_id = resp.id
            self.depth_ready = True
            self.get_logger().info(f'Depth camera spawned id={resp.id}')
            topic = f'/carla/{self.ego_role}/my_depth_camera/image'
            self.create_subscription(RosImage, topic, self.depth_cb, 10)
            self._maybe_configure_mounts()

    # ------------------------ Spawn RGB ------------------------
    def spawn_rgb_camera(self, vid):
        if not self.destroying_rgb and not self.current_rgb_id:
            req = SpawnObject.Request()
            req.id = 'rgb_camera'
            req.type = 'sensor.camera.rgb'
            req.attach_to = vid
            req.transform = Pose(position=Point(x=self.cam_offset_x, y=self.cam_offset_y, z=self.cam_offset_z),
                                  orientation=Quaternion(w=1.0))
            req.attributes = [
                KeyValue(key='role_name', value='my_rgb_camera'),
                KeyValue(key='fov', value=str(self.cam_fov)),
                KeyValue(key='sensor_tick', value=str(0.025)),
            ]
            self.spawn_cli.call_async(req).add_done_callback(self.rgb_spawn_resp)

    def rgb_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_rgb_id = resp.id
            self.rgb_ready = True
            self.get_logger().info(f'RGB camera spawned id={resp.id}')
            topic = f'/carla/{self.ego_role}/my_rgb_camera/image'
            self.create_subscription(RosImage, topic, self.rgb_cb, 10)
            self._maybe_configure_mounts()

    # ------------------------ Configure Mounts & Pseudo ------------------------
    def _maybe_configure_mounts(self):
        if self.lidar_ready and self.rgb_ready and self.depth_ready and not self.pseudo_spawned:
            self.spawn_pseudo_odom()
            self.publish_static_mounts()
            self.pseudo_spawned = True

    # spawn only odometry pseudo sensor
    def spawn_pseudo_odom(self):
        if not self.destroying_odom and not self.current_odom_id:
            req = SpawnObject.Request()
            req.id = 'odometry'
            req.type = 'sensor.pseudo.odom'
            req.attach_to = self.current_vehicle_id
            req.transform = Pose()
            req.attributes = [KeyValue(key='role_name', value='odometry'),
                               KeyValue(key='sensor_tick', value='0.02')]
            self.spawn_cli.call_async(req).add_done_callback(self.odom_spawn_resp)

    def odom_spawn_resp(self, future):
        resp = future.result()
        if resp.id >= 0:
            self.current_odom_id = resp.id
            topic = f'/carla/{self.ego_role}/odometry'
            self.create_subscription(Odometry, topic, self.odometry_cb, 10)

    # ------------------------ Odometry Callback ------------------------
    def odometry_cb(self, msg: Odometry):
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        #self.last_odom_stamp = msg.header.stamp

        tf.header.frame_id = 'map'
        tf.child_frame_id = 'base_link'
        tf.transform.translation = Vector3(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            z=msg.pose.pose.position.z
        )
        tf.transform.rotation = msg.pose.pose.orientation
        self.tf_dyn.sendTransform(tf)
        self.last_odom_stamp = msg.header.stamp

    # ------------------------ Static Mounts ------------------------
    def publish_static_mounts(self):
        now = self.get_clock().now().to_msg()
        ego = self.ego_role
        # LiDAR mount
        t0 = TransformStamped()
        t0.header.stamp = now
        t0.header.frame_id = 'base_link'
        t0.child_frame_id = f"{ego}/my_lidar"
        t0.transform.translation.z = self.lidar_z
        t0.transform.rotation.w = 1.0
        # RGB mount
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = 'base_link'
        t1.child_frame_id = f"{ego}/my_rgb_camera"
        t1.transform.translation.x = self.cam_offset_x
        t1.transform.translation.y = self.cam_offset_y
        t1.transform.translation.z = self.cam_offset_z
        t1.transform.rotation.w = 1.0
        # Depth mount
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'base_link'
        t2.child_frame_id = f"{ego}/my_depth_camera"
        t2.transform.translation.x = self.cam_offset_x
        t2.transform.translation.y = self.cam_offset_y
        t2.transform.translation.z = self.cam_offset_z
        t2.transform.rotation.w = 1.0
        self.tf_stat.sendTransform([t0, t1, t2])

    # ------------------------ Callbacks & Inference ------------------------
    def rgb_cb(self, msg: RosImage):
        self.latest_rgb = msg
        self.try_infer()

    def depth_cb(self, msg: RosImage):
        self.latest_depth = msg
        self.try_infer()
'''
'''####
    
    def camera_to_base_link(self, u, v, depth, width, height, fov):
        fx = fy = (width/2)/math.tan(math.radians(fov/2))
        cx, cy = width/2, height/2
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = f"{self.ego_role}/my_rgb_camera"
        pt.point.x = (u - cx)*depth/fx
        pt.point.y = (v - cy)*depth/fy
        pt.point.z = depth
        try:
            xf = self.tf_buffer.lookup_transform('ego_vehicle/my_lidar', pt.header.frame_id,
                                                 rclpy.time.Time(),
                                                 timeout=rclpy.duration.Duration(seconds=0.1))
            return do_transform_point(pt, xf).point
        except Exception as e:
            self.get_logger().error(f'TF error: {e}')
            return None

    def try_infer(self):
        if not (self.latest_rgb and self.latest_depth):
            print("returning ")
            return
        rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'rgb8')
        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        height, width = rgb.shape[:2]
        fov = self.cam_fov
        det_arr = Detection2DArray()
        det_arr.header.stamp = self.get_clock().now().to_msg()
        det_arr.header.frame_id = self.latest_rgb.header.frame_id #'base_link'
        results = self.yolo(PILImage.fromarray(rgb))
        for r in results:
            for box in r.boxes:
                xy = box.xyxy.cpu().numpy().flatten()
                cx, cy = int((xy[0]+xy[2])/2), int((xy[1]+xy[3])/2)
                z = float(depth[cy, cx]) if 0<=cx<width and 0<=cy<height else 0.0
                p3d = self.camera_to_base_link(cx, cy, z, width, height, fov)
                if not p3d:
                    continue
                det = Detection2D()
                det.bbox.center = Pose2D(position=Point2D(x=float(cx), y=float(cy)), theta=0.0)
                det.bbox.size_x = float(xy[2]-xy[0])
                det.bbox.size_y = float(xy[3]-xy[1])
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self.labels[int(box.cls.cpu())]
                hyp.hypothesis.score = float(box.conf.cpu())
                hyp.pose = PoseWithCovariance()
                hyp.pose.pose.position = p3d
                hyp.pose.pose.orientation.w = 1.0
                det.results.append(hyp)
                det_arr.detections.append(det)
        self.vision_pub.publish(det_arr)
'''  #####
        
''' ##
    def camera_to_base_link(self, u, v, depth, width, height, fov, stamp_msg):
        fx = fy = (width / 2) / math.tan(math.radians(fov / 2))
        cx, cy = width / 2, height / 2

        pt = PointStamped()
        pt.header.stamp = stamp_msg
        pt.header.frame_id = f"{self.ego_role}/my_rgb_camera"
        pt.point.x = (u - cx) * depth / fx
        pt.point.y = (v - cy) * depth / fy
        pt.point.z = depth

        try:
            lookup_time = rclpy.time.Time.from_msg(stamp_msg)
            tf = self.tf_buffer.lookup_transform(
                'ego_vehicle/my_lidar',  # desired target frame
                pt.header.frame_id,      # source frame
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            return do_transform_point(pt, tf).point
        except Exception as e:
            self.get_logger().error(f'TF error: {e}')
            return None

    def try_infer(self):
        if not (self.latest_rgb and self.latest_depth):
            return

        rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'rgb8')
        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        height, width = rgb.shape[:2]
        fov = self.cam_fov

        # Use a safe timestamp for TF
        now_msg = self.get_clock().now().to_msg()

        det_arr = Detection2DArray()
        det_arr.header.stamp = now_msg
        det_arr.header.frame_id = 'ego_vehicle/my_lidar'

        results = self.yolo(PILImage.fromarray(rgb))
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.cpu())
                label = self.labels[cls_id]
                if label not in ('car', 'truck'):  # optional filtering
                    continue

                conf = float(box.conf.cpu())
                xy = box.xyxy.cpu().numpy().flatten()
                cx, cy = int((xy[0] + xy[2]) / 2), int((xy[1] + xy[3]) / 2)

                # Validate depth pixel
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                z = float(depth[cy, cx])
                if z <= 0.1:  # skip invalid or near-zero depth
                    continue

                # Transform 3D point to LiDAR frame
                p3d = self.camera_to_base_link(cx, cy, z, width, height, fov, now_msg)
                if not p3d:
                    continue

                # Build detection
                det = Detection2D()
                det.bbox.center = Pose2D(position=Point2D(x=float(cx), y=float(cy)), theta=0.0)
                det.bbox.size_x = float(xy[2] - xy[0])
                det.bbox.size_y = float(xy[3] - xy[1])

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score = conf

                pose = PoseWithCovariance()
                pose.pose.position = p3d
                pose.pose.orientation.w = 1.0
                pose.covariance = [0.0] * 36

                hyp.pose = pose
                det.results.append(hyp)
                det_arr.detections.append(det)

        self.vision_pub.publish(det_arr)

        
        
''' ##

           
'''
    # ── inside class SensorFusionAndDetection ──

    def camera_to_base_link(self, u, v, depth, width, height, fov):
        fx = fy = (width / 2) / math.tan(math.radians(fov / 2))
        cx, cy = width / 2, height / 2

        pt = PointStamped()
        pt.header.stamp = self.latest_rgb.header.stamp
        pt.header.frame_id = f"{self.ego_role}/my_rgb_camera"
        pt.point.x = (u - cx) * depth / fx
        pt.point.y = (v - cy) * depth / fy
        pt.point.z = depth

        try:
            # Get latest common time between base_link and camera
            lookup_time = self.tf_buffer.get_latest_common_time('base_link', pt.header.frame_id)
            xf = self.tf_buffer.lookup_transform(
                'ego_vehicle/my_lidar',
                pt.header.frame_id,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return do_transform_point(pt, xf).point

        except Exception as e:
            self.get_logger().error(f'TF error: {e}')
            return None


    def try_infer(self):
        if not (self.latest_rgb and self.latest_depth):
            return

        rgb   = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'rgb8')
        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        height, width = rgb.shape[:2]
        fov = self.cam_fov

        det_arr = Detection2DArray()

        # Use the earlier of image and odometry timestamp
        img_ts  = self.latest_rgb.header.stamp
        odom_ts = self.last_odom_stamp or img_ts
        odom_sec = odom_ts.sec + odom_ts.nanosec * 1e-9
        img_sec  = img_ts.sec  + img_ts.nanosec * 1e-9
        det_arr.header.stamp = odom_ts if odom_sec < img_sec else img_ts

        det_arr.header.frame_id = self.latest_rgb.header.frame_id

        results = self.yolo(PILImage.fromarray(rgb))
        for r in results:
            for box in r.boxes:
                xy = box.xyxy.cpu().numpy().flatten()
                cx, cy = int((xy[0] + xy[2]) / 2), int((xy[1] + xy[3]) / 2)
                z = float(depth[cy, cx]) if 0 <= cx < width and 0 <= cy < height else 0.0
                if z == 0.0:
                    continue

                p3d = self.camera_to_base_link(cx, cy, z, width, height, fov)
                if not p3d:
                    continue

                det = Detection2D()
                det.bbox.center = Pose2D(position=Point2D(x=float(cx), y=float(cy)), theta=0.0)
                det.bbox.size_x = float(xy[2] - xy[0])
                det.bbox.size_y = float(xy[3] - xy[1])
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self.labels[int(box.cls.cpu())]
                hyp.hypothesis.score = float(box.conf.cpu())
                hyp.pose = PoseWithCovariance()
                hyp.pose.pose.position = p3d
                hyp.pose.pose.orientation.w = 1.0

                det.results.append(hyp)
                det_arr.detections.append(det)

        self.vision_pub.publish(det_arr)




def main():
    rclpy.init()
    node = SensorFusionAndDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()

    # Fixes and additions:
    # - Added destroy_lidar(), destroy_depth(), destroy_rgb(), destroy_odom() to properly clean up old sensors.
    # - ego_info_cb now calls destroy_all_sensors() before respawning, resetting flags and IDs.
    # - Removed pseudo TF spawn; only odometry pseudo sensor is spawned and destroyed.
    # - Ensured sensors’ ready flags reset so _maybe_configure_mounts() triggers correctly.
    # - Wrapped each destroy call with flags to prevent duplicate requests.


'''

