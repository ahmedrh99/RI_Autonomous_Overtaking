'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import numpy as np
from scipy.optimize import linear_sum_assignment

from autoware_perception_msgs.msg import DetectedObjects, DetectedObject

from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import BoundingBox3D #Center
from geometry_msgs.msg import Point
from std_msgs.msg import Header

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        # Subscriptions
        self.cam_sub = self.create_subscription(
            Detection2DArray,
            '/vision_detections',
            self.cam_cb,
            10)
        self.lidar_sub = self.create_subscription(
            DetectedObjects,
            '/perception/objects',
            self.lidar_cb,
            10)
        # Publisher for fused 3D detections
        self.fused_pub = self.create_publisher(
            Detection3DArray,
            '/fused_detections',
            10)

        # Internal buffers
        self.latest_cam = None
        self.latest_lidar = None
        self.matching_threshold = 1.5  # meters
        self.get_logger().info('SensorFusionNode initialized with threshold=%0.1f m.' % self.matching_threshold)

    def cam_cb(self, msg: Detection2DArray):
        self.latest_cam = msg
        self.try_fuse()

    def lidar_cb(self, msg: DetectedObjects):
        self.latest_lidar = msg
        self.try_fuse()

    def try_fuse(self):
        # Fuse only when both or either arrives
        if self.latest_cam is None and self.latest_lidar is None:
            return
        fused = self.fuse(self.latest_cam, self.latest_lidar)
        header = self.latest_cam.header if self.latest_cam else self.latest_lidar.header
        self.publish_fused(fused, header)
        self.latest_cam = None
        self.latest_lidar = None

    def fuse(self, cam_arr: Detection2DArray, lidar_arr: DetectedObjects):
        # Build detection lists: positions, labels, scores
        cam_list = []
        if cam_arr:
            for det in cam_arr.detections:
                if det.results:
                    hypo = det.results[0]
                    pos = np.array([hypo.pose.pose.position.x,
                                    hypo.pose.pose.position.y,
                                    hypo.pose.pose.position.z])
                    cam_list.append({'pos':pos, 'label':hypo.hypothesis.class_id, 'score':hypo.hypothesis.score})
        lidar_list = []
        if lidar_arr:
            for det in lidar_arr.objects:
                # Extract position from pose
                pos = np.array([
                    det.kinematics.pose_with_covariance.pose.position.x,
                    det.kinematics.pose_with_covariance.pose.position.y,
                    det.kinematics.pose_with_covariance.pose.position.z
                ])

                # Extract classification
                if det.classification:
                    label = det.classification[0].label
                    score = det.classification[0].probability
                else:
                    label = 0  # or some "unknown" ID
                    score = 0.0

                lidar_list.append({'pos': pos, 'label': label, 'score': score})

        fused = []
        if cam_list and lidar_list:
            # Compute cost matrix based on Euclidean distance
            cost = np.zeros((len(cam_list), len(lidar_list)))
            for i, c in enumerate(cam_list):
                for j, l in enumerate(lidar_list):
                    cost[i, j] = np.linalg.norm(c['pos'] - l['pos'])
            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_cam = set()
            assigned_lidar = set()
            # Process matches
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < self.matching_threshold:
                    c = cam_list[i]
                    l = lidar_list[j]
                    # fused position: weighted by score
                    w_sum = c['score'] + l['score']
                    fuse_pos = (c['pos'] * c['score'] + l['pos'] * l['score']) / w_sum
                    fuse_label = c['label']
                    fuse_score = max(c['score'], l['score'])
                    fused.append({'pos':fuse_pos, 'label':fuse_label, 'score':fuse_score})
                    assigned_cam.add(i)
                    assigned_lidar.add(j)
            # Unmatched camera detections
            for i, c in enumerate(cam_list):
                if i not in assigned_cam:
                    fused.append(c)
            # Unmatched LiDAR detections
            for j, l in enumerate(lidar_list):
                if j not in assigned_lidar:
                    fused.append(l)
        else:
            # One or both lists empty: include all
            fused.extend(cam_list if cam_list else [])
            fused.extend(lidar_list if lidar_list else [])

        return fused

    def publish_fused(self, fused_list, header: Header):
        out = Detection3DArray()
        out.header = header
        for item in fused_list:
            pos = item['pos']
            label, score = item['label'], item['score']
            det3d = Detection3D()
            center = BoundingBox3D()
            det3d.bbox.center.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            #det3d.bbox = center
            hypo = ObjectHypothesisWithPose()
            hypo.hypothesis.class_id = label
            hypo.hypothesis.score = score
            det3d.results.append(hypo)
            out.detections.append(det3d)
        self.fused_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
'''



#!/usr/bin/env python3 this was working 
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.optimize import linear_sum_assignment

from autoware_perception_msgs.msg import DetectedObjects, DetectedObject, ObjectClassification
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import Header


CLASS_NAME_TO_ID = {
    'unknown': 0,
    'car': 1,
    'truck': 2,
    'bus': 3,
    'trailer': 4,
    'motorcycle': 5,
    'motorbike': 5,
    'bicycle': 6,
    'pedestrian': 7,
}

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscriptions
        self.cam_sub = self.create_subscription(
            Detection2DArray,
            '/vision_detections',
            self.cam_cb,
            10)
        self.lidar_sub = self.create_subscription(
            DetectedObjects,
            '/perception/objects',
            self.lidar_cb,
            10)

        # Publisher for fused detections in Autoware format
        self.fused_pub = self.create_publisher(
            DetectedObjects,
            '/fused_detections',
            10)

        self.latest_cam = None
        self.latest_lidar = None
        self.matching_threshold = 1.5  # meters
        self.get_logger().info('SensorFusionNode initialized with threshold=%0.1f m.' % self.matching_threshold)

    def cam_cb(self, msg: Detection2DArray):
        self.latest_cam = msg
        self.try_fuse()

    def lidar_cb(self, msg: DetectedObjects):
        self.latest_lidar = msg
        self.try_fuse()

    def try_fuse(self):
        if self.latest_cam is None and self.latest_lidar is None:
            return
        fused = self.fuse(self.latest_cam, self.latest_lidar)
        header = self.latest_cam.header if self.latest_cam else self.latest_lidar.header
        self.publish_fused(fused, header)
        self.latest_cam = None
        self.latest_lidar = None

    def fuse(self, cam_arr: Detection2DArray, lidar_arr: DetectedObjects):
        cam_list = []
        if cam_arr:
            for det in cam_arr.detections:
                if det.results:
                    hypo = det.results[0]
                    pos = np.array([
                        hypo.pose.pose.position.x,
                        hypo.pose.pose.position.y,
                        hypo.pose.pose.position.z])
                    cam_list.append({'pos': pos, 'label': hypo.hypothesis.class_id, 'score': hypo.hypothesis.score})

        lidar_list = []
        if lidar_arr:
            for det in lidar_arr.objects:
                pos = np.array([
                    det.kinematics.pose_with_covariance.pose.position.x,
                    det.kinematics.pose_with_covariance.pose.position.y,
                    det.kinematics.pose_with_covariance.pose.position.z])
                if det.classification:
                    label = det.classification[0].label
                    score = det.classification[0].probability
                else:
                    label = 0
                    score = 0.0
                lidar_list.append({'pos': pos, 'label': label, 'score': score})

        fused = []
        if cam_list and lidar_list:
            cost = np.zeros((len(cam_list), len(lidar_list)))
            for i, c in enumerate(cam_list):
                for j, l in enumerate(lidar_list):
                    cost[i, j] = np.linalg.norm(c['pos'] - l['pos'])

            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_cam = set()
            assigned_lidar = set()

            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < self.matching_threshold:
                    c = cam_list[i]
                    l = lidar_list[j]
                    w_sum = c['score'] + l['score']
                    fuse_pos = (c['pos'] * c['score'] + l['pos'] * l['score']) / w_sum
                    fuse_label = c['label']
                    fuse_score = max(c['score'], l['score'])
                    fused.append({'pos': fuse_pos, 'label': fuse_label, 'score': fuse_score})
                    assigned_cam.add(i)
                    assigned_lidar.add(j)

            for i, c in enumerate(cam_list):
                if i not in assigned_cam:
                    fused.append(c)
            for j, l in enumerate(lidar_list):
                if j not in assigned_lidar:
                    fused.append(l)
        else:
            fused.extend(cam_list if cam_list else [])
            fused.extend(lidar_list if lidar_list else [])

        return fused

    def publish_fused(self, fused_list, header: Header):
        

        
        out = DetectedObjects()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "ego_vehicle/lidar"  # or your desired frame
        #out.header = header

        for item in fused_list:
            pos = item['pos']
            label, score = item['label'], item['score']

            obj = DetectedObject()
            #obj.header = header

            # Classification
            classification = ObjectClassification()
            if isinstance(label, str):
                classification.label = CLASS_NAME_TO_ID.get(label.lower(), 0)
            else:
                classification.label = label
            classification.probability = float(score)
            obj.classification.append(classification)

            # Position (minimal pose)
            obj.kinematics.pose_with_covariance.pose.position.x = float(pos[0])
            obj.kinematics.pose_with_covariance.pose.position.y = float(pos[1])
            obj.kinematics.pose_with_covariance.pose.position.z = float(pos[2])
            obj.kinematics.pose_with_covariance.pose.orientation.w = 1.0  # neutral rotation
            
            #obj.kinematics.header.stamp = out.header.stamp
            #obj.kinematics.header.frame_id = out.header.frame_id


            # Optional default values
            #obj.existence_probability = 1.0
            # Shape (important for tracker and RViz)
            obj.shape.dimensions.x = 4.0   # Length of car (in meters)
            obj.shape.dimensions.y = 1.8   # Width
            obj.shape.dimensions.z = 1.6   # Height
            obj.shape.type = 1            # 1 = BOUNDING_BOX

            # Optional flags (not mandatory, but nice)
            obj.kinematics.orientation_availability = 0  # 0 = UNKNOWN
            obj.kinematics.has_twist = False
            obj.kinematics.has_twist_covariance = False
            obj.kinematics.has_position_covariance = False

            # Required for tracker logic
            obj.existence_probability = 1.0

            out.objects.append(obj)

        self.fused_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()



'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.optimize import linear_sum_assignment

from autoware_perception_msgs.msg import DetectedObjects, DetectedObject, ObjectClassification
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped, Pose, Quaternion
from std_msgs.msg import Header

import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Point

CLASS_NAME_TO_ID = {
    'unknown': 0,
    'car': 1,
    'truck': 2,
    'bus': 3,
    'trailer': 4,
    'motorcycle': 5,
    'motorbike': 5,
    'bicycle': 6,
    'pedestrian': 7,
}

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.cam_sub = self.create_subscription(
            Detection2DArray,
            '/vision_detections',
            self.cam_cb,
            10)
        self.lidar_sub = self.create_subscription(
            DetectedObjects,
            '/perception/objects',
            self.lidar_cb,
            10)

        # Publisher for fused detections
        self.fused_pub = self.create_publisher(
            DetectedObjects,
            '/fused_detections',
            10)

        self.latest_cam = None
        self.latest_lidar = None
        self.matching_threshold = 1.5  # meters
        self.get_logger().info('SensorFusionNode initialized with threshold=%.1f m.' % self.matching_threshold)

    def cam_cb(self, msg: Detection2DArray):
        self.latest_cam = msg
        self.try_fuse()

    def lidar_cb(self, msg: DetectedObjects):
        self.latest_lidar = msg
        self.try_fuse()

    def try_fuse(self):
        if self.latest_cam is None and self.latest_lidar is None:
            return
        stamp = self.latest_cam.header.stamp if self.latest_cam else self.latest_lidar.header.stamp
        fused = self.fuse(self.latest_cam, self.latest_lidar, stamp)
        self.publish_fused(fused, stamp)
        self.latest_cam = None
        self.latest_lidar = None

    def transform_to_map(self, pos_np, source_frame, stamp):
        try:
            point = PointStamped()
            point.header.stamp = stamp
            point.header.frame_id = source_frame
            point.point.x = float(pos_np[0])
            point.point.y = float(pos_np[1])
            point.point.z = float(pos_np[2])

            tf = self.tf_buffer.lookup_transform(
                'map',
                source_frame,
                rclpy.time.Time(),  # <- Empty time = use latest available transform
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            transformed = do_transform_point(point, tf)
            return np.array([transformed.point.x, transformed.point.y, transformed.point.z])
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None


    def fuse(self, cam_arr, lidar_arr, stamp):
        cam_list = []
        if cam_arr:
            for det in cam_arr.detections:
                if det.results:
                    hypo = det.results[0]
                    pos = np.array([
                        hypo.pose.pose.position.x,
                        hypo.pose.pose.position.y,
                        hypo.pose.pose.position.z])
                    cam_list.append({'pos': pos, 'label': hypo.hypothesis.class_id, 'score': hypo.hypothesis.score})

        lidar_list = []
        if lidar_arr:
            for det in lidar_arr.objects:
                pos = np.array([
                    det.kinematics.pose_with_covariance.pose.position.x,
                    det.kinematics.pose_with_covariance.pose.position.y,
                    det.kinematics.pose_with_covariance.pose.position.z])
                label = det.classification[0].label if det.classification else 0
                score = det.classification[0].probability if det.classification else 0.0
                lidar_list.append({'pos': pos, 'label': label, 'score': score})

        fused = []
        if cam_list and lidar_list:
            cost = np.zeros((len(cam_list), len(lidar_list)))
            for i, c in enumerate(cam_list):
                for j, l in enumerate(lidar_list):
                    cost[i, j] = np.linalg.norm(c['pos'] - l['pos'])

            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_cam = set()
            assigned_lidar = set()

            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < self.matching_threshold:
                    c = cam_list[i]
                    l = lidar_list[j]
                    w_sum = c['score'] + l['score']
                    fuse_pos = (c['pos'] * c['score'] + l['pos'] * l['score']) / w_sum
                    map_pos = self.transform_to_map(fuse_pos, 'ego_vehicle/lidar', stamp)
                    if map_pos is not None:
                        fused.append({'pos': map_pos, 'label': c['label'], 'score': max(c['score'], l['score'])})
                    assigned_cam.add(i)
                    assigned_lidar.add(j)

            for i, c in enumerate(cam_list):
                if i not in assigned_cam:
                    map_pos = self.transform_to_map(c['pos'], 'ego_vehicle/lidar', stamp)
                    if map_pos is not None:
                        fused.append({'pos': map_pos, 'label': c['label'], 'score': c['score']})

            for j, l in enumerate(lidar_list):
                if j not in assigned_lidar:
                    map_pos = self.transform_to_map(l['pos'], 'ego_vehicle/lidar', stamp)
                    if map_pos is not None:
                        fused.append({'pos': map_pos, 'label': l['label'], 'score': l['score']})
        else:
            source_frame = 'ego_vehicle/lidar'
            if cam_list:
                for c in cam_list:
                    map_pos = self.transform_to_map(c['pos'], source_frame, stamp)
                    if map_pos is not None:
                        fused.append({'pos': map_pos, 'label': c['label'], 'score': c['score']})
            if lidar_list:
                for l in lidar_list:
                    map_pos = self.transform_to_map(l['pos'], source_frame, stamp)
                    if map_pos is not None:
                        fused.append({'pos': map_pos, 'label': l['label'], 'score': l['score']})

        return fused

    def publish_fused(self, fused_list, stamp):
        out = DetectedObjects()
        out.header.stamp = stamp
        out.header.frame_id = "map"

        for item in fused_list:
            pos = item['pos']
            label, score = item['label'], item['score']

            obj = DetectedObject()
            classification = ObjectClassification()
            if isinstance(label, str):
                classification.label = CLASS_NAME_TO_ID.get(label.lower(), 0)
            else:
                classification.label = label
            classification.probability = float(score)
            obj.classification.append(classification)

            obj.kinematics.pose_with_covariance.pose.position.x = float(pos[0])
            obj.kinematics.pose_with_covariance.pose.position.y = float(pos[1])
            obj.kinematics.pose_with_covariance.pose.position.z = float(pos[2])
            obj.kinematics.pose_with_covariance.pose.orientation.w = 1.0

            obj.shape.dimensions.x = 4.0
            obj.shape.dimensions.y = 1.8
            obj.shape.dimensions.z = 1.6
            obj.shape.type = 1

            obj.kinematics.orientation_availability = 0
            obj.kinematics.has_twist = False
            obj.kinematics.has_twist_covariance = False
            obj.kinematics.has_position_covariance = False
            obj.existence_probability = 1.0

            out.objects.append(obj)

        self.fused_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
'''