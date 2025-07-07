"""
world_env.py
------------
This module defines the CARLA environment wrapper used for reinforcement learning.
It includes logic for environment reset, reward computation, step processing, and data
integration from sensors like LiDAR, IMU, and camera.

Main class:
    CarlaEnv - environment wrapping the CARLA simulation.
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import numpy as np
import math
import time
import weakref
from gymnasium import Env, spaces
from ultralytics import YOLO
import cv2

from carla import ColorConverter as cc
from ultralytics import YOLO # type: ignore
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation
#from CARLA.Risk.collision_prediction.coll_prediction import collison_pred
#from CARLA.Risk.risk_assessment.seq_assessment import risk_assess


import random
import open3d as o3d 
from matplotlib import cm
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from collections import deque
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
#from CarlaICP import ICPTester, CarlaICPDemo
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env  # or from gym import Env
from gymnasium import spaces
import csv

from CARLA.env.sensors import CameraManager, CollisionSensor, IMUSensor  


try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


VIRIDIS = np.array(cm.get_cmap('plasma', 256)(np.linspace(0, 1, 256)))

VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Define this based on your semantic labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name



# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(Env): 
                    #(object):
    def __init__(self, carla_world, args,client):
        
        self.episode = 0
        self.collision_number = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.world = carla_world        
        self.client =client
        self.done = False
        self.sync = args.sync
        
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        
        
        self.player = None
        self.follower = None  # Follower car instance
        self.follower_camera = None  # Camera attached to the follower car
        self.depth_camera = None
        self.npc1 = None
        self.follower_lidar = None
        self.lidar_data = None
        self.lidar_sensor = None 
        #self.lidar_visualizer = LidarVisualizer()  
        self.actors=[]
        self.vehicles = []  # List to store NPC vehicles
        self.pedestrians = []   # List to store pedestrians

        self.yolo_model = YOLO("yolov8n.pt")
       
        #self.npc_manager = NPCManager(self.world)

        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None      
        self.log_file = None        
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.reset()
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        self.frame_counter = 1        
        
    def reset(self, seed=None, options=None):
        """
        Reset the CARLA environment and returns the initial observation.
        Respawns necessary components

       
        """
        
        self.prev_p = 1
        self.vertical = 2
        #self.dbscan = DBSCAN(eps=0.6, min_samples=5)       
        self.prev_p0 = 0        
        self.p = 0  
        self.pp = 0
        self.p3 = 0
        self.a = 0
        self.step_count = 0        
        self.loc = 0       
        #self.tracker3d = Tracker3D()
        self.frame_id  = 0
        #self.prev_centers  = {}   
        #self.tracks = []
        #self.next_id = 0  
        #self.tracker = ObjectTracker()  # Once, not every frame
        #tracked_centers = self.tracker.update(new_centerXAs) 
        #self.KalmanTracker = KalmanTracker()
        #self.trackers = []
        #self.velocity_tracker = {}  # Add this in your class constructor

        self.lidar_accumulated = []
        self.initial_position = None
        self.initial_time = None 

        self.tracked_history = []  # to store past frames
        self.img_size = 800
        self.scale = 1  # how many pixels per meter
        self.origin = (self.img_size // 2, self.img_size // 2)  # center of canvas


        self.prev_steer = 0.0
        self.episode_start_time = time.time()
        self.done = False
        self.truncated = False

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        self.reference_set = False
        
        self.reference_orientation = 0.0
        self.actor_list=[]
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

      
        # Spawn the player.
        blueprint_library = self.world.get_blueprint_library()
        player_blueprint = blueprint_library.filter('cybertruck')[0]
        #player_blueprint.set_attribute('color', '255,0,0')

        # Spawn the player.
        if self.player is not None:
            spawn_point = carla.Transform(
                #carla.Location(x=-184.580444 -3.5 , y=81.771435, z=1.0),  # Set specific location
                #carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)  # Set specific rotation
                carla.Location(x= 50, y=9.8 + 3.5, z=15.0),
                carla.Rotation(pitch=0.0, yaw=-179.8, roll=0.0)
            )
            self.destroy()
            self.player = self.world.try_spawn_actor(player_blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        else:
            # Define the exact spawn point if the player is None
            spawn_point = carla.Transform(
                #carla.Location(x=-184.580444 - 3.5, y= 81.771435, z=1.0),  # Set specific location
                #carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)  # Set specific rotation
                    carla.Location(x= 50, y=9.8 + 3.5, z=15.0),
                    carla.Rotation(pitch=0.0, yaw=-179.8, roll=0.0)
            )
            self.player = self.world.try_spawn_actor(player_blueprint, spawn_point)
            if self.player is None:
                print('Failed to spawn the player at the specified location.')
                sys.exit(1)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
            self.actor_list.append(self.player)

        
        #follower car addition 
        
        #blueprint_library = self.world.get_blueprint_library()
        follower_blueprint = blueprint_library.filter('model3')[0]
        follower_blueprint.set_attribute('role_name', 'ego_vehicle')  # ✅ Required for ROS topic
        # Get the transform of the manual car
        manual_car_transform = self.player.get_transform()
        
        # Extract location and rotation
        manual_car_location = manual_car_transform.location
        manual_car_rotation = manual_car_transform.rotation

        # Calculate a position behind the manual car
        distance_behind = 6.0  
        offset_x = -distance_behind * math.cos(math.radians(manual_car_rotation.yaw))
        offset_y = -distance_behind * math.sin(math.radians(manual_car_rotation.yaw))

        # New location for the follower car
        follower_location = carla.Location(
            x=manual_car_location.x + offset_x,
            y=manual_car_location.y + offset_y + 0.5,
            z=manual_car_location.z
        )    
        follower_car_rotation = carla.Rotation(pitch=0.0, yaw=165.8, roll=0.0)

        #follower_blueprint.set_attribute('role_name', 'follower')
        if self.follower is not None:
            follower_spawn_point = carla.Transform(follower_location, follower_car_rotation)  # Spawn 10 meters behind the player car
            self.destroy()
            self.follower = self.world.try_spawn_actor(follower_blueprint, follower_spawn_point)

            self.modify_vehicle_physics(self.follower)
            self.actor_list.append(self.follower)
        else:
            follower_spawn_point = carla.Transform(follower_location, follower_car_rotation)  # Spawn 10 meters behind the player car
            self.follower = self.world.try_spawn_actor(follower_blueprint, follower_spawn_point)
            if self.follower is None:
                print('Failed to spawn the follower at the specified location.')
                sys.exit(1)

            self.modify_vehicle_physics(self.follower)
            self.actor_list.append(self.follower)
        
        
        if not self.reference_set:
            self.reference_location = follower_location
            self.reference_orientation = manual_car_rotation.yaw
            self.reference_set = True 
        
                    
        # spawnm first npc
        npc1_blueprint = blueprint_library.filter('cybertruck')[0]      
        npc1_location = carla.Location(
            x=manual_car_location.x + 3 ,#offset_x - 3.5,
            y=manual_car_location.y + 3.5,
            z=manual_car_location.z
        )
        if self.npc1 is not None:
            npc1_spawn_point = carla.Transform(npc1_location, manual_car_rotation)
            self.destroy()
            self.npc1 = self.world.try_spawn_actor(npc1_blueprint, npc1_spawn_point)
            self.modify_vehicle_physics(self.follower)
            self.actor_list.append(self.npc1)
        else: 
            npc1_spawn_point = carla.Transform(npc1_location, manual_car_rotation)  
            self.npc1 = self.world.try_spawn_actor(npc1_blueprint, npc1_spawn_point)
            if self.npc1 is None:
                print('Failed to spawn the npc1 at the specified location.')
                sys.exit(1)
            self.modify_vehicle_physics(self.follower)
            self.actor_list.append(self.npc1)
        
  
         
            '''
        if self.follower is not None:
            print("Follower car spawned successfully.")
            
            # Attach a camera to the follower car
            camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_blueprint.set_attribute('image_size_x', '640')
            camera_blueprint.set_attribute('image_size_y', '480')
            camera_blueprint.set_attribute('fov', '110')
            camera_transform = carla.Transform(carla.Location(x=2.4, z=1.4))
            self.follower_camera = self.world.spawn_actor(
                camera_blueprint, camera_transform, attach_to=self.follower
            )
            #self.follower_camera.listen(lambda data: process_img(data))

            self.actor_list.append(self.follower_camera)
            
            
            depth_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
            depth_blueprint.set_attribute('image_size_x', '640')
            depth_blueprint.set_attribute('image_size_y', '480')
            depth_blueprint.set_attribute('fov', '110')

            # Set the camera position and attach it to a follower
            camera_transform = carla.Transform(carla.Location(x=2.4, z=1.4))
            self.depth_camera = self.world.spawn_actor(
                depth_blueprint, camera_transform, attach_to=self.follower
            )

            # Append the camera to actor list for cleanup
            self.actor_list.append(self.depth_camera)
            
            
            '''
            '''
            
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')

            # Set attributes
            lidar_bp.set_attribute('range', '50')
            lidar_bp.set_attribute('rotation_frequency', '50')
            lidar_bp.set_attribute('channels', '32')
            #lidar_bp.set_attribute('upper_fov', '10')
            #lidar_bp.set_attribute('lower_fov', '-30')
            lidar_bp.set_attribute('points_per_second', '500000')
            lidar_bp.set_attribute('role_name', 'lidar')
            
            lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.5))  # Set where you want to attach


            self.follower_lidar = self.world.spawn_actor(
                lidar_bp,
                lidar_transform,
                attach_to=self.follower,
                attachment_type=carla.AttachmentType.Rigid
            )
            self.actor_list.append(self.follower_lidar)


            weak_self = weakref.ref(self)
            self.follower_lidar.listen(lambda data: self._parse_lidar1(data))   
        
            '''
            

            #folder = "sequences"

            #if os.path.exists(folder):
            #    shutil.rmtree(folder, ignore_errors=True)  # Deletes folder and contents
            #    print(f"Deleted folder: {folder}")
            
            #output_folder = "C://Users/riahi/carla/CARLA_0.9.15/carla_latest/PythonAPI/examples/sequences"
            #os.makedirs(output_folder, exist_ok=True)
            
            '''
            def save_frame(image):
                
                
                # 1) Process the raw data (e.g., YOLO inference)
                #processed_img = process_img(image)  
                depth_filename = os.path.join(output_folder, f"{self.frame_counter:05d}.jpg")
                #image.tofile(depth_filename)
                image.save_to_disk(depth_filename)

                #image.save_to_disk(f"{output_folder}/{self.frame_counter}.jpg")
                self.frame_counter += 1 
            '''
            '''
            def save_depth(image):
                """Save decoded depth frame from CARLA RGB format (as meters)."""
                depth_data = np.frombuffer(image.raw_data, dtype=np.uint8)
                depth_data = depth_data.reshape((image.height, image.width, 4))  # RGBA

                # Extract RGB channels
                R = depth_data[:, :, 0].astype(np.float32)
                G = depth_data[:, :, 1].astype(np.float32)
                B = depth_data[:, :, 2].astype(np.float32)

                # Decode depth in meters using CARLA formula
                normalized = (R + G * 256.0 + B * 256.0**2) / (256.0**3 - 1)
                depth_in_meters = normalized * 1000.0  # Max depth = 1000m

                # Save depth as .npy file
                depth_filename = os.path.join(output_folder, f"{self.frame_counter:04d}.npy")
                np.save(depth_filename, depth_in_meters)

                # (Optional) Save depth as 16-bit PNG for visualization
                # depth_vis = (depth_in_meters * 1000).astype(np.uint16)  # in millimeters
                # cv2.imwrite(os.path.join(output_folder, f"{self.frame_counter:04d}_depth.png"), depth_vis)

                            
                
            self.follower_camera.listen(save_frame)
            self.depth_camera.listen(save_depth)

            '''
            #self.follower_camera.listen(save_frame)
            
        time.sleep(1) 
        def process_img(image):
            # Convert the raw data to a NumPy array
            i = np.frombuffer(image.raw_data, dtype=np.uint8)

            # Reshape it into an image array (height, width, channels)
            i2 = i.reshape((image.height, image.width, 4))  # Correct shape from CARLA image data  
            # Extract the RGB channels (first three channels)
            i3 = i2[:, :, :3]  # Ignore the alpha channel
            # Convert the color channels from BGRA to RGB for OpenCV
            i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(i3)

            #self.visualize_detections(i3, results)
            #cv2.imshow("Camera Feed", i3)
            #cv2.waitKey(1)
            # Normalize the image for further processing (if needed)
            return i3 / 255.0  # Scaled to [0, 1]


        # Set up the sensors for follower car.
        
        self.collision_sensor = CollisionSensor(self.follower)
        #self.imu_sensor = IMUSensor(self.follower)
        self.camera_manager = CameraManager(self.follower, self._gamma, self.world)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index)
        #time.sleep(3)
        #self.camera_manager.transform_index = cam_pos_index
        #self.camera_manager.set_sensor(cam_index)    
        self.episode += 1     
        
         
        return self.get_state(), {}
    


        
    def get_transformed_lidar_points(self, lidar_data, location, rotation):
        if lidar_data is None:
            return None

        # Step 1: Filter far points 
        #filtered_points = lidar_data[np.linalg.norm(lidar_data, axis=1) < 20.0]  # shape (N, 3)
        
        # Step 2: Get follower's pose
        #transform = transforms
        location = location

        # Step 3: Convert rotation to radians
        pitch = 0
        yaw   = np.deg2rad(rotation)
        roll  = 0

        # Step 4: Build full 3D rotation matrix (roll → pitch → yaw order)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx  # Combined rotation

        # Step 5: Apply rotation and translation to LiDAR points
        rotated = lidar_data @ R.T
        translated = rotated + np.array([[location[0], location[1], location[2]]])

        self.transformed_lidar_points = translated
        return translated

        
   

   


    def merge_clusters(self, clustered_objects, merge_distance=1):
        merged = []
        used = set()

        for i in range(len(clustered_objects)):
            if i in used:
                continue

            center_i = np.array(clustered_objects[i]["center"])
            points_i = clustered_objects[i]["points"]
            merged_cluster = points_i.copy()
            used.add(i)

            for j in range(i + 1, len(clustered_objects)):
                if j in used:
                    continue

                center_j = np.array(clustered_objects[j]["center"])
                points_j = clustered_objects[j]["points"]
                dist = np.linalg.norm(center_i - center_j)

                if dist < merge_distance:
                    merged_cluster = np.vstack((merged_cluster, points_j))
                    used.add(j)

            # Fit rectangle to merged points
            if len(merged_cluster) >= 3:
                rect = cv2.minAreaRect(merged_cluster.astype(np.float32))
                center = rect[0]
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                merged.append({
                    "center": center,
                    "box": box,
                    "angle": rect[2],
                    "points": merged_cluster
                })

        return merged
   
        
    def cluster_points(self, points_2d, follower_rotation_new, follower_location_new, eps=0.7, min_samples=2):
        
    
        if points_2d is None or len(points_2d) == 0:
            return []
        
        R = np.array([
            [math.cos(follower_rotation_new), -math.sin(follower_rotation_new)],
            [math.sin(follower_rotation_new),  math.cos(follower_rotation_new)]
        ])

        # Rotate the points first, then translate
        rotated_points = points_2d[:, [1, 0]] @ R.T  # Rotate the points
        transformed_points = rotated_points + follower_location_new  # Translate to global frame
        
        
        transformed_points = transformed_points[(np.abs(transformed_points[:, 0]) < 6) & (transformed_points[:, 1] > 0)]

        #transformed_points = points_2d[(np.abs(points_2d[:, 0]) < 6) & (points_2d[:, 1] > 0)]
   
        transformed_points = np.atleast_2d(transformed_points)
        
               
        if transformed_points.shape[0] < min_samples:
            return []

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(transformed_points) #was filtered points
      
        labels = clustering.labels_

        clustered_objects = []  # Will hold all rect-based clusters
        centers = []
       
        for label in np.unique(labels):
            
            if label == -1:
                continue  # Noise

            cluster = transformed_points[labels == label] #was filtered points
            if len(cluster) < 3:
                continue  # minAreaRect needs at least 3 points

            # Fit minimum-area bounding box
            rect = cv2.minAreaRect(cluster.astype(np.float32))
            center = rect[0]  # (x, y) of box center
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            centers.append(center)
            clustered_objects.append({
                "center": center,
                "box": box,
                "angle": rect[2],
                "points": cluster
            })

            # Print object info (position relative to LIDAR)
            print(f"[LIDAR] Object at: x = {center[0]:.2f} m, y = {center[1]:.2f} m")

        centers = [obj["center"] for obj in clustered_objects]
        
        #tracker = ObjectTracker()
        tracked_centers = self.tracker.update(centers)
        # Assign tracked centers back
        for i, obj in enumerate(clustered_objects):
            if i < len(tracked_centers):
                obj["tracked_center"] = tracked_centers[i]

            # Print again after merging
            c = obj["center"]
            print(f"[MERGED LIDAR] Object at: x = {c[0]:.2f} m, y = {c[1]:.2f} m")
            
        for obj in clustered_objects:
            x, y = obj["tracked_center"] 
            
            if abs(x) < 0.6 and self.initial_position is None:
                # Store the initial position and time for the first object with x < 0.5
                self.initial_position = y
                print (f"yyyyyyyy  {y}")
                self.initial_time = time.time()
                print(f"Initial Position (at frame 0): {self.initial_position}")
                break  # Only process the first object that meets the condition
            elif self.initial_position is not None: 
                break 

        # After 0.5 seconds, simulate the object detection in the next frame (frame 1)
        for obj in clustered_objects:
            x, y = obj["tracked_center"]
            
            if abs(x) < 0.6 and self.initial_position is not None:
                # Capture the position after 0.5 seconds
                time_difference = time.time() - self.initial_time
                if time_difference >= 0.05: 
                    final_position = y
                    

                    # Calculate the distance traveled (Euclidean distance between initial and final positions)
                    distance_traveled = (final_position - self.initial_position )
                    time_diff1 = time.time() - self.initial_time
                    # Calculate the velocity (distance / delta_t)
                    velocity = (distance_traveled / time_diff1)*3.6

                    print(f"Final Position (after {0.1} seconds): {final_position}")
                    print(f"Distance traveled: {distance_traveled:.2f} meters")
                    print(f"Velocity: {velocity:.2f} km/h")
                    self.initial_position = None
                    break
                else: 
                    break 
            elif self.initial_position is None:
                break
        return clustered_objects
    


        
    def to_image_coords(self, x, y):
        u = int(self.origin[0] + x * 30)
        v = int(self.origin[1] - y * 2)
        return (u, v)
       
        
    def visualize(self, tracked_objects):
        """ Used for visualizing 2D tracks through clustering"""
        # Store history of tracked centers only
        centers = [obj["tracked_center"] for obj in tracked_objects]
        
        
        self.tracked_history.append(centers.copy())
        
        
        if len(self.tracked_history) > 100:
            self.tracked_history.pop(0)

        # Create blank canvas
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255

        # Draw trails (history of centers)
        for past_dets in self.tracked_history:
            for (x, y) in past_dets:
                pt = self.to_image_coords(x, y)
                cv2.circle(img, pt, 2, (200, 200, 255), -1)  # light blue trail

        # Draw each tracked object
        for obj in tracked_objects:
            center = obj["tracked_center"]
            box = obj["box"]

            # Convert to image coords
            center_pt = self.to_image_coords(*center)
            box_pts = [self.to_image_coords(*pt) for pt in box]

            # Draw bounding box
            for j in range(4):
                cv2.line(img, box_pts[j], box_pts[(j + 1) % 4], (0, 255, 0), 2)

            # Draw center point
            cv2.circle(img, center_pt, 6, (0, 0, 255), -1)  # red center

            # Optionally: draw position text
            cv2.putText(
                img,
                f"({center[0]:.1f}, {center[1]:.1f})m",
                (center_pt[0] + 10, center_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        # Draw reference axes
        cv2.line(img, (self.origin[0], 0), (self.origin[0], self.img_size), (0, 0, 0), 1)
        cv2.line(img, (0, self.origin[1]), (self.img_size, self.origin[1]), (0, 0, 0), 1)

        # Show the image
        cv2.imshow("Tracked Objects", img)
        cv2.waitKey(1)



    def override_rl(self, lidar_sectors, rl_action):
        """ Used for overriding reinforcement learning agent after collision prediction"""
        # Thresholds
        front_thresh = 5.0
        side_thresh = 3.0
        emergency_dist = 3.0
        min_dist = 2  # clamp noisy readings

        # Extract key sectors
        d = [max(min_dist, d) for d in lidar_sectors]
        d_front = d[0]
        d_fl = d[7]
        d_fr = d[1]
        d_left = d[6]
        d_right = d[2]

        if d_front < emergency_dist:
            return {'steer': 0.0, 'throttle': 0.0, 'brake': 1.0}  # Full brake

        elif d_front < front_thresh:
            steer = 0.0
            brake = np.clip((front_thresh - d_front) / front_thresh, 0.2, 0.8)
            throttle = 0.0

            # Choose safer direction
            if d_left > d_right and d_left > side_thresh:
                steer = -0.4  # steer left
            elif d_right > d_left and d_right > side_thresh:
                steer = 0.4   # steer right
            else:
                steer = 0.0  # no safe side, stay in lane and brake more

            return {'steer': steer, 'throttle': throttle, 'brake': brake}

        elif d_fl < side_thresh:
            if d_right > side_thresh:
                return {'steer': 0.4, 'throttle': 0.0, 'brake': 1.0}
            else:
                return {'steer': 0.0, 'throttle': 0.0, 'brake': 1.0}

        elif d_fr < side_thresh:
            if d_left > side_thresh:
                return {'steer': -0.4, 'throttle': 0.0, 'brake': 1.0}
            else:
                return {'steer': 0.0, 'throttle': 0.0, 'brake': 1.0}

        else:
            # No override, safe to follow RL
            return rl_action

        
    
    
    
    
    def get_state(self):
        """Updates state information per time step """
        # Get the transforms of both vehicles
        
        follower_transform = self.follower.get_transform()
        player_transform = self.player.get_transform()

        follower_rotation = follower_transform.rotation
        # Get locations
        follower_location = follower_transform.location
        player_location = player_transform.location
        
        if follower_rotation.yaw > 0:
            
            follower_rotation_new =   follower_rotation.yaw + self.reference_orientation
        else:
            follower_rotation_new =   follower_rotation.yaw - self.reference_orientation
        print (f"reference rotatioon and rotation {self.reference_orientation}, {follower_rotation.yaw}")

        print (f"follower rotation true : {follower_rotation_new}")
        
        yaw = np.deg2rad(follower_transform.rotation.yaw)

        
        player_location_new = np.array([-(player_location.y - self.reference_location.y), -(player_location.x - self.reference_location.x)]) 
                                        
        follower_location_new = np.array([-(follower_location.y - self.reference_location.y), -(follower_location.x - self.reference_location.x)]) 
       
        
        print(f"follower location true {follower_location_new}")
        print(f"player location true {player_location_new}")
        
        
        relative_distance = np.linalg.norm(follower_location_new - player_location_new)
          
        R = np.array([
            [math.cos(follower_rotation_new), -math.sin(follower_rotation_new)],
            [math.sin(follower_rotation_new),  math.cos(follower_rotation_new)]
        ])

        #detections = self.cluster_points(points_2d, follower_rotation_new, follower_location_new)
        #trackers = self.update_trackers(self.trackers, detections)  # First frame, no prior trackers

        #self.visualize_frame(points, detections, trackers)
        #print("Tracked objects:", detections)
        #self.visualize(detections)
        
        # Get velocities
        follower_velocity = self.follower.get_velocity()
        player_velocity = self.player.get_velocity()

        follower_speed = np.sqrt(follower_velocity.x**2 + follower_velocity.y**2)*3.6   
        player_speed = np.sqrt(player_velocity.x**2 + player_velocity.y**2)*3.6
        
        print(f"player speed true  {player_speed}")
        
        print(f"follower speed true {follower_speed}")
         
        
        if not hasattr(self, "prev_player"):
            self.prev_player = player_speed
           
        self.prev_player = player_speed
            
        # Compute goal position
        goal_position = self.calculate_goal_position(player_location_new)
        
        self.sector_distances = self.camera_manager.get_sector_distances()        
        goal_distance = abs(follower_location_new[1] -  goal_position[1])
        speed_difference = follower_speed - player_speed
        vertical_distance = (player_location_new[1] - follower_location_new[1])
        horizontal_distance = (player_location_new[0] - follower_location_new[0])
    
        print(f"distance 5 6 7........., {self.sector_distances [5]}, {self.sector_distances [6]}, {self.sector_distances [7]}")

        if vertical_distance > 10:
            phase = [1,0,0,0]
            #player_speed = 0
        elif ((vertical_distance < 10 and vertical_distance > -2) and ((self.sector_distances[5]< 4.5) or (self.sector_distances [6]< 4.5) or (self.sector_distances[7]< 6))):
            phase = [0,1,0,0]
            self.pp = 1
            self.p = 0
            self.vertical = vertical_distance
            self.a= 1
            print(f"vertical distance {self.vertical}")
            
        elif ((vertical_distance < - 5) and ((self.sector_distances[1]> 4) and (self.sector_distances [2]> 4) and (self.sector_distances[3] > 4))):
            phase = [0,0,0,1]
            self.p3 = 1
            self.pp = 0
            self.p = 0
            
        else: 
            phase = [0,0,1,0]
            self.vertical = vertical_distance
            print(f"vertical distance {self.vertical}")
            self.p = 1
            self.p3 = 0
            self.a=1
            
            
        self.step_count += 1 
            

        print(f"follower_rotatation [[[[[]]]]] {follower_rotation_new}")
        
        
        goal_distance = np.array([goal_distance])  # Convert to NumPy array
        follower_speed =np.array ([follower_speed])
        player_speed =np.array ([player_speed])
        horizontal_distance = np.array([horizontal_distance])
        vertical_distance = np.array([vertical_distance])
        follower_rotation_new = np.array([follower_rotation_new])  # Convert rotation to array

        self.loc = follower_location_new [1]
    
        #for logs22
        state = {
            #"Secotr_LiDAR_distances": sector_distances,
            "status ": phase,
            "goal_distance" : goal_distance,
            "follower_speed" : follower_speed, 
            "player_speed" : player_speed,         
            "horizontal_distance" : horizontal_distance,
            "vertical_distance"  : vertical_distance,
            "follower_orientation": follower_rotation_new,
    
        }
        
        # Flatten the state dictionary into a single array
        flat_state = np.concatenate([value for value in state.values()])
        print(f"STATE {flat_state}")
       
        return flat_state




    def calculate_goal_position(self, player_location):
        """
        Calculate the goal position 15 meters ahead of the player vehicle in the same orientation.
        """
        offset_x = 0
        offset_y = 15
        goal_location = np.array([player_location[0] + offset_x, 
                                  player_location[1] + offset_y]) 
        return goal_location
    

    
    def step(self, action): #done):
        """
        Applies the agent's action: acceleration, steering, and braking.
        """
        done = self.done if hasattr(self, "done") else False
        action = torch.FloatTensor(action).to(device)  
        #print(f"Action Tensor: {action}, Shape: {action.shape}")
        '''
        #Potential smoothing 
        
        alpha = self.action_smoothing_alpha
        # --- Smooth throttle ---
        smoothed_throttle = alpha * throttle + (1 - alpha) * self.prev_throttle
        self.prev_throttle = smoothed_throttle

        # --- Smooth brake ---
        smoothed_brake = alpha * brake + (1 - alpha) * self.prev_brake
        self.prev_brake = smoothed_brake

        # --- Smooth steer ---
        smoothed_steer = alpha * steer + (1 - alpha) * self.prev_steer
        self.prev_steer = smoothed_steer
        '''
        # rescale actions
        acceleration = (1 + action[0].item())/2
        steer = action[1].item()  # Steering is already in [-1,1]
        
        print(f"acceleration and steer {acceleration}, {steer}")
                # --- Smooth steer ---
        smoothed_steer = 0.2 * steer + (1 - 0.2) * self.prev_steer
        self.prev_steer = smoothed_steer
  
        if (acceleration < 0.2): 
            throttle = 0
            brake = 1 - acceleration
        else: 
            throttle = acceleration
            brake = 0

        velocity = self.follower.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        
        # Apply control
        control = carla.VehicleControl()
        control.throttle =  float(throttle)  
        control.brake = float(brake)  
        control.steer =  float(steer)
        
        
        
        #if self.p and self.pp:
        #    control.steer = float(steer - 0.2)
        
        
        '''
        self.collision_prediction  = collision_pred()

        rl_actions= [float(steer), float(throttle), float(brake)]
        if self.collision_prediction == 1:
            Action = self.override_rl(self.sector_distances,rl_actions)
                
        '''
        self.state = self.get_state()
        
        velocity1 = self.player.get_velocity()
        total_speed1 = math.sqrt(velocity1.x**2 + velocity1.y**2)*3.6
        
        velocity1 = self.npc1.get_velocity()
        total_speed2 = math.sqrt(velocity1.x**2 + velocity1.y**2)*3.6
        max_speed1 = 40 
        max_speed2 = 70 
        min_speed1 = 20
        min_speed2 = 20
        
        
        target_speed1 = random.uniform(min_speed1, max_speed1)
        target_speed2 = random.uniform(min_speed2, max_speed2)


        control1 = carla.VehicleControl()
        if total_speed1 < 70:
            control1 = carla.VehicleControl(throttle= 0.5 , steer=0.0, brake=0.0)  
        else:
            control1 = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3) 
        self.player.apply_control(control1)
        

        control2 = carla.VehicleControl()
        if total_speed2 < 70:
            control2 = carla.VehicleControl(throttle=0.5 ,steer=0.0, brake=0.0)  # Allow acceleration was 0.5 
        else:
            control2 = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3) 
            
        self.npc1.apply_control(control2)
        self.follower.apply_control(control)
        

        reward, done, truncated = self.compute_reward(self.state)

        # Tick the simulation forward
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        return self.state, reward, done, truncated, {}
    
    
    
    def compute_reward(self, state):
        """Computes the reward based on the state information"""
      
        
        p0 = state[0]
        p1 = state[1]
        p2 = state[2]
        p3 = state[3]
        goal_distance = state [4] 
        follower_speed = state [5]
        player_speed = state[6]
        horizontal_distance = state [7]
        vertical_distance = state [8]    
        follower_orientation = state[9]


        r_goal = 0
        r_stay = 0
        r_collision = 0
        r_alignment=0 
        r_speed = 0
        r_overtaking = 0
        r_time = 0 
        r_progress = 0
        r_distance = 0
        r_accelerate = 0
        r_shift = 0
        risk_assessment  = risk_assess()
        if not hasattr(self, "prev_time"):
            self.prev_time = time.time()
        
        
        if not hasattr(self, "prev_vel"):
            self.prev_vel = follower_speed
            
        if not hasattr(self, "prev_goal"):
            self.prev_goal = goal_distance
            
            
        if not hasattr(self, "prev_p0"):
            self.prev_p0 = 0
            
        time_diff = time.time() - self.prev_time
        self.prev_time = time.time()
        
        #if not hasattr(self, "prev_p"):
        #    self.prev_p = 1
               
               
               
        # =================== collision reward 
        collision_detected = self.check_collision()
        if collision_detected:
            r_collision += -50 
            self.collision_number += 1
            self.done = True
           
        # ========================== goal progress reward 
        r_progress += -(goal_distance - self.prev_goal)      
        self.prev_goal = goal_distance

        # ========================= remaining within speed boundaries
        if follower_speed > 60:
            r_speed -= 0.3 * (follower_speed - 55) 
        elif follower_speed < 20:
            r_speed += -2 #was -4 
            
        '''   
        if risk_assessment == 1:
            r_collision += -10
        else:
            r_collision += 5
            
        ''' 
        
        # ======================== cases for overtaking process
        if p0: # pre-overtaking phase
            #self.prev_p0 += 1
            #if self.prev_p0 > 120: 
            #    r_goal += -1
            if follower_speed > 20:
                r_alignment += max(-0.3, 0.5*(1 - (abs(follower_orientation)/5)))         
                if abs(horizontal_distance) < 0.4:
                    r_stay += 0.6  # keep close to center while moving 
                else: 
                    r_stay += -  0.3* abs(horizontal_distance)          
                #r_speed +=  0.1*(follower_speed - 45)
                r_accelerate += min(0.1 * (follower_speed - self.prev_vel), 0.2)
                self.prev_vel = follower_speed
                #if (p0-self.prev_p) == 1:
                #    r_shift += -2
                    
                    

        if p2:  # overtaking phase: no obstacle
            self.prev_p = 0      
            r_goal += 0.4 #wasn't there  
            
            if (horizontal_distance > 1.25) and (horizontal_distance < 4):
                r_overtaking +=  0.3*min (horizontal_distance, 3.5)  # was 0.3 for logs44 normalized reward for initiating proper overtaking
                r_accelerate += 0.1*(follower_speed - player_speed)
                #r_accelerate += 0.1*(follower_speed - 50) #was -50
                r_alignment += max(-0.5, 1 - (abs(follower_orientation)/5))   

                
                
            else: 
                r_overtaking += -1.0

                    
        if p1: # maintaining phase
            self.prev_p = 0
            r_goal+= 0.4               
            r_stay += - (abs(vertical_distance - 8))  
            r_alignment += max(-0.3, 1 - (abs(follower_orientation)/5))     
            if abs(horizontal_distance) < 0.5: 
                r_stay += 0.7 
            else: 
                r_stay += - 0.5*abs(horizontal_distance)  
                
                  
        if p3:   # returning phase     
            r_overtaking+= 1
            r_alignment += max(-0.3, 1 - (abs(follower_orientation)/5))   
            if abs(horizontal_distance) < 0.6:
                r_stay += 1 
            else: 
                r_stay += - 0.5*abs(horizontal_distance)  #was -0.3 
                
            r_speed += 0.1 *(follower_speed - player_speed)
            
        
        # ======================= time delay reward
        elapsed_time = time.time() - self.episode_start_time
        if elapsed_time > 30:
            self.done = True
            self.truncated = True
            r_time -= 10 # time penalty
            print("Episode ended due to time limit.")
            
        # ======================= reaching goal reward
        if ((goal_distance < 3) and (abs(horizontal_distance) < 0.6)):
            r_goal += 70  
            self.done = True
            print ("episode done because gaol reached")

        if self.step_count > 300:
            self.done = True
        # Clip final reward to prevent extreme values
        total_reward = r_goal + r_collision + r_alignment + r_stay + r_overtaking + r_speed + r_time + r_progress + r_distance + r_accelerate + r_shift
            
        
        print(f"Reward breakdown state '{p0,p1,p2,p3}': Goal={r_goal}, alignment={r_alignment},  Collision={r_collision}, Speed={r_speed}, r_overtaking ={r_overtaking}, r_progress ={r_progress}, r_stay ={r_stay}, r_accelerate = {r_accelerate}, r_distance = {r_distance}, r_shift = {r_shift}")
        
        print(f" step reward episode {self.episode} ====================================== {total_reward}")
    
        return total_reward, self.done, self.truncated
    

    def check_collision(self):
        """Check if the follower vehicle has collided with another actor."""
        if self.collision_sensor is not None:
            return self.collision_sensor.has_collided()
        return False
    

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Dsteroys spawned actors at the end of an episode"""
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            #self.camera_manager.sensor,
            self.follower_camera, 
            self.collision_sensor.sensor,
            self.follower_lidar]
            #self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
                sensor = None

        if self.player is not None:
            self.player.destroy()
            self.player = None

        if self.follower is not None: 
            self.follower.destroy()
            self.follower = None
            
        if self.npc1 is not None: 
            self.npc1.destroy()
            self.npc1 = None