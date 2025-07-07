"""
sensors.py
----------
CARLA sensor classes used by the World environment. Includes collision detection,
IMU measurements, and RGB/LiDAR data streaming and sector-based processing.

Classes:
    - CollisionSensor: Handles vehicle collision events and history.
    - IMUSensor: Reads accelerometer, gyroscope, and compass values.
    - CameraManager: Manages RGB and LiDAR sensors, and processes LiDAR data.
"""

import carla
import math
import weakref
import numpy as np
import collections
import time
import threading
from carla import ColorConverter as cc










# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class CollisionSensor(object):
    """
    A CARLA sensor that detects and stores collision events for the attached actor.
    """
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        #self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    def has_collided(self):
        """
        Check if a collision has occurred based on the history.
        
        Returns:
            True if there is at least one collision in the history, False otherwise.
        """
        return len(self.history) > 0

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        #self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)



# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================

class IMUSensor(object):
    """
    A CARLA sensor that captures IMU (Inertial Measurement Unit) data:
    accelerometer, gyroscope, and compass.
    """
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))
        
        self.latest_imu=[]
        #self.velocity = np.zeros(3)  # (vx, vy, vz)
        #self.orientation = np.eye(3)  # Rotation matrix (3x3)
        self.last_time = time.time()  # Store last update time

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
                # Call method to update position and velocity
                
        self.latest_imu ={
        "accel": self.accelerometer,
        "guro": self.gyroscope,
        "compass" : self.compass
            
        }
        
        return self.latest_imu
                
        '''              
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:  # Prevent division by zero
            return
        self.last_time = current_time
        
        # Update orientation using gyroscope data
        delta_rotation = R.from_euler('xyz', self.gyroscope * dt, degrees=False).as_matrix()
        self.orientation = np.dot(self.orientation, delta_rotation)
        # Integrate acceleration to estimate velocity
        self.velocity += self.accelerometer * dt
        print(f"IMU Update: dt={dt:.4f}s | Velocity: {self.velocity} | Orientation Matrix:\n{self.orientation}")
        '''
    def get_imu_latest(self):
        return self.latest_imu
    
   

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
from sklearn.decomposition import PCA
from types import SimpleNamespace
import yaml

import torch


class CameraManager(object):
    """
    Handles RGB camera and LiDAR sensors in CARLA for a given vehicle.
    Includes setup, transformation, threading, and sector-based LiDAR processing.
    """
    def __init__(self, parent_actor, gamma_correction, world):
        self.camerasensor = None
        self.device = 'cpu'  # Ensure you're using CPU
        
        self.world = world
        self.lidarsensor = None
        self.surface = None
        self._parent = parent_actor
        
        self.timer = 0
        follower_transform = self._parent.get_transform()
        
        follower_location = follower_transform.location
        follower_rotation = follower_transform.rotation
        
        
        if not hasattr(self, "prev_ref"):
            self.reference_location = follower_location
            self.reference_orientation = follower_rotation.yaw
            self.prev_ref = True 
            
        self.tracks = []
        self.next_id = 0
        

        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType


        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                #(carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=0.0, y=0.0, z = 1.5)), Attachment.Rigid)  # Lidar on top

                ]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0.0, y=0.0, z = 1.5)), Attachment.Rigid)
                ]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.lidar.ray_cast', None, '2D-LiDAR', {'range': '50', 
            'rotation_frequency': '100',  # 100 Hz LiDAR
            'channels': '1',  # was 1Single plane LiDAR
            'upper_fov': '0',  # No was 0 vertical spread
            'lower_fov': '0',
            'points_per_second': '400000'  # was 400k Typical 2D LiDAR
                                                                                                                                                                
            }]  # Lidar sensor

            #['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            #['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}], 
             ]
        #print(f"All sensors: {self.sensors}")

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('role_name', 'lidar')  # ✅ Enables ROS to pick up the topic

            if item[0].startswith('sensor.lidar'):
                self.lidar_range = 50        

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None
        self.tracked_position = None  # (angle, distance)
        self.lost_count = 0  # Track how long the target has been missing
        self.distance_threshold = 2.0
        self.angle_threshold = 10.0
        self.tracking_threshold = 1.0
        self.tracked_object = None

        self.x_threshold = 0.5  # threshold distance to track the object around y=0
        self.y_threshold = 0.2
        self.initialized = False

        self.lidar_data = None
        self.sector_distances = np.full(8, 30.0)
        self.sensor_thread = None
        self.sensor_thread_event = threading.Event()
        self.sensor_lock = threading.Lock()  # Ensure thread safety


        self.global_transformation = np.eye(4)  # Initial pose
        self.prev_pcd = None  # Store previous frame
        self.latest_lidar = None
        
        self.prev_time = time.time()
        self.initial_pos = None
        self.position_now = None
        

    def set_sensor(self, index, force_respawn=False):
        self.initial_detection_done = False
        self.tracked_position = None
     
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.camerasensor is not None:
                self.camerasensor.destroy()
                self.surface = None
            if self.lidarsensor is not None:
               self.lidarsensor.destroy()

               self.surface = None

            self.lidarsensor = self._parent.get_world().spawn_actor(
                #self.sensors[index][-1],
                self.sensors[1][-1],
                self._camera_transforms[1][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[1][1])


            weak_self = weakref.ref(self)
            print ("lidar being set")
            # add more callbackls for other sensors

            self.lidarsensor.listen(lambda image: CameraManager._parse_imagelidar(weak_self, image))

        self.index = index
    


    def _parse_imagelidar(weak_self, image):
        """
        Callback function for the LiDAR sensor. Converts raw data to (x, y) point cloud
        and computes sector distances.
        """
        self = weak_self()
        if not self:
            return
            
        points = np.frombuffer(image.raw_data, dtype=np.float32).reshape(-1,4)
        valid_points = points[np.isfinite(points).all(axis=1)]
        
        self.lidar_data = valid_points[:, :2]
        self.sector_distances = self.compute_sector_distances(self.lidar_data)
        

    def get_transformed_lidar_points(self, lidar_data, location, rotation):
        if lidar_data is None:
            return None

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

    def compute_sector_distances(self, lidar_data):
        """
        Divide the LIDAR readings into 8 sectors (each 45°):
        - Sector 0: Front (-22.5° to +22.5°)
        - Sector 1: Front-Right (+22.5° to +67.5°)
        - Sector 2: Right (+67.5° to +112.5°)
        - Sector 3: Rear-Right (+112.5° to +157.5°)
        - Sector 4: Rear (+157.5° to -157.5°)
        - Sector 5: Rear-Left (-157.5° to -112.5°)
        - Sector 6: Left (-112.5° to -67.5°)
        - Sector 7: Front-Left (-67.5° to -22.5°)
        """
        if lidar_data is None or len(lidar_data) == 0:
            print("no lidar data")
            return np.full(8, 30.0)  # No LIDAR data, return infinite distances
        
        angles = np.arctan2(lidar_data[:, 1], lidar_data[:, 0])  # Compute angles
        distances = np.linalg.norm(lidar_data[:, :2], axis=1)  # Compute distances
        
        # Define sector masks
        sector_masks = [
            (angles >= -np.pi/8) & (angles < np.pi/8),  # Front
            (angles >= np.pi/8) & (angles < 3*np.pi/8),  # Front-Right
            (angles >= 3*np.pi/8) & (angles < 5*np.pi/8),  # Right
            (angles >= 5*np.pi/8) & (angles < 7*np.pi/8),  # Rear-Right
            (angles >= 7*np.pi/8) | (angles < -7*np.pi/8),  # Rear
            (angles >= -7*np.pi/8) & (angles < -5*np.pi/8),  # Rear-Left
            (angles >= -5*np.pi/8) & (angles < -3*np.pi/8),  # Left
            (angles >= -3*np.pi/8) & (angles < -np.pi/8)   # Front-Left
        ]
        
        # Compute minimum distance in each sector
        
        sector_distances = [
            np.min(distances[mask]) if np.any(mask) else 30.0 for mask in sector_masks
        ]
        
        return np.array(sector_distances)

    def get_sector_distances(self):
        """
        Returns the latest sector distances computed from LiDAR.

        Returns:
            np.ndarray: Array of 8 distances.
        """
        #print("collecting data....")
        #time.sleep(1)
        return self.sector_distances
    def get_lidar_points(self):
        print(" ======= this is for tracking ")
        return self.lidar_data
