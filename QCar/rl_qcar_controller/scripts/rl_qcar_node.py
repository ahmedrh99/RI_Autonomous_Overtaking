#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_quaternion

from stable_baselines3 import PPO


class RLQCarControllerSB3:
    """
    ROS1 node that:
      - Loads a pre-trained SB3 PPO policy.
      - Subscribes to LaserScan (for 8-sector LiDAR) and two PoseStamped topics
        (follower and player from OptiTrack).
      - Computes a 17-dimensional “state” exactly as in the CARLA get_state( ) method,
        scaled for QCar’s 0.35 m lane (vs. CARLA’s 3.5 m).
      - Feeds the state into the PPO model to get (throttle, steer) ∈ [–1,1].
      - Maps those to a geometry_msgs/Twist on /cmd_vel (max linear 0.2 m/s, max yaw 1.0 rad/s).
    """

    def __init__(self):
        # ----------------------------------------
        # 1) ROS PARAMETERS (set via rosparam or launch file)
        # ----------------------------------------
        #
        #   ~model_path           : path to your saved SB3 PPO model (e.g. “ppo_carla_qcar.zip”)
        #   ~max_linear_speed     : QCar’s max forward speed [m/s] (default 0.2)
        #   ~scan_topic           : LaserScan topic (default “/scan”)
        #   ~follower_pose_topic  : OptiTrack PoseStamped for follower (ego) car
        #   ~player_pose_topic    : OptiTrack PoseStamped for player (target) car
        #   ~control_topic        : where to publish Twist (default “/cmd_vel”)
        #
        self.model_path = rospy.get_param("~model_path")
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.2)
        self.scan_topic = rospy.get_param("~scan_topic", "/qcar1/scan")
        self.follower_pose_topic = rospy.get_param("~follower_pose_topic", "/natnet_ros/qcar1/pose")
        self.player_pose_topic = rospy.get_param("~player_pose_topic", "/natnet_ros/qcar2/pose")
        self.control_topic = rospy.get_param("~control_topic", "/cmd_vel")

        # ----------------------------------------
        # 2) LOAD THE SB3 PPO MODEL (inference-only)
        # ----------------------------------------
        try:
            self.model = PPO.load(self.model_path)
            rospy.loginfo(f"[RLQCarControllerSB3] Loaded SB3 PPO model from '{self.model_path}'")
        except Exception as e:
            rospy.logerr(f"[RLQCarControllerSB3] Failed to load SB3 PPO model: {e}")
            rospy.signal_shutdown("Could not load SB3 PPO model.")
            return

        # ----------------------------------------
        # 3) INTERNAL STATE VARIABLES
        # ----------------------------------------
        self.latest_scan = None                   # sensor_msgs/LaserScan
        self.follower_pose = None                 # tuple (x, y, yaw)
        self.player_pose = None                   # tuple (x, y, yaw)
        self.reference_set = False                # True once we record ref_follower_pos/yaw
        self.ref_follower_pos = np.zeros(2)       # [x_ref, y_ref]
        self.ref_follower_yaw = 0.0               # yaw_ref
        # For finite-difference speed estimate at 10 Hz:
        self._prev_f_rel = None                   # numpy [2,]
        self._prev_p_rel = None                   # numpy [2,]

        # ----------------------------------------
        # 4) ROS SUBSCRIBERS
        # ----------------------------------------
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.follower_pose_topic, PoseStamped, self.follower_pose_callback, queue_size=1)
        rospy.Subscriber(self.player_pose_topic, PoseStamped, self.player_pose_callback, queue_size=1)

        # ----------------------------------------
        # 5) ROS PUBLISHER (cmd_vel)
        # ----------------------------------------
        self.cmd_pub = rospy.Publisher(self.control_topic, Twist, queue_size=1)

        # ----------------------------------------
        # 6) START THE CONTROL LOOP AT 10 Hz
        # ----------------------------------------
        self.control_rate = rospy.Rate(20)
        rospy.loginfo("[RLQCarControllerSB3] Initialization complete. Entering control loop...")
        self._control_loop()

    # -----------------------------------------------------------------------------
    # SECTION A: ROS CALLBACKS
    # -----------------------------------------------------------------------------

    def scan_callback(self, msg: LaserScan):
        """Cache the latest LaserScan message."""
        self.latest_scan = msg

    def follower_pose_callback(self, msg: PoseStamped):
        """
        Cache the follower’s PoseStamped, which arrives as:
        msg.pose.position.{x,y,z} and msg.pose.orientation as a quaternion (x,y,z,w).
        We convert the quaternion to Euler (roll, pitch, yaw) via tf.transformations.euler_from_quaternion,
        then discard roll/pitch and keep yaw. On the *first* follower callback, record that
        (x, y, yaw) as the reference frame.
        """
        # 1) Extract position
        x = msg.pose.position.x
        y = msg.pose.position.y

        # 2) Extract quaternion
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        # 3) Convert quaternion → Euler (roll, pitch, yaw)
        (roll, pitch, yaw) = euler_from_quaternion([qx, qy, qz, qw])
        # We only care about yaw for planar locomotion.

        # 4) On the very first callback, record reference
        if not self.reference_set:
            self.ref_follower_pos[0] = x
            self.ref_follower_pos[1] = y
            self.ref_follower_yaw = yaw
            self.reference_set = True
            rospy.loginfo(
                "[RLQCarControllerSB3] Reference follower position/yaw set."
            )

        # 5) Store the follower pose as (x, y, yaw)
        self.follower_pose = (x, y, yaw)

    def player_pose_callback(self, msg: PoseStamped):
        """
        Cache the player’s PoseStamped. We do not need a separate reference for player,
        because state uses only ref_follower for both.
        """
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.player_pose = (x, y, yaw)

    # -----------------------------------------------------------------------------
    # SECTION B: STATE CONSTRUCTION (mirroring CARLA’s get_state, scaled for QCar)
    # -----------------------------------------------------------------------------

    def _compute_sector_distances(self):
        """
        Convert the latest LaserScan into 8 sector distances (45° each). Returns a list of 8 floats.
        Sector indexing:
          0: [−22.5°, +22.5°]
          1: (22.5°, 67.5°], 2: (67.5°, 112.5°], 3: (112.5°, 157.5°]
          4: (157.5°, 180°] ∪ (−180°, −157.5°]
          5: (−157.5°, −112.5°], 6: (−112.5°, −67.5°], 7: (−67.5°, −22.5°]
        If no valid reading in a sector, use scan.range_max.
        """
        scan = self.latest_scan
        if scan is None:
            return None

        ranges = np.array(scan.ranges)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        num_readings = len(ranges)

        # Compute angle for each range index
        angles = angle_min + np.arange(num_readings) * angle_inc  # radians

        sector_distances = []
        for s in range(8):
            if s == 4:
                # Wrap-around sector: [157.5°, 180°] ∪ [−180°, −157.5°]
                idx1 = np.where((angles >= math.radians(157.5)) & (angles <= math.pi))[0]
                idx2 = np.where((angles >= -math.pi) & (angles <= math.radians(-157.5)))[0]
                idx = np.hstack((idx1, idx2))
            else:
                # Determine low/high bounds in degrees
                # We’ll use a small lookup, skipping the wrap-around sector
                if s == 0:
                    low_deg, high_deg = -22.5, 22.5
                elif s == 1:
                    low_deg, high_deg = 22.5, 67.5
                elif s == 2:
                    low_deg, high_deg = 67.5, 112.5
                elif s == 3:
                    low_deg, high_deg = 112.5, 157.5
                elif s == 5:
                    low_deg, high_deg = -157.5, -112.5
                elif s == 6:
                    low_deg, high_deg = -112.5, -67.5
                elif s == 7:
                    low_deg, high_deg = -67.5, -22.5
                else:
                    # Should never happen (s=4 handled above)
                    low_deg, high_deg = 0.0, 0.0

                low = math.radians(low_deg)
                high = math.radians(high_deg)
                idx = np.where((angles > low) & (angles <= high))[0]

            if idx.size == 0:
                # No readings → assume max range
                sector_distances.append(scan.range_max)
            else:
                valid_vals = ranges[idx]
                valid_vals = valid_vals[np.isfinite(valid_vals)]
                if valid_vals.size == 0:
                    sector_distances.append(scan.range_max)
                else:
                    sector_distances.append(float(np.min(valid_vals)))


        print("Sector Distances [0:Front → 7:Front-Left]:")
        print(["{:.2f}".format(d) for d in sector_distances])
        return sector_distances  # length 8

    def _compute_state(self):
        """
        Build a 17-dimensional “flat_state” exactly as your CARLA get_state( ) did, but:
          • Distances scaled by lane_scale = 0.35/3.5 = 0.10
          • Speeds estimated by finite difference at 10 Hz
          • Only reference is ref_follower_pos/yaw
        Returns: numpy.ndarray of shape (17,) or None if data not ready.
        """
        # Ensure we have everything
        if (not self.reference_set) or (self.latest_scan is None) \
        or (self.follower_pose is None) or (self.player_pose is None):
            return None

        # 1) Extract raw poses
        fx, fy, fyaw = self.follower_pose
        px, py, pyaw = self.player_pose

        # 2) Compute global‐frame deltas from the follower reference
        delta_fx = fx - self.ref_follower_pos[0]
        delta_fy = fy - self.ref_follower_pos[1]

        delta_px = px - self.ref_follower_pos[0]
        delta_py = py - self.ref_follower_pos[1]

        # 3) Build the 2×2 rotation matrix R(-ref_yaw) 
        #    so that we rotate global deltas into follower‐local frame:
        cos_r = math.cos(self.ref_follower_yaw)
        sin_r = math.sin(self.ref_follower_yaw)
        R_inv = np.array([
            [ cos_r,  sin_r],
            [-sin_r,  cos_r]
        ])

        # 4) Rotate both follower and player deltas into follower‐local frame
        f_local = R_inv.dot(np.array([delta_fx, delta_fy]))
        p_local = R_inv.dot(np.array([delta_px, delta_py]))

        f_rel_x_local = f_local[0]  # should be 0.0 if we always treat follower as origin
        f_rel_y_local = f_local[1]  # should be 0.0 as well
        print (f"follower pose y {f_rel_x_local}")

        p_rel_x_local = p_local[0]  # lateral offset of player in follower‐local
        p_rel_y_local = p_local[1]  # forward offset of player in follower‐local
        print (f"player pose y {p_rel_x_local}")

        # 5) Now horizontal & vertical distances (player w.r.t. follower) are simply:
        horizontal_distance = p_rel_y_local - f_rel_y_local
        vertical_distance   = p_rel_x_local - f_rel_x_local
        print(f"vertical distance {vertical_distance}")
        

        # 6) Rotation offset for the follower itself (used later if you need it)
        f_rot = fyaw - self.ref_follower_yaw  # unchanged from before, in radians




        # 5) Speeds via finite difference (10 Hz)
        if (self._prev_f_rel is None) or (self._prev_p_rel is None):
            follower_speed = 0.0
            player_speed = 0.0
        else:
            dt = 1.0 / 20.0  # control loop runs at 10 Hz
            follower_speed = np.linalg.norm(np.array([f_rel_x_local, f_rel_y_local]) - self._prev_f_rel) / dt
            player_speed = np.linalg.norm(np.array([p_rel_x_local, p_rel_y_local]) - self._prev_p_rel) / dt

        # Store for next iteration
        self._prev_f_rel = np.array([f_rel_x_local, f_rel_y_local])
        self._prev_p_rel = np.array([p_rel_x_local, p_rel_y_local])

        # 6) Compute goal distance:
        #    In CARLA, goal_position = calculate_goal_position(player_location_new)
        #    We infer: goal is a fixed gap behind player (5 m in CARLA → 0.5 m in QCar).
        lane_scale = 0.35 / 3.5  # = 0.10
        desired_carla_follow_gap = 5.0
        desired_qcar_follow_gap = desired_carla_follow_gap * lane_scale  # = 0.5 m
        goal_y = p_rel_y_local + 1.5
        goal_distance = abs(f_rel_y_local - goal_y)

        # 7) Determine phase (4-dim one-hot) with scaled thresholds:
        #    CARLA: if vertical_distance > 10  → pre-overtake (phase[1,0,0,0])
        #           elif vertical ∈ (−2,10) & (sector5<4.5 or sector6<4.5 or sector7<6) → overtaking (phase[0,1,0,0])
        #           elif vertical < −5 & side clear → returning (phase[0,0,0,1])
        #           else → cruising (phase[0,0,1,0]).
        th1 = 10.0 * lane_scale   # 10 m → 1.0 m
        th2 = -2.0 * lane_scale   # -2 m → -0.2 m
        th3 = -5.0 * lane_scale   # -5 m → -0.5 m

        sector_distances = self._compute_sector_distances()
        if sector_distances is None:
            return None

        s5 = 4.5 * lane_scale    # 4.5 m → 0.45 m
        s6 = 4.5 * lane_scale    # 0.45 m
        s7 = 6.0 * lane_scale    # 6 m → 0.6 m

        # Initialize phase
        phase = [0, 0, 0, 0]
        if vertical_distance > 1.0:
            phase = [1, 0, 0, 0]
        elif (vertical_distance <= 1.0 and vertical_distance > -0.2) and ((sector_distances[5] < 0.45) or (sector_distances[4] < 0.45) or (sector_distances[3] < 0.6)):
            phase = [0, 1, 0, 0]
        elif (vertical_distance < -0.5) and ((sector_distances[1] > 0.4) and (sector_distances[0] > 0.4 ) and (sector_distances[7] > 0.4)):
            phase = [0, 0, 0, 1]
        else:
            phase = [0, 0, 1, 0]

        # 8) Build the final 17-dimensional state in EXACT order used during SB3 training:
        #    [phase0, phase1, phase2, phase3,
        #     goal_distance,
        #     follower_speed,
        #     player_speed,
        #     horizontal_distance,
        #     vertical_distance,
        #     follower_orientation,
        #     sector_5, sector_6, sector_7,
        #     sector_0, sector_1, sector_2, sector_3]
        final_state = []
        # 8.1 Phase (4 dims)
        final_state.extend(phase)  # indices 0..3
        print (f"phase {phase}")
        # 8.2 Six scalars
        final_state.append(goal_distance)       # idx 4
        final_state.append(follower_speed)      # idx 5
        final_state.append(player_speed)        # idx 6
        final_state.append(horizontal_distance) # idx 7
        final_state.append(vertical_distance)   # idx 8
        final_state.append(f_rot)               # idx 9



        # Convert to numpy array of dtype float32
        flat_state = np.array(final_state, dtype=np.float32)

        return flat_state

    # -----------------------------------------------------------------------------
    # SECTION C: CONTROL LOOP (10 Hz) → model.predict → publish Twist
    # -----------------------------------------------------------------------------

    def _control_loop(self):
        while not rospy.is_shutdown():
            flat_state = self._compute_state()
            if flat_state is not None:
                # SB3 expects either (n_dims,) or (1, n_dims). Passing (17,) is fine for a single sample.
                action, _ = self.model.predict(flat_state, deterministic=True)
                # action is a numpy array of shape (2,): [throttle, steer], each ∈ [–1,1]

                # Map throttle to forward velocity
                raw_throttle = float(action[0])
                acceleration = (1 + action[0].item())/2
                
                if acceleration < 0.2:
                    linear_vel = 0.0
                else:
                    linear_vel = raw_throttle * self.max_linear_speed

                # Map steer [–1,1] to yaw rate [–max_yaw_rate, +max_yaw_rate]
                max_yaw_rate = 1.0  # rad/s (tune on your QCar)
                angular_vel = float(action[1]) * max_yaw_rate
                
                print (f"raw_trottle {acceleration}, {angular_vel}")

                # Publish Twist
                cmd = Twist()
                cmd.linear.x = 0.0 #linear_vel
                cmd.angular.z = 0.5 #angular_vel
                self.cmd_pub.publish(cmd)

            # Sleep until next cycle
            self.control_rate.sleep()


if __name__ == "__main__":
    rospy.init_node("rl_qcar_controller_sb3")
    try:
        RLQCarControllerSB3()
    except rospy.ROSInterruptException:
        pass
