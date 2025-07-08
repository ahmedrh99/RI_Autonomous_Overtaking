#  Risk-Informed Autonomous Overtaking

This project implements a **risk-aware overtaking system** for autonomous driving in both **simulation** (CARLA) and **real-world** (Quanser QCar). The system integrates RoadScene2Vec as a risk assessment module to the reinforcement learning trained policy (PPO). 

##  Project Structure

```
RI_Autonomous_overtaking/
├── CARLA/                # Simulation environment
│   ├── env/              # Environment logic (sensors, world)
│   ├── agent/            #  agent (DDPG)
│   ├── main.py           # Simulation entry point
│   ├── train.py          # PPO training script
│   ├── test.py           # Evaluation script
│   └── Risk/             # Risk assessment module
│       └── collision_prediction/
│       └── risk_assessment/
│       └── visualization/
        __
├── QCar/                 # Real-world QCar implementation
│   ├── qcar/             # ROS nodes for control and sensing
│   ├── rl_qcar_controller/
│   └── Risk/             # Optional risk assessment (QCar side)
├── Risk_assessment/      # RoadScene2Vec network and risk models
└── docs/                 # Auto-generated documentation (via pdoc)
```

##  Features

-  **Autonomous overtaking** in complex mixed-traffic scenarios
-  **Sensor integration** (LiDAR, RGB, IMU, etc.)
-  **Reinforcement learning agent** (PPO) trained in CARLA
-  **Multi-object tracking** using Kalman Filter & Hungarian algorithm
-  **Scene graph-based risk prediction** using RoadScene2Vec
-  **Real-time deployment** on Quanser QCar (ROS 2)

##  Simulation (CARLA)

- **Simulator:** CARLA 0.9.15
- **RL Agent:** PPO with environment reward shaping
- **Sensors:** LiDAR (sectorized), RGB-D, IMU, Collision
- **Risk Model:** Predicts binary SAFE/UNSAFE class  or Collision/No Collision based on classification task
- **Requirement:** RoadScene2Vec https://github.com/AICPS/roadscene2vec 
- **Execution:**  
  ```bash
  cd CARLA
  python main.py      # For simulation
  python train.py     # Train PPO
  python test.py      # Evaluate agent
  ```

##  Real-World Deployment (QCar)

- **Platform:** Quanser QCar
- **Framework:** ROS 1/2
- **Perception:** RGB-D + LiDAR 
- **Controller:** QCar RL node executes actions from PPO policy
- **Execution:**  
  Launch nodes from QCar/ directory using ROS launch files.

##  Risk Assessment Module

- **Model:** RoadScene2Vec pretrained graph neural network
- **Input:** Scene graphs extracted from simulated frames or QCar Camera topic
- **Prediction:** Binary classification of collision risk
- **Execution:**
  ```bash
  cd CARLA/Risk/collision_prediction
  python coll_prediction.py
  ```


##  Tracker Module

### Requirements
- ROS 2 Humble
- Autoware Universe (must include:
  `autoware_lidar_centerpoint`, `autoware_multi_object_tracker`)
- Carla ROS Bridge
- CUDA-enabled GPU for CenterPoint inference
- Python 3.8+, PyTorch, Torch Geometric

- **Input:** Lidar points, IMU, velocity encoder and RGB camera
- **Detection:** Centerpoint for LiDAR and YOLO for Camera
### Execution 
  ```bash
  cd CARLA/tracking/ROS2/carla_ros_bridge/launch/
  ros2 launch carla_ros_bridge Multi_object_tracker.launch.py
  ```
