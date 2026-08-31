# Autonomous Inspection Robot with Real-time Defect Detection

An autonomous robot that patrols a simulated environment using ROS2 Nav2, detects defects in real time using computer vision, and reports the location of any anomalies found.

## Workspace Structure

```
inspection_ws/
├── src/
│   ├── inspection_interfaces/   # srv/msg definitions (e.g. DetectDefect.srv)
│   └── inspection_robot/        # nodes: perception, navigation, defect_detector_server
├── launch/                      # launch files
├── config/                      # nav2 params, costmap config
├── docs/                        # project docs, diagrams
└── README.md
```

## Requirements

- Ubuntu 22.04
- ROS2 Humble
- Nav2
- Python 3.10+
- Ultralytics YOLO
- (for edge deployment) TensorRT, ONNX

## Installation

```bash
# clone repo
git clone <repo_url> inspection_ws
cd inspection_ws

# install dependencies
rosdep install --from-paths src --ignore-src -r -y

# build workspace
colcon build --symlink-install
source install/setup.bash
```

## Usage

```bash
# 1. launch simulation
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 2. launch Nav2 stack
ros2 launch nav2_bringup bringup_launch.py map:=config/map.yaml

# 3. launch the full system (navigation + perception)
ros2 launch inspection_robot full_system.launch.py
```

## Service Interface

`/detect_defect` — takes an image and returns the defect type, bounding box, confidence score, and the location where it was found.

Full definition at `src/inspection_interfaces/srv/DetectDefect.srv`

## Author

Pongsakorn Srithong (Sprite)
AI Engineering and Data Science, Bangkok University (Rangsit)
