# Project Summary
This repository contains a **full ROS2 package for controlling a TurtleBot3 Waffle** to:
* autonomously explore a 4x4 meter arena with obstacles,
* visit twelve outer zones,
* and produce a SLAM map

... all within a 90-second time budget, with no prior knowledge of the obstacle layout.

The robot is dropped into the 4×4 m arena divided into a 4×4 grid of 1 m² zones. The twelve perimeter ("outer") zones must each be visited at least once. The arena contains internal obstacles whose positions are not known in advance, so the robot must build a map of its surroundings online while simultaneously planning paths through it.

Video link: https://youtu.be/aWCGr5v9SjY

This work was completed as part of project work in the ELE434 (Mobile Robotics) module at the University of Sheffield (2025-26). 

## Approach

The package implements a layered autonomy stack:

* SLAM runs Cartographer on the Waffle's LDS-02 LIDAR to build the map and publish the ```map``` → ```odom``` transform.
* Nav2 handles path planning and path following, with the costmaps fed by both the laser scans and the live SLAM map.
* A custom strategic layer (```frontier_nav.py```) sits on top of Nav2. It scores all outer zones at each decision point — combining distance, heading alignment, and outer-zone bonus — and dispatches goals to Nav2 to visit the highest-scoring zone next. When the robot passes through any zone en route to its current goal, the visit is detected and the controller is preempted with a new goal so no time is wasted finishing a now-redundant trip.


## Architecture
```
┌────────────────────────────────────────────────────────────────┐
│  frontier_nav.py        ← strategic layer (this package)       │
│  • Zone scoring & selection                                    │
│  • Visit detection via TF (map → base_footprint)               │
│  • Preemption on en-route visits                               │
│  • Map save at t = max_runtime                                 │
└────────────────────────────────────────────────────────────────┘
                │  NavigateToPose action
                ▼
┌────────────────────────────────────────────────────────────────┐
│  Nav2 stack                                                    │
│  • bt_navigator → planner_server (NavFn)                       │
│  • controller_server (Regulated Pure Pursuit)                  │
│  • global & local costmaps (obstacle + inflation layers)       │
└────────────────────────────────────────────────────────────────┘
                │  /cmd_vel_nav  (Twist)
                ▼
┌────────────────────────────────────────────────────────────────┐
│  cmd_vel_relay.py  ← Twist → TwistStamped conversion           │
└────────────────────────────────────────────────────────────────┘
                │  /cmd_vel  (TwistStamped)
                ▼
        TurtleBot3 Waffle hardware

       ▲                              ▲
       │                              │
   /scan (LIDAR)             map ← Cartographer SLAM
```

## Repository Layout
```
ele434_team07_2026/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── nav2_params.yaml          ← Nav2 tuning (costmap, controller, smoother)
├── launch/
│   └── explore.launch.py         ← Brings up SLAM, Nav2, relay, and strategy node
├── ele434_team07_2026_modules/
│   ├── frontier_nav.py           ← Strategic layer
│   └── cmd_vel_relay.py          ← Twist → TwistStamped converter
└── maps/                         ← Saved SLAM maps land here
```

## Stack
* ROS 2 Jazzy
* Zenoh middleware (real robot) / DDS (simulation)
* Cartographer SLAM
* Nav2 (NavFn planner, Regulated Pure Pursuit controller)
* Python (tf2_ros, rclpy)
* TurtleBot3 Waffle — hardware information at https://www.turtlebot.com/turtlebot3/
