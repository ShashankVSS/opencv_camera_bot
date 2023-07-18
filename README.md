## OpenCV Optical Flow for Velocity Determination

This Gazebo sim utilizes OpenCV and Gazebo along with ROS2 Foxy to simulate robot movement along with determining the average velocities of specific objects in the robots FOV.

### Requirements:
```
rclpy
numpy
cv2
cv_bridge
sensor_msgs
std_msgs
geometry_msgs
```

### Running Commands
In 3 separate terminals:
```
ros2 launch camera_bot launch_sim.launch.py world:=./src/camera_bot/worlds/obstacles.world
ros2 run teleop_twist_keyboard teleop_twist_keyboard
ros2 launch camera_bot opencv_pub.launch.py
```

To view published data: 
```
ros2 topic echo /x_velocity
ros2 topic echo /y_velocity
```
