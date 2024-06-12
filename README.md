# diff_opt_tracking

This repository is associated with our RAL submission paper: "Differentiable-Optimization Based Neural Policy for Occlusion-Aware Target Tracking".

# Dependencies

* [JAX](https://github.com/google/jax)
* [bebop_simulator](https://github.com/Houman-HM/bebop_simulator/tree/bebop_hokuyo)
* [gazebo2rviz](https://github.com/andreasBihlmaier/gazebo2rviz) (If you need the RViz visualization)

## Installation procedure
After installing the dependencies, you can build our propsed MPC package as follows:
``` 
cd your_catkin_ws/src
git clone https://github.com/Houman-HM/diff_opt_tracking
cd .. && catkin build
source your_catkin_ws/devel/setup.bash
```
## Running the algorithm

In order to run the MPC for the target tracking, follow the procedure below:

### In the first terminal:
```
roslaunch diff_opt_tracking random_world.launch
```

This launches a random environment with various obstacles in Gazebo.
### In the second terminal:

```
rosrun diff_opt_tracking planner.py
```
You can now publish velcoities on "target/cmd_vel" topic to move the target. As the target moves, the robot would follow it while avoiding occlusions and collisions.
