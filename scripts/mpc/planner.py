#!/usr/bin/env python3
import rospy
import message_filters
from nav_msgs.msg import Odometry
import rospkg
import sys
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import rospkg
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import threading

import queue
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

import bernstein_coeff_order10_arbitinterval
import mpc_module as mpc_module
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud
import rospkg
import copy
import open3d
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import random


robot_cmd_publisher = None
robot_pose_vel = []
target_pose_vel = []

is_received = False
robot_cmd_publisher = None
robot_traj_marker_publisher = None
robot_traj_publisher = None
robot_traj_publisher = None
pointcloud_publisher = None
obstacle_points = None
marker_array_pub = None
pointcloud_mutex = threading.Lock()
publish_pointcloud_mutex = threading.Lock()
odom_mutex = threading.Lock()
obstacles_for_publishing = None
num_laser_points = 720
num_down_sampled = 200
publish_points = False

xyz = np.random.rand(720, 3)

def pointcloudCallback(pointcloud):
    global x_obs_pointcloud, x_obs_pointcloud, obstacle_points, is_received, pointcloud_mutex, xyz, num_laser_points, num_down_sampled
    
    pointcloud_mutex.acquire()
    msg_len = len(pointcloud.points)
    x_obs_pointcloud_vehicle = np.ones((num_laser_points, 1)) * 1000
    y_obs_pointcloud_vehicle = np.ones((num_laser_points, 1)) * 1000
    
    for nn in range(0, msg_len):
        x_obs_pointcloud_vehicle[nn] = pointcloud.points[nn].x
        y_obs_pointcloud_vehicle[nn] = pointcloud.points[nn].y

    xyz[:, 0] = x_obs_pointcloud_vehicle.flatten()
    xyz[:, 1] = y_obs_pointcloud_vehicle.flatten()
    xyz[:, 2] = 1

    idxes = np.argwhere(xyz[:, :] >= 300)
    xyz[idxes, 0] = 9
    xyz[idxes, 1] = 9


    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=0.19)
    
    downpcd_array = np.asarray(downpcd.points)

    num_down_sampled_points = downpcd_array[:, 0].shape[0]

    x_obs_down_sampled = np.ones((200, 1)) * 1000
    y_obs_down_sampled = np.ones((200, 1)) * 1000
    x_obs_down_sampled[0:num_down_sampled_points, 0] = downpcd_array[:, 0]
    y_obs_down_sampled[0:num_down_sampled_points, 0] = downpcd_array[:, 1]
    min_idx = np.argmin(x_obs_down_sampled)

    x_obs_down_sampled[num_down_sampled_points:, 0] = x_obs_down_sampled[min_idx, 0]
    y_obs_down_sampled[num_down_sampled_points:, 0] = y_obs_down_sampled[min_idx, 0]
    obstacle_points= np.hstack(
        (x_obs_down_sampled, y_obs_down_sampled))
    pointcloud_mutex.release()
    is_received = True
    

def odomCallback(robot_odom, target_odom):

    # print("In the call back")
    global is_received, robot_pose_vel, target_pose_vel, odom_mutex
    odom_mutex.acquire()
    robot_orientation_q = robot_odom.pose.pose.orientation
    robot_orientation_list = [robot_orientation_q.x, robot_orientation_q.y, robot_orientation_q.z, robot_orientation_q.w]
    target_orientation_q = target_odom.pose.pose.orientation
    target_orientation_list = [target_orientation_q.x, target_orientation_q.y, target_orientation_q.z, target_orientation_q.w]

    target_orientation_q = target_odom.pose.pose.orientation
    target_orientation_list = [target_orientation_q.x, target_orientation_q.y, target_orientation_q.z, target_orientation_q.w]

    (robot_roll, robot_pitch, robot_yaw) = euler_from_quaternion (robot_orientation_list)
    (target_roll, target_pitch, target_yaw) = euler_from_quaternion (target_orientation_list)

    robot_pose_vel = [robot_odom.pose.pose.position.x, robot_odom.pose.pose.position.y, robot_yaw, 
                    robot_odom.twist.twist.linear.x, robot_odom.twist.twist.linear.y, robot_odom.twist.twist.angular.z]
    target_pose_vel = [target_odom.pose.pose.position.x, target_odom.pose.pose.position.y, target_yaw, 
                    target_odom.twist.twist.linear.x, target_odom.twist.twist.linear.y, target_odom.twist.twist.angular.z]
    odom_mutex.release()


def mpc():

    global is_received, robot_pose_vel, target_pose_vel, robot_cmd_publisher, \
    robot_traj_publisher, publish_pointcloud_mutex, publish_points,obstacles_for_publishing, robot_traj_marker_publisher, pointcloud_publisher, obstacle_points, marker_array_pub
    rospy.loginfo("MPC thread started sucessfully!")
    

    rospack = rospkg.RosPack()
    package_path = rospack.get_path("diff_opt_tracking")
    
    maxiter_mpc = 50000

    ############# parameters

    t_fin = 5.0
    num = 100
    tot_time = np.linspace(0, t_fin, num)
    tot_time_copy = tot_time.reshape(num, 1)
            
    P, Pdot, Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
    nvar = np.shape(P)[1]

    tot_time_jax = jnp.asarray(tot_time)

    ###################################
    t_update = 0.05
    num_up = 200
    dt_up = 0.01#t_fin/num_up
    tot_time_up = np.linspace(0, t_fin, num_up)
    tot_time_copy_up = tot_time_up.reshape(num_up, 1)

    P_up, Pdot_up, Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)

    P_up_jax = jnp.asarray(P_up)
    Pdot_up_jax = jnp.asarray(Pdot_up)
    Pddot_up_jax = jnp.asarray(Pddot_up)

    ########################################

    x_init =  0
    vx_init = 0.0

    y_init =  0
    vy_init = 0.0


    x_target_init = target_pose_vel[0] - robot_pose_vel[0]
    y_target_init = target_pose_vel[1] - robot_pose_vel[1]

    vx_target = 0.0
    vy_target = 0.4


    x_target = x_target_init + vx_target*tot_time
    y_target = y_target_init + vy_target*tot_time

    x_target = x_target.reshape(1, num)
    y_target = y_target.reshape(1, num)

    weight_biases_mat_file = loadmat(package_path + "/nn_weights/occlusion.mat")


    occlusion_weight = 10000000

    prob = mpc_module.batch_occ_tracking(P, Pdot, Pddot, P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight)
    
    prob.W0, prob.b0, prob.W1, \
    prob.b1, prob.W2, prob.b2, \
    prob.W3, prob.b3 = mpc_module.get_weights_biases(weight_biases_mat_file)

    alpha_init = np.arctan2(y_target_init - y_init, x_target_init - x_init)

    cvae_model = mpc_module.CVAE(package_path)

    rospy.loginfo("Waiting for initial JAX compilation!")
    
    for i in range(0, maxiter_mpc):
        if (rospy.is_shutdown()):
            break
        start_time = time.time()
        pointcloud_mutex.acquire()
        odom_mutex.acquire()
        robot_poses_local = copy.deepcopy(robot_pose_vel)
        target_poses_local = copy.deepcopy(target_pose_vel) 
        obstacles_local = copy.deepcopy(obstacle_points)
        obstacles_local[:,0] = obstacles_local[:,0] - robot_poses_local[0]
        obstacles_local[:,1] = obstacles_local[:,1] - robot_poses_local[1]

        # obstacle_points = jnp.asarray(obstacles_local)
        obstacles_for_init = obstacles_local

        vx_target = target_pose_vel[3] * np.cos(target_pose_vel[2])
        vy_target = target_pose_vel[3] * np.sin(target_pose_vel[2])
        alpha_init = robot_pose_vel[2]

        pointcloud_mutex.release()
        odom_mutex.release()

        

        drone_states = np.hstack((0,0, vx_init,vy_init,  x_target_init, y_target_init, 
                                  vx_target, vy_target))
    
        closest_obstacles = cvae_model.get_closet_obstacle(obstacles_for_init[:,0], obstacles_for_init[:,1]) 
        primal_sol, accumulated_res_primal, primal_sol_init, x, y = cvae_model.compute_initial_guess(drone_states, closest_obstacles, np.hstack((obstacles_for_init[:,0], obstacles_for_init[:,1])))


        c_x_samples_init, c_y_samples_init = primal_sol[:, 0:11], primal_sol[:, 11:22]
        x_samples_init, y_samples_init = x ,y
        jax_obstacles = jnp.asarray(closest_obstacles)
        
        c_x_best, c_y_best, x_best, y_best = prob.compute_cem(c_x_samples_init, c_y_samples_init, x_samples_init, y_samples_init, x_target, y_target, jax_obstacles)

        vx_control_local, vy_control_local, ax_control, \
        ay_control, vangular_control, robot_traj_x, robot_traj_y, vx_control, vy_control= prob.compute_controls(c_x_best, c_y_best, dt_up, vx_target, vy_target, 
							                                                    t_update, tot_time_copy_up, x_init, y_init, alpha_init,
                                                                                 x_target_init, y_target_init)

        if (i!=0):
            cmd = Twist()
            cmd.linear.x= vx_control_local
            cmd.linear.y= vy_control_local
            cmd.angular.z = vangular_control
            robot_cmd_publisher.publish(cmd)
        time_taken = time.time() - start_time
        rospy.loginfo ("Time taken: %s", str(time_taken))

        odom_mutex.acquire()
        
        x_init = 0
        y_init = 0

        vx_init = vx_control
        vy_init = vy_control

        x_target_init = target_pose_vel[0] - robot_pose_vel[0]
        y_target_init = target_pose_vel[1] - robot_pose_vel[1]

        x_target = x_target_init + vx_target * tot_time_jax
        y_target = y_target_init + vy_target * tot_time_jax

        x_target = x_target.reshape(1, num)
        y_target = y_target.reshape(1, num)
        
        odom_mutex.release()
    

if __name__ == "__main__":

	
    rospy.init_node('nn_mpc_node')
    rospack = rospkg.RosPack()
    marker_array_pub = rospy.Publisher("/trajectories", MarkerArray, queue_size=10)
    robot_cmd_publisher = rospy.Publisher('bebop/cmd_vel', Twist, queue_size=10)
    robot_traj_marker_publisher = rospy.Publisher('/robot_traj', Marker, queue_size=10)
    pointcloud_publisher = rospy.Publisher('/generated_pointcloud', Marker, queue_size=10)
    robot_traj_publisher = rospy.Publisher('command/pose', PoseStamped, queue_size=10)
    rospy.Subscriber("/pointcloud", PointCloud, pointcloudCallback)
    robot_odom_sub = message_filters.Subscriber('bebop/odom', Odometry)
    target_odom_sub = message_filters.Subscriber('/target/odom', Odometry)
    ts = message_filters.ApproximateTimeSynchronizer([robot_odom_sub, target_odom_sub], 1,1, allow_headerless=True)
    ts.registerCallback(odomCallback)
    mpc_thread = threading.Thread(target=mpc)
    mpc_thread.start()
    rospy.spin()


