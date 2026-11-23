#!/usr/bin/env python3
"""
QBot Platform ROS Bridge v5.1
Uses commanded velocity for heading + position estimation.
mode=0 for working motors.
"""

import rospy
import time
import math
import numpy as np
import os
import sys

QUARC_DIR = os.path.expanduser(
    '~/Documents/Quanser/Mobile Robotics/l1_forward_kinematics/python')
os.chdir(QUARC_DIR)
sys.path.insert(0, QUARC_DIR)

from geometry_msgs.msg import Twist, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf
import tf.transformations

from pal.products.qbot_platform import QBotPlatformDriver
from pal.utilities.lidar import Lidar
from hal.content.qbot_platform_functions import QBPMovement

latest_linear  = 0.0
latest_angular = 0.0


def cmd_vel_callback(msg):
    global latest_linear, latest_angular
    latest_linear  = msg.linear.x
    latest_angular = msg.angular.z


def main():
    global latest_linear, latest_angular

    rospy.init_node('qbot_bridge')
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
    scan_pub = rospy.Publisher('/lidar_scan', LaserScan, queue_size=10)
    odom_pub = rospy.Publisher('/odom', Odometry, queue_size=50)
    odom_broadcaster = tf.TransformBroadcaster()

    print("Connecting to QBot hardware...")
    qbot = QBotPlatformDriver(mode=0, ip='localhost')
    move = QBPMovement()
    print("QBot connected!")

    print("Connecting to LiDAR...")
    try:
        lidar = Lidar(type='LeishenM10P', numMeasurements=384)
        lidar_ok = True
        print("LiDAR connected!")
    except Exception as e:
        print(f"WARNING: LiDAR failed: {e}")
        lidar_ok = False

    # Odometry state
    x, y, theta = 0.0, 0.0, 0.0
    start     = time.time()
    prev_time = time.time()
    rate      = rospy.Rate(20)
    loop_count = 0

    print("\n=== Bridge running! ===")

    while not rospy.is_shutdown():
        now = time.time()
        dt  = now - prev_time
        prev_time = now
        t = now - start

        # Send motor commands
        wL, wR = move.diff_drive_inverse_velocity_kinematics(
            latest_linear, latest_angular)
        commands = np.array([wL, wR], dtype=np.float64)
        new = qbot.read_write_std(timestamp=t, arm=1, commands=commands)

        # Update odometry from COMMANDED velocities
        v = latest_linear
        w = latest_angular
        theta += w * dt
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt

        # Diagnostics every 2 seconds
        loop_count += 1
        if loop_count % 40 == 0:
            print(f"cmd=[{latest_linear:.2f},{latest_angular:.2f}] "
                  f"pose=[{x:.2f},{y:.2f},{math.degrees(theta):.1f}] "
                  f"new={new}")

        # Publish TF: origin -> base_link
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        current_time = rospy.Time.now()
        odom_broadcaster.sendTransform(
            (x, y, 0.0), odom_quat, current_time,
            "base_link", "origin")

        # Publish Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = "origin"
        odom_msg.child_frame_id  = "base_link"
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.orientation = Quaternion(*odom_quat)
        odom_msg.twist.twist.linear.x  = v
        odom_msg.twist.twist.angular.z = w
        odom_pub.publish(odom_msg)

        # LiDAR
        if lidar_ok:
            try:
                if lidar.read():
                    scan_msg = LaserScan()
                    scan_msg.header.stamp    = current_time
                    scan_msg.header.frame_id = "base_link"
                    angles    = np.array(lidar.angles).flatten()
                    distances = np.array(lidar.distances).flatten()
                    if len(angles) > 0:
                        sort_idx  = np.argsort(angles)
                        angles    = angles[sort_idx]
                        distances = distances[sort_idx]
                        scan_msg.angle_min       = float(angles[0])
                        scan_msg.angle_max       = float(angles[-1])
                        scan_msg.angle_increment = float(
                            (angles[-1] - angles[0]) / max(len(angles)-1, 1))
                        scan_msg.range_min = 0.05
                        scan_msg.range_max = 12.0
                        scan_msg.ranges    = distances.tolist()
                        scan_pub.publish(scan_msg)
            except Exception:
                pass

        rate.sleep()

    print("Shutting down...")
    qbot.read_write_std(timestamp=time.time()-start, arm=0,
                        commands=np.zeros(2))
    qbot.terminate()
    if lidar_ok:
        lidar.terminate()


if __name__ == '__main__':
    main()
