#!/usr/bin/env python
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
import moveit_commander
import rospy

rospy.init_node('manage_moveit_scene_node')
box_pose = PoseStamped()
box_pose.header.frame_id = 'world'
box_pose.pose.orientation.x = 0.0
box_pose.pose.orientation.y = 0.0
box_pose.pose.orientation.z = 0.0
box_pose.pose.orientation.w = 1.0
box_pose.pose.position.x = 0.0 
box_pose.pose.position.y = 0.3
box_pose.pose.position.z = 0.5*0.1 - 0.35
box_name = "box"
scene = moveit_commander.PlanningSceneInterface()
rospy.sleep(0.5)
scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))
rospy.sleep(0.5)