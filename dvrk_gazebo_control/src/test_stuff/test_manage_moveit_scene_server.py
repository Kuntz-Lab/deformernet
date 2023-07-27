#!/usr/bin/env python

# import roslib; roslib.load_manifest('grasp_pipeline')
import sys
import rospy
from moveit_commander import RobotCommander, PlanningSceneInterface, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from grasp_pipeline.srv import *
import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('grasp_pipeline')
import numpy as np


#To do: change this file to a ros server to load different objects.
class ManageSceneInMoveit:
    def __init__(self):
        rospy.init_node('manage_moveit_scene_node')
    
    def handle_create_moveit_scene(self, req):        
        scene = PlanningSceneInterface()
        rospy.sleep(1)
        
        print ("object_mesh_path: ", req.object_mesh_path)
        scene.add_mesh('obj_mesh', req.object_pose_world, req.object_mesh_path, size = tuple(np.array([1,1,1])*req.mesh_scaling_factor)) 


        rospy.sleep(1)
        
        response = ManageMoveitSceneResponse()
        response.success = True
        return response


    def create_moveit_scene_server(self):
        rospy.Service('create_moveit_scene', ManageMoveitScene, self.handle_create_moveit_scene)
        rospy.loginfo('Service create_scene:')
        rospy.loginfo('Ready to create the moveit_scene.')


    def handle_clean_moveit_scene(self, req):        
        scene = PlanningSceneInterface()
        rospy.sleep(1)
    
        # clean the scene
        #scene.remove_world_object('table_box')
        scene.remove_world_object('obj_mesh')
         
        response = ManageMoveitSceneResponse()
        response.success = True
        return response
   

    def clean_moveit_scene_server(self):
        rospy.Service('clean_moveit_scene', ManageMoveitScene, self.handle_clean_moveit_scene)
        rospy.loginfo('Service clean_scene:')
        rospy.loginfo('Ready to clean the moveit_scene.')


if __name__=='__main__':
    ms = ManageSceneInMoveit()
    ms.create_moveit_scene_server()
    ms.clean_moveit_scene_server()
    rospy.spin()
