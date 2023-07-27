#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np
# from isaacgym import gymapi
# from isaacgym import gymtorch
# from isaacgym import gymutil
from copy import copy, deepcopy
import rospy
# from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO

# from utils.record_data_h5 import RecordGraspData_sparse
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image

# from core import Robot
# from behaviors import MoveToPose, TaskVelocityControl

sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
# from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
from architecture import DeformerNet2
import torch


ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86

sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *


    
# Set up DNN:
device = torch.device("cuda")
model = DeformerNet2(normal_channel=False)
weight_path = "/home/baothach/shape_servo_data/RL_shapeservo/box/weights"
model.load_state_dict(torch.load(os.path.join(weight_path, "run1epoch 300")))  
model.eval()


# #    Get goal pc:
# data_recording_path = "/home/baothach/shape_servo_data/comparison/RRT/goal_data"
# with open(os.path.join(data_recording_path, "sample " + str(9) + ".pickle"), 'rb') as handle:
#     data = pickle.load(handle)
#     goal_pc_numpy = data["partial pc"]
#     farthest_indices,_ = farthest_point_sampling(goal_pc_numpy, 1024)
#     goal_pc_numpy = goal_pc_numpy[farthest_indices.squeeze()]

#     goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float() 
#     pcd_goal = open3d.geometry.PointCloud()
#     pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
#     goal_position = data["positions"]  
# mesh = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# pcd = open3d.io.read_point_cloud("/home/baothach/Documents/pc_original.pcd")
# center = pcd.get_center()
# # pcd.translate((center[0], center[1], center[2]))
# current_pc = np.asarray(pcd.points) 
# current_pc = np.array([point for point in current_pc if point[2]>0.18])
# current_pc[:,1] = -current_pc[:,1]
# pcd.points = open3d.utility.Vector3dVector(current_pc) 
# farthest_indices,_ = farthest_point_sampling(current_pc, 1024)
# current_pc = current_pc[farthest_indices.squeeze()]         
# current_pc = torch.from_numpy(np.swapaxes(current_pc,0,1)).float() 

# # pcd_goal = open3d.io.read_point_cloud("/home/baothach/Documents/pc_move_up.pcd")
# pcd_goal = open3d.io.read_point_cloud("/home/baothach/Documents/pc_move_right.pcd")
# # center = - pcd_goal.get_center()
# # pcd_goal.translate((center[0], center[1], center[2]))
# goal_pc = np.asarray(pcd_goal.points) 
# goal_pc = np.array([point for point in goal_pc if point[2]>0.18])
# goal_pc[:,1] = -goal_pc[:,1]
# pcd_goal.points = open3d.utility.Vector3dVector(goal_pc) 

# farthest_indices,_ = farthest_point_sampling(goal_pc, 1024)
# goal_pc = goal_pc[farthest_indices.squeeze()]    
# goal_pc = torch.from_numpy(np.swapaxes(goal_pc,0,1)).float()                
       
# open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.2,0,0))])             
# open3d.visualization.draw_geometries([mesh.translate(tuple(center)), pcd, pcd_goal.translate((0.2,0,0))])       




# with torch.no_grad():
#     desired_position = model(current_pc.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().numpy()*(0.001)  

# print(desired_position)
 
pcd = open3d.io.read_point_cloud("/home/baothach/Documents/pc_move_right.pcd")
current_pc = np.asarray(pcd.points) 
max_z = max(abs(x[0]) for x in current_pc)
min_z = min(abs(x[0]) for x in current_pc)
print(min_z, max_z)