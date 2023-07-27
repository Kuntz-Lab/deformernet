#!/usr/bin/env python3
import open3d
import os
import numpy as np
import pickle
import timeit

import sys

import argparse
from sklearn.neighbors import NearestNeighbors
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

from copy import deepcopy

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/visualization/architecture_drawing_bimanual"



ROBOT_Z_OFFSET = 0.25


for i in range(0,1):

    
    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
 
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    pc = data["partial pcs"][0].transpose(1,0)
    pc_goal = data["partial pcs"][1].transpose(1,0)

    # num_pts = 1024
    # pc = down_sampling(data["partial pcs"][0], num_pts=num_pts).transpose(1,0)
    # pc_goal = down_sampling(data["partial pcs"][1], num_pts=num_pts).transpose(1,0)

    mp_pos_1 = np.array(list(data["mani_point"][0]["p"]))
    mp_pos_2 = np.array(list(data["mani_point"][1]["p"]))


    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc.transpose(1,0))
    
    _, nearest_idxs_1 = neigh.kneighbors(mp_pos_1.reshape(1, -1))
    mp_channel_1 = np.zeros(pc.shape[1])
    mp_channel_1[nearest_idxs_1.flatten()] = 1
    
    _, nearest_idxs_2 = neigh.kneighbors(mp_pos_2.reshape(1, -1))
    mp_channel_2 = np.zeros(pc.shape[1])
    mp_channel_2[nearest_idxs_2.flatten()] = 1        
    
    modified_pc = np.vstack([pc, mp_channel_1, mp_channel_2])# pc with 4th and 5th channel with value of 1 if near the MP point and 0 elsewhere
    

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pc.transpose(1,0)))
    pcd.paint_uniform_color([0,0,0])
    
    pcd_ori = deepcopy(pcd)
    
    # colors = np.zeros((pc.shape[1],3))
    # colors[nearest_idxs.flatten()] = [1,0,0]
    # pcd.colors =  open3d.utility.Vector3dVector(colors)

    mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point_1_sphere.paint_uniform_color([0,0,1])
    mani_point_1_sphere.translate(tuple(mp_pos_1))
    mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point_2_sphere.paint_uniform_color([1,0,0])
    mani_point_2_sphere.translate(tuple(mp_pos_2))

    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(pc_goal.transpose(1,0))
    pcd_goal.paint_uniform_color([1,0,0])
    
    # open3d.visualization.draw_geometries([pcd, mani_point_1_sphere, mani_point_2_sphere])  
    open3d.visualization.draw_geometries([pcd_goal])  

    # open3d.visualization.draw_geometries([pcd, mani_point_1_sphere, mani_point_2_sphere, pcd_ori.translate((0.6,0,0)), pcd_goal.translate((1.2,0,0))])           
    

