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

data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/visualization/architecture_drawing"



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

    mani_point = np.array(list(data["mani_point"]["p"]))
    # print(mani_point)
    # mani_point = data["mani_point"]        
    # mp_pos = np.array([-mani_point[0,3], -mani_point[1,3], mani_point[2,3] + ROBOT_Z_OFFSET])

    neigh = NearestNeighbors(n_neighbors=100)
    neigh.fit(pc.transpose(1,0))
    _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
    # _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc.shape[1])
    mp_channel[nearest_idxs.flatten()] = 1
    modified_pc = np.vstack([pc, mp_channel])# pc with 4th channel with value of 1 if near the MP point and 0 elsewhere
    
    # print(modified_pc[:4,100])
    # assert modified_pc.shape == (4,1024)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pc.transpose(1,0)))
    pcd.paint_uniform_color([0,0,0])
    
    pcd_ori = deepcopy(pcd)
    
    colors = np.zeros((pc.shape[1],3))
    colors[nearest_idxs.flatten()] = [1,0,0]
    pcd.colors =  open3d.utility.Vector3dVector(colors)

    mp = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mp.paint_uniform_color([0,1,0])  
    mp.translate(tuple(mani_point))

    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(pc_goal.transpose(1,0))
    pcd_goal.paint_uniform_color([1,0,0])

    open3d.visualization.draw_geometries([pcd, mp, pcd_ori.translate((0.6,0,0)), pcd_goal.translate((1.2,0,0))])           
    

