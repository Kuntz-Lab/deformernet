#!/usr/bin/env python3
import open3d
import numpy as np
import os
import pickle5 as pickle
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def pcd_ize(pc):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)    
    return pcd

def add_occlusion(pc, num_occlusion, num_neighbors=50, nn_radius=[0.02], vis=False):
    occluded_pc = deepcopy(pc)
    
    if vis:
        pcds = []
        occlusion_points = []
        pcds.append(deepcopy(pcd_ize(pc)).paint_uniform_color([0,0,0]))
    
    for i in range(num_occlusion):
        rand_idx = np.random.randint(0, occluded_pc.shape[0])
        rand_point = occluded_pc[rand_idx]
        # print("rand_idx:", rand_idx, occluded_pc.shape[0])
        
        # neigh = NearestNeighbors(n_neighbors=num_neighbors)
        if len(nn_radius) == 1:
            neigh = NearestNeighbors(radius=nn_radius[0])
        elif len(nn_radius) == 2:
            neigh = NearestNeighbors(radius=np.random.uniform(nn_radius[0], nn_radius[1]))   
            # print("radius min and max")     
        neigh.fit(occluded_pc)
        
        # _, nearest_idxs = neigh.kneighbors(rand_point.reshape(1, -1))   
        _, nearest_idxs = neigh.radius_neighbors(rand_point.reshape(1, -1))
        # print(nearest_idxs[0])
        # occluded_pc = np.delete(occluded_pc, nearest_idxs, axis=0)
        occluded_pc = np.delete(occluded_pc, nearest_idxs[0], axis=0)
        # print(occluded_pc.shape)

        if vis:
            pcds.append(deepcopy(pcd_ize(occluded_pc)).translate((0.2*(i+1), 0, 0)).paint_uniform_color([0,0,0]))     
            occlusion_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            occlusion_point.paint_uniform_color(list(np.random.uniform(0.5,1,size=3)))
            occlusion_point.translate(tuple(list(rand_point)))
            occlusion_points.append(occlusion_point)  
            
    if vis:
        open3d.visualization.draw_geometries(pcds+occlusion_points)
        
    return occluded_pc
        
        
        
       

path = "/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_box_1kPa/evaluate_occlusion/goal_data"

with open(os.path.join(path, "box_1.pickle"), 'rb') as handle:
    data = pickle.load(handle)   
    
goal_pc = down_sampling(data["partial pcs"][0])
# print(goal_pc.shape)

for _ in range(10):
    occluded_pc = add_occlusion(goal_pc, num_occlusion=3, nn_radius=[0.03], vis=True) #0.02 - 0.03