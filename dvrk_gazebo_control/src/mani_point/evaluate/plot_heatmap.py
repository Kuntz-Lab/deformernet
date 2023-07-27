import open3d
import os
import numpy as np
import pickle
import torch
import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax



def down_sampling(pc):
    farthest_indices,_ = farthest_point_sampling(pc, 2048)
    pc = pc[farthest_indices.squeeze()]  
    return pc


data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/heatmaps/goal_data"
# file_names = sorted(os.listdir(data_recording_path))

result_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/heatmaps/results"
all_chamfers = []
all_mps = []

sample_count = 1
for sample_count in range (0,9):
    for mp_count in range(0,101):

        if mp_count == 86:
            break

        with open(os.path.join(result_path, f"sample {sample_count} mp {mp_count}.pickle"), 'rb') as handle:
            data = pickle.load(handle) 

        all_chamfers.append(data["chamfer"])
        all_mps.append(list(data["mani_point"][0]))


    # for i in [620]:
    file_name = os.path.join(data_recording_path, "goal " + str(sample_count) + ".pickle")
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
        

    # pcd = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/manipulation_points/box/init_box_pc.pcd")    
    pc = down_sampling(data["partial pcs"][0])   
    pc_goal = down_sampling(data["partial pcs"][1])
    gt_mp = np.array(list(data["mani_point"][0]))

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pc))
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(np.array(pc_goal))    

    all_mps = np.array(all_mps)
    # print(all_mps.shape)
    # print(pc.shape)

    num_nn = 5
    neigh = NearestNeighbors(n_neighbors=num_nn)
    neigh.fit(pc)
    _, nearest_idxs = neigh.kneighbors(all_mps)

    colors = np.zeros((2048,3))

    # probs = softmax(min(all_chamfers)/all_chamfers)
    # colors[nearest_idxs.flatten()] = np.array([[prob,0,0] for prob in probs for _ in range(num_nn)])

    colors[nearest_idxs.flatten()] = np.array([[min(all_chamfers)/chamf,0,0] for chamf in all_chamfers for _ in range(num_nn)])
    pcd.colors =  open3d.utility.Vector3dVector(colors)


    mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point.paint_uniform_color([0,0,1])
    mani_point.translate(tuple(gt_mp))

    open3d.visualization.draw_geometries([pcd, mani_point, pcd_goal.translate((0.2,0,0))]) 


    all_chamfers = []
    all_mps = []    