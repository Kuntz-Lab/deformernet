import sys
sys.path.append("../../")
import os
from utils.point_cloud_utils import pcd_ize, spherify_point_cloud_open3d
from utils.miscellaneous_utils import read_pickle_data, write_pickle_data
import open3d

import numpy as np

data_recording_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/data/retraction_cutting"
categoies = ["cylinder", "ellipsoid"]    #["cylinder", "ellipsoid"]
num_object_per_category = 50     #50cc vxc v

for context_idx in range(0,10): 
    for category in categoies:
        for object_idx in range(0, num_object_per_category):
            obj_name = f"{category}_{object_idx}"
            data = read_pickle_data(f"{data_recording_path}/{obj_name}_{context_idx}.pickle")
            goal_pcs = data["partial_goal_pcs"]
            init_pc = data["partial_init_pc"]

            pcd_goal_1 = pcd_ize(goal_pcs[0], color=[0, 0, 1])
            pcd_goal_2 = pcd_ize(goal_pcs[1], color=[1, 0, 0])
            pcd_init = pcd_ize(init_pc, color=[0, 0, 0])
            # context_pcd = pcd_ize(data["context"], color=[0, 1, 0])
            context_pcd = spherify_point_cloud_open3d(data["context"], color=[0, 1, 0])
            open3d.visualization.draw_geometries([pcd_goal_1, pcd_goal_2, pcd_init, context_pcd])
            # open3d.visualization.draw_geometries([pcd_goal_1, context_pcd])
