from copy import deepcopy
import numpy as np
import open3d
import pickle
import os
from copy import deepcopy

mp_classifer_data_path = "/home/baothach/shape_servo_data/manipulation_points/bimanual/multi_boxes_1000Pa/data" 
sample_count = 0
mp_count = 3

for sample_count in range(0,20):
    with open(os.path.join(mp_classifer_data_path, f"sample {sample_count}.pickle"), 'rb') as handle:
        data = pickle.load(handle)


    pcs = data["partial pcs"]
    mp_1 = tuple(data["mani_point_1"][0])
    mp_2 = tuple(data["mani_point_2"][0])

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pcs[0]))
    
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(np.array(pcs[1]))
    pcd_goal.paint_uniform_color([1,0,0])
    
    # pcd_final = open3d.geometry.PointCloud()
    # pcd_final.points = open3d.utility.Vector3dVector(np.array(final_pc))
    # pcd_final.paint_uniform_color([0,0,0])
    mani_point_1 = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point_1.paint_uniform_color([0,0,1])
    mani_point_2 = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point_2.paint_uniform_color([0,1,0])


    open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.,0,0.2)),\
                    mani_point_1.translate(mp_1),\
                    mani_point_2.translate(mp_2)]) 

