#!/usr/bin/env python3
from copy import deepcopy
import numpy as np
import open3d
import pickle
import os
from copy import deepcopy

# mp_classifer_data_path = "/home/baothach/shape_servo_data/manipulation_points/box/mp_classifer_data" 
# sample_count = 0
# mp_count = 3

# for mp_count in range(0,4):
#     with open(os.path.join(mp_classifer_data_path, f"sample {sample_count} mp {mp_count}.pickle"), 'rb') as handle:
#         data = pickle.load(handle)


#     pcs = data["partial pcs"]
#     # final_pc = data["final pc"]
#     chamfer = data["chamfer"]
#     mp = tuple(data["mani_point"][0])
#     gt_chamfer = data["gt chamfer"]
#     gt_mp = tuple(data["gt mani_point"][0])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(np.array(pcs[0]))
    
#     pcd_goal = open3d.geometry.PointCloud()
#     pcd_goal.points = open3d.utility.Vector3dVector(np.array(pcs[1]))
#     pcd_goal.paint_uniform_color([1,0,0])
    
#     # pcd_final = open3d.geometry.PointCloud()
#     # pcd_final.points = open3d.utility.Vector3dVector(np.array(final_pc))
#     # pcd_final.paint_uniform_color([0,0,0])
#     mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
#     mani_point.paint_uniform_color([0,0,1])
#     gt_mani_point = deepcopy(mani_point)
#     gt_mani_point.paint_uniform_color([1,0,0])


#     print("Chamfer:", chamfer)
#     print("Gt Chamfer:", gt_chamfer)
#     # open3d.visualization.draw_geometries([pcd, pcd_final.translate((0.15,0,0)), pcd_goal.translate((0.3,0,0))])  
#     # open3d.visualization.draw_geometries([pcd, pcd_final.translate((0.15,0,0)), pcd_goal.translate((0.3,0,0)),\
#     #                 mani_point.translate(mp), gt_mani_point.translate(gt_mp)])  
#     open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.3,0,0)),\
#                     mani_point.translate(mp), gt_mani_point.translate(gt_mp)]) 



data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa_2/classifer_gt_data"
for _ in range(0,10):
    i = np.random.randint(low=0, high=5000)
    with open(os.path.join(data_recording_path, "sample " + str(i) + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)

    pcs = data["partial pcs"]
    # final_pc = data["final pc"]
    gt_mp = tuple(data["mani_point"]["p"])
    print(data["mani_point"]["p"])


    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pcs[0]))
    pcd.paint_uniform_color([0,0,0]) 
    
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(np.array(pcs[1]))
    pcd_goal.paint_uniform_color([1,0,0])        

    gt_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    gt_mani_point.paint_uniform_color([0,1,0])
    gt_mani_point.translate(gt_mp)

    open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.15,0,0)),\
                        gt_mani_point]) 