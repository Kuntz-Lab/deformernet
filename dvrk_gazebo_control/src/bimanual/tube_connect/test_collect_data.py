import open3d
import os
import numpy as np
import pickle

ROBOT_Z_OFFSET = 0.25
# two_robot_offset = 1.0


data_recording_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/data"

for i in range(0, 20):    
    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    # print("=============")
    # print("deltas:", np.concatenate((data["pos_1"].flatten(), data["pos_2"].flatten())))
    # print("data type:", data["data type"])


    pc = data["partial pcs"][0] 
    pc_goal = data["partial pcs"][1]
    mani_point_1 = data["mani_point_1"]   
    two_robot_offset = data["two_robot_offset"]  
    mp_pos_1 = np.array([mani_point_1[0,3]+two_robot_offset.x, two_robot_offset.y + mani_point_1[1,3], mani_point_1[2,3] + ROBOT_Z_OFFSET])
    # mp_pos_1 = np.array([mani_point_1[0,3], -two_robot_offset + mani_point_1[1,3], mani_point_1[2,3] + ROBOT_Z_OFFSET])
    mani_point_2 = data["mani_point_2"]        
    mp_pos_2 = np.array([-mani_point_2[0,3], -mani_point_2[1,3], mani_point_2[2,3] + ROBOT_Z_OFFSET])

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color([0,0,0]) 
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(pc_goal)
    pcd_goal.paint_uniform_color([1,0,0])
    
    mani_point_sphere_1 = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point_sphere_1.paint_uniform_color([0,0,1])   
    mani_point_sphere_1.translate(tuple(mp_pos_1))
    mani_point_sphere_2 = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point_sphere_2.paint_uniform_color([0,1,0]) 
    mani_point_sphere_2.translate(tuple(mp_pos_2))
    open3d.visualization.draw_geometries([pcd, pcd_goal, mani_point_sphere_1, mani_point_sphere_2], width = 600, height=600)






