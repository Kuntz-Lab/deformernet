import numpy as np
import pickle
import open3d
import os
import transformations

goal_recording_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/evaluate/goal_data/box_1k/inside/"
with open(os.path.join(goal_recording_path, "box_0_test" + ".pickle"), 'rb') as handle:
    goal_datas = pickle.load(handle) 


# for i in [0]:
for i in range(10):
    print("===============")
    print("index:", i)

    goal_pc_numpy = goal_datas[i]["full pcs"][1]   # first goal pc

    goal_pos = goal_datas[i]["pos"] 
    goal_rot = goal_datas[i]["rot"] 
    temp2 = np.eye(4)
    temp2[:3,:3] = goal_rot            
    print("goal_pos, goal_rot:", goal_pos, np.array(transformations.euler_from_matrix(temp2))/np.pi) 

    # pcd_goal = open3d.geometry.PointCloud()
    # pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy)  
    # pcd_goal.paint_uniform_color([1,0,0]) 
    # open3d.visualization.draw_geometries([pcd_goal])