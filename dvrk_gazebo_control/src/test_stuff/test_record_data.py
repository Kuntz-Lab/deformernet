import open3d
import numpy as np
import torch
import pickle
import os
from shutil import copyfile

# with open('/home/baothach/shape_servo_data/batch_3b(using_camera)', 'rb') as handle:
#     data = pickle.load(handle)

# pcs = np.array(data["point clouds"])
# positions = np.array(data["positions"]) 
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0]) 
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])
 
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])
 
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

# # initial_state = data["saved intial state"]

# # print(pcs.shape)
# # print("===========")
# # print(positions.shape)
# # print(positions)
# # print("===========")
# # print(initial_state.shape)
# # print(type(initial_state))
# # print("===========")


# # max_z_0 = max([x[2] for x in pcs[0]])
# # max_z_1 = max([x[2] for x in pcs[1]])
# # print(max_z_1 - max_z_0)

# pcd_vis = []
# pcd_goal = open3d.geometry.PointCloud()
# pcd_goal.points = open3d.utility.Vector3dVector(pcs[1][-1])  
# pcd_goal.paint_uniform_color([0, 1, 0])
# for i in range(len(pcs)):
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pcs[i][0])  
#     pcd.paint_uniform_color([1, 0.706, 0])
#     # open3d.visualization.draw_geometries([pcd])
#     # pcd_vis.append(pcd)
#     open3d.visualization.draw_geometries([pcd, pcd_goal])

# open3d.visualization.draw_geometries(pcd_vis)


# for i in range(0,71):
#     with open('/home/baothach/shape_servo_data/generalization/surgical_setup/data/sample ' + str(i) + '.pickle', 'rb') as handle:
#         data = pickle.load(handle)

#     # pc = data["partial pcs"]
#     pc = data["full pcs"]
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pc[0])
#     pcd_goal = open3d.geometry.PointCloud()
#     pcd_goal = open3d.geometry.PointCloud()
#     pcd_goal.points = open3d.utility.Vector3dVector(pc[1])
#     open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.2,0,0))])
    

# print(data["positions"])
# print(data["grasp_pose"])
# nan at 4915, 4916, 4917, 4918, 7844, 7845, 7846 surgical setup
count = 0 
for i in range(0,13000):    
    file_name = '/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data/processed sample ' + str(i) + '.pickle'
    if not os.path.isfile(file_name):
        continue 
    
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    # pc = data["partial pcs"]
    # if np.isnan(pc[0]).any() or np.isnan(pc[1]).any():
    #     print("nan:", i)
    #     os.remove(file_name)
    
    pos = data["twist"]
    if np.isnan(pos.flatten()).any():        
        count += 1
        print("nan:", i)   
        # os.remove(file_name) 
    else:
        fname = '/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data_w_mp/mp sample ' + str(i) + '.pickle'
        new_fname = '/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data_w_mp_twist/mp sample ' + str(i) + '.pickle'
        copyfile(fname, new_fname)
print("total:", count)
  

# with open('/home/baothach/shape_servo_data/generalization/surgical_setup/data/sample ' + str(7844) + '.pickle', 'rb') as handle:
#     data = pickle.load(handle)      
# pc = data["full pcs"][0]
# print(pc)
# inva_idx = np.argwhere(np.isnan(pc))
# for idx in inva_idx:
#     print(pc[idx[0],idx[1]])

# for i in range(5,20):    
#     file_name = '/home/baothach/shape_servo_data/RL_shapeservo/box/data/sample ' + str(i) + '.pickle'
#     with open(file_name, 'rb') as handle:
#         data = pickle.load(handle)
#     # print(data.keys())
#     # pc = data["partial pcs"]
#     pc = data["full pcs"]
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(pc[0])   
#     pcd_goal = open3d.geometry.PointCloud()
#     pcd_goal.points = open3d.utility.Vector3dVector(pc[1])    
#     open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.2,0,0))]) 
