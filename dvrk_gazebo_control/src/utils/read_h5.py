import h5py
import numpy as np
import open3d
# import os

# os.chdir('/home/baothach/dvrk_grasp_data/batch1') 


# f = h5py.File("grasp_data.h5")
# print(list(f))
# # print(f.keys())

# print(f['cur_object_name'][()])
# print(f['object_1_name'][()])
# print(f['object_0_grasp_23_preshape_palm_world_pose'][()])
# print(f['object_2_grasp_2_true_preshape_palm_world_pose'][()])
# print(f['object_0_grasp_23_object_world_sim_pose'][()])
# print(f['object_1_grasp_0_object_world_seg_pose'][()])
# print(f['object_0_grasp_0_top_grasp'][()])

# print(f['object_0_name'][()])


# f = h5py.File("/home/baothach/shape_servo_data/multi_grasps/batch_1")
# # print(list(f))
# print(f['cur_grasp_id'][()])
# # print(f['manipulation pose 66'][()])
# # print(f['manipulation pose 1'][()])
# # print(f['manipulation pose 2'][()])
# # pc = f['point clouds 11'][()]
# # print(np.array(pc).shape)

# pos = f['positions 11'][()]
# print(np.array(pos).shape)
# group = f.get('grasp 2')
# print(group.items())
# print(group['point clouds'][()])
# print(group['positions'][()]),

#==========================================================
f = h5py.File("/home/baothach/shape_servo_data/keypoints/batch7/point_cloud/source/ordered_point_clouds.h5")
# print(list(f))
print(f['cur_id'][()])

# pc = f['point cloud 5 target'][()]
# print(pc)
# print(type(pc), pc.shape)

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(pc.reshape(-1, 3))
# open3d.visualization.draw_geometries([pcd])   

#==========================================================
# f = h5py.File("/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch1/shape_servo_data.h5")
# # print(list(f))
# print(f['cur_grasp_id'][()])
# pc = f['point cloud init 1'][1:3].toarray()
# pc_goal = f['point cloud goal 1'][1:3].toarray()
# print(f['position 2'][()])

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(pc) 

# pcd_goal = open3d.geometry.PointCloud()
# pcd_goal.points = open3d.utility.Vector3dVector(pc_goal)
# open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.2, 0, 0))])  