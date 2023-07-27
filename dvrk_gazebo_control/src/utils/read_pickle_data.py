import numpy as np
import open3d
import pickle

with open('/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch2/data/sample 4.pickle', 'rb') as handle:
    data = pickle.load(handle)


print(data["positions"])
pc = data["point clouds"][0]    
pc_goal = data["point clouds"][1] 

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(pc.toarray()) 

pcd_goal = open3d.geometry.PointCloud()
pcd_goal.points = open3d.utility.Vector3dVector(pc_goal.toarray())
open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.2, 0, 0))])  

#===================================================

# with open('/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch1/processed/processed sample 0.pickle', 'rb') as handle:
#     data = pickle.load(handle)


# print(data["positions"])
# print(data["grasp_pose"])
# pc = data["keypoints"][0]    
# pc_goal = data["keypoints"][1] 

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(np.transpose(pc, (1, 0))) 

# pcd_goal = open3d.geometry.PointCloud()
# pcd_goal.points = open3d.utility.Vector3dVector(np.transpose(pc_goal, (1, 0)))
# open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.2, 0, 0))])  