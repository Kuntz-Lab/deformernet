#!/usr/bin/env python3
import sys
sys.path.append('/home/baothach/dvrk_shape_servo/src/dvrk_env/dvrk_gazebo_control/src/shape_servo')
from ShapeServo import *
import os
import pickle
import open3d
from utils import open3d_ros_helper as orh
from sklearn.decomposition import PCA
import numpy as np



rospy.init_node('isaac_grasp_client')
VFH_135s = []
VFH_135s_flatten = []
VFH_30s = []
positions = []
# with open('/home/baothach/shape_servo_data/batch_1_shuffled', 'rb') as handle:
#     data = pickle.load(handle)


data_recording_path = "/home/baothach/shape_servo_data/VFH/batch1/data"
data_processed_path = "/home/baothach/shape_servo_data/VFH/batch1/processed"
# i = 6000

for i in range(0, 3000): 

    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)    

    temp = []
    for points in data["point clouds"]:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))            
        ros_cloud = orh.o3dpc_to_rospc(pcd)
        feature_vector_135 = np.array(VFH_client(ros_cloud))
        
        VFH_135s_flatten.append(feature_vector_135)
        
        temp.append(feature_vector_135)

    positions.append(data["positions"])
    VFH_135s.append(temp)
    



pca = PCA(n_components=30)
# pca.fit(VFH_135s)   
pca.fit(VFH_135s_flatten)

for i, feature_vector_135 in enumerate(VFH_135s):
    feature_vector_30_current = pca.transform(feature_vector_135[0].reshape(1, -1))
    feature_vector_30_goal = pca.transform(feature_vector_135[1].reshape(1, -1))
    VFH_30 = (feature_vector_30_current, feature_vector_30_goal)
    # VFH_30s.append(feature_vector_30_goal - feature_vector_30_current)
    if i % 50 == 0:
        print("count: ", i)

   
    processed_data = {"positions": positions[i], "VFH_30": VFH_30, "VFH_135": feature_vector_135}
    with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)   








    

 









# for two_pc in data["point clouds"]:
#     temp = []
#     for points in two_pc:
#         pcd = open3d.geometry.PointCloud()
#         pcd.points = open3d.utility.Vector3dVector(np.array(points))            
#         ros_cloud = orh.o3dpc_to_rospc(pcd)
#         feature_vector_135 = np.array(VFH_client(ros_cloud))
        
#         VFH_135s_flatten.append(feature_vector_135)
        
#         temp.append(feature_vector_135)
#     VFH_135s.append(temp)

# data = {"point clouds": data["point clouds"], "target": data["positions"], "input": VFH_30s}

# with open('/home/baothach/shape_servo_data/batch_1_shuffled_VFH_30', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



# data = VFH_135s_flatten

# with open('/home/baothach/shape_servo_data/feature_vectors_135', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    