import os
import numpy as np
import trimesh
import open3d
import pickle
import transformations
from copy import deepcopy
from wrap_utils import *


def pcd_ize(pc):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)    
    return pcd

### Statistics
data_recording_path = "/home/baothach/shape_servo_data/tissue_wrap/evaluate/run1"


intersection_percents = []
distances = []

for i in range(0, 1):
    
    # if i in [85]:
    #     continue
    
    if i % 1 == 0:
        print(f"Current index: {i}")
    
    save_path = os.path.join(data_recording_path, f"sample {i}.pickle")
    with open(save_path, 'rb') as handle:
        data = pickle.load(handle)   

    final_pc = data["final_full_pc"]    
    backbone = final_pc[data["backbone_idxs"]]

    open3d.visualization.draw_geometries([pcd_ize(final_pc)])
    # open3d.visualization.draw_geometries([pcd_ize(backbone)])


    percent = compute_intersection_percent(backbone, data["cylinder_shift"], vis = True)

    edge_1 = final_pc[data["edges_idxs_1"]]
    edge_2 = final_pc[data["edges_idxs_2"]]
    distance = np.mean(edge_1, axis=0)-np.mean(edge_2, axis=0)
    distance = np.linalg.norm(distance)

    intersection_percents.append(percent)
    distances.append(distance)
    
    print("distance 2 edges:", np.mean(edge_1, axis=0)-np.mean(edge_2, axis=0)) 
    print(f"Percent intersect: {percent*100} %")
    print("========================================")

# with open(os.path.join(data_recording_path, "saved_percents.pickle"), 'wb') as handle:
#     pickle.dump(intersection_percents, handle, protocol=pickle.HIGHEST_PROTOCOL)  


# with open(os.path.join(data_recording_path, "saved_distances.pickle"), 'wb') as handle:
#     pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)  

print("average intersection percent:", np.mean(intersection_percents))
print("min intersection percent:", np.min(intersection_percents))
print("max intersection percent:", np.max(intersection_percents))

print("average distance:", np.mean(distances))



# ### Visualize
# with open(os.path.join(data_recording_path, "saved_percents.pickle"), 'rb') as handle:
#     intersection_percents = pickle.load(handle)   

# with open(os.path.join(data_recording_path, "saved_distances.pickle"), 'rb') as handle:
#     distances = pickle.load(handle)

# # sorted_idxs = np.argsort(intersection_percents)

# # print("average intersection percent:", np.mean(distances), np.mean(intersection_percents))
# # print("min intersection percent:", np.min(distances), np.min(intersection_percents))
# # print("max intersection percent:", np.max(distances), np.max(intersection_percents))

# sorted_percents = np.sort(intersection_percents)
# print(sorted_percents)


#############################

# data_recording_path = "/home/baothach/shape_servo_data/tissue_wrap/evaluate/run1"

# for i in [sorted_idxs[len(distances) // 4]]:
#     save_path = os.path.join(data_recording_path, f"sample {i}.pickle")
#     with open(save_path, 'rb') as handle:
#         data = pickle.load(handle)   

#     final_pc = data["final_full_pc"]
#     cylinder_shift = data["cylinder_shift"]
#     print(distances[i])
#     print(intersection_percents[i])

# quat = [0.5, 0.5,0.5, 0.5]
# trans_mat = transformations.quaternion_matrix(quat)
# trans_mat[:3,3] = np.array([0.00+0.00, -0.5+0.00, 0.04]) + cylinder_shift
# cylinder_mesh = trimesh.creation.annulus(r_min=0.0149*0.6, r_max=0.015*0.6, height=0.1, transform = trans_mat)
# cylinder_pc = trimesh.sample.sample_surface(cylinder_mesh, count=1024*4)[0]




# final_pcd = open3d.geometry.PointCloud()
# final_pcd.points = open3d.utility.Vector3dVector(final_pc)
# final_pcd.paint_uniform_color([0,0,0])

# cylinder_pcd = open3d.geometry.PointCloud()
# cylinder_pcd.points = open3d.utility.Vector3dVector(cylinder_pc)
# cylinder_pcd.paint_uniform_color([1,0,0])
# open3d.visualization.draw_geometries([final_pcd, cylinder_pcd]) 