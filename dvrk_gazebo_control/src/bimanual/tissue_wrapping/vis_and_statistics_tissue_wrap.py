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

# for i in range(0, 206):
# # for i in []:
    
#     # if i in [85]:
#     #     continue
#     # if os.path.isfile(file_name):
    
#     if i % 1 == 0:
#         print(f"Current index: {i}")
    
#     save_path = os.path.join(data_recording_path, f"sample {i}.pickle")
#     with open(save_path, 'rb') as handle:
#         data = pickle.load(handle)   

#     final_pc = data["final_full_pc"]    

#     # open3d.visualization.draw_geometries([pcd_ize(final_pc)])
#     # open3d.visualization.draw_geometries([pcd_ize(backbone)])


#     percent = compute_intersection_percent(final_pc, data["tri_indices"], cylinder_shift=data["cylinder_shift"], vis = False)

 

#     intersection_percents.append(percent)

#     print(f"Percent intersect: {percent*100} %")
#     print("========================================")

# # with open(os.path.join(data_recording_path, "saved_percents.pickle"), 'wb') as handle:
# #     pickle.dump(intersection_percents, handle, protocol=pickle.HIGHEST_PROTOCOL)  


# print("average intersection percent:", np.mean(intersection_percents)*100)
# print("min intersection percent:", np.min(intersection_percents)*100)
# print("max intersection percent:", np.max(intersection_percents)*100)





### Visualize
with open(os.path.join(data_recording_path, "saved_percents.pickle"), 'rb') as handle:
    intersection_percents = np.array(pickle.load(handle))  



sorted_idxs = np.argsort(intersection_percents)[::-1]   # descending
# print(sorted_idxs[:100])

max_len = 150
quartile_idxs = np.array([0,round(max_len*0.25),round(max_len*0.5),round(max_len*0.75),max_len-1])

# print(intersection_percents[sorted_idxs[:max_len]])
print("average intersection percent:", np.mean(intersection_percents[sorted_idxs[:max_len]])*100)

print(sorted_idxs[quartile_idxs])
print(intersection_percents[sorted_idxs[quartile_idxs]]*100)



# for i in [41, 192,  26, 113,  35]:
    
#     if i % 1 == 0:
#         print(f"Current index: {i}")
    
#     save_path = os.path.join(data_recording_path, f"sample {i}.pickle")
#     with open(save_path, 'rb') as handle:
#         data = pickle.load(handle)   

#     final_pc = data["final_full_pc"]    

#     percent = compute_intersection_percent(final_pc, data["tri_indices"], cylinder_shift=data["cylinder_shift"], vis = True)
#     print(f"Percent intersect: {percent*100} %")
#     print("========================================")


