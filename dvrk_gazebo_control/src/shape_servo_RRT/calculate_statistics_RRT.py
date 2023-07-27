import numpy as np
import open3d
import os
import pickle5 as pickle
from robotRRT import pcd_ize, down_sampling

def compute_chamfer_dist(pcd, pcd_goal):
    return np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd))) 

np.set_printoptions(suppress=True)

goal_data_path = "/home/baothach/shape_servo_data/comparative_study/goal_data"
result_path = "/home/baothach/shape_servo_data/comparative_study/RRT/results"
thresholds = np.arange(0.1, 1.1, 0.1)

all_min_chamfers = []
times_at_specific_threshold = []

for i in range(0,10):

    # print(f"----- Goal {i} -------")
    
    with open(os.path.join(goal_data_path, f"sample {i}.pickle"), 'rb') as handle:
        goal_pc = down_sampling(pickle.load(handle)["partial pc"], num_pts=1024)

    with open(os.path.join(result_path, f"sample {i}.pickle"), 'rb') as handle:
        data = pickle.load(handle)
        
    chamfers = np.array(data["chamfer"])
    times = np.array(data["time"])
    pcs = data["pc"]    
    # print(pcs[0].shape)
    
    thres_idxs = []
    for thres in thresholds:    
        idx = next((idx for idx, val in enumerate(chamfers) if val <= thres), -1)
        thres_idxs.append(idx)
    
    thres_idxs = np.array(thres_idxs)    
    # print("thres_idxs:", thres_idxs)                                 
    # print("times (minutes):", times[thres_idxs]/60)
    times_at_specific_threshold.append(round(times[thres_idxs[3]]/60, 2)) # at 0.4 chamfer threshold
    
    # pcd_goal = pcd_ize(goal_pc, color=[1,0,0])
    
    # for j in range(10):
    #     pcd = pcd_ize(pcs[thres_idxs[j]], color=[0,0,0])
    #     print(f"threshold: {thresholds[j]:2f}; actual: {compute_chamfer_dist(pcd, pcd_goal):2f}")        
    #     open3d.visualization.draw_geometries([pcd, pcd_goal])
    
    all_min_chamfers.append(round(min(chamfers), 2))
    
# print("all_min_chamfers:", ", ".join([str(elem) for elem in all_min_chamfers]))
print("times_at_specific_threshold:", times_at_specific_threshold)
print("min, max, mean, std:", min(times_at_specific_threshold), max(times_at_specific_threshold), \
        round(np.mean(times_at_specific_threshold),2), round(np.std(times_at_specific_threshold),2))

