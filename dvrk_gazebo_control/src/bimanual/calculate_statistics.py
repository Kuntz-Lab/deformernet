import numpy as np
import os
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import open3d
import pickle

result_dir = "/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/evaluate/chamfer_results_w_pred_mp"
# result_dir = "/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/evaluate/chamfer_results"

num_lost_contact = 0

def get_results(result_dir):
    chamfers = []
    for i in range(len(os.listdir(result_dir))):
        with open(os.path.join(result_dir, "sample " + str(i) + ".pickle"), 'rb') as handle:
            data = pickle.load(handle)
        
        if data < 999:
            chamfers.append(data)
        else:
            # print("huhuhu:", data, i)
            # num_lost_contact += 1
            chamfers.append(data)
            # chamfers.append(data-999)
    return chamfers


chamfers = get_results(result_dir="/home/baothach/shape_servo_data/bimanual/box/evaluate/chamfer_results")

# # print(chamfers[:2])
# mean = np.mean(chamfers)
# std = np.std(chamfers)
# max = np.max(chamfers)
# min = np.min(chamfers)
# # max = np.sort(chamfers)[-10:]
# print(mean, std, max, min)

# largest_idx = np.argsort(chamfers)[-1]
# largest_idx = np.argsort(chamfers)[-17-num_lost_contact]
# valid_length = len(chamfers)-num_lost_contact
# largest_idx = np.argsort(chamfers)[(valid_length*3)//4]
# print(largest_idx)
# print(chamfers[largest_idx])
# largest_idx = maintain_contact_idxs[largest_idx]
# print(largest_idx)


# # For single box:
# largest_idx = np.argsort(chamfers)[(len(chamfers)*3)//4]
# # largest_idx = np.argsort(chamfers)[0]
# print(largest_idx)
# print(chamfers[largest_idx])

idxs = np.argsort(chamfers)[(len(chamfers)*3)//4-10:(len(chamfers)*3)//4+10]
goal_recording_path =  "/home/baothach/shape_servo_data/bimanual/box/evaluate/goals"

for i in idxs:
    print("================")
    print(i)
    print(chamfers[i])    
    with open(os.path.join(goal_recording_path, "sample " + str(i) + ".pickle"), 'rb') as handle:
    # with open(os.path.join(goal_recording_path, "sample " + str(data_point_count) + ".pickle"), 'rb') as handle:    
        data = pickle.load(handle)
        goal_pc_numpy = data["full pcs"][1]

        pcd_goal = open3d.geometry.PointCloud()
        pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
        open3d.visualization.draw_geometries([pcd_goal]) 

#===========================
# chamfers_gt = get_results(result_dir="/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/evaluate/chamfer_results")
# chamfers_pred_mp = get_results(result_dir="/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/evaluate/chamfer_results_w_pred_mp")

# df =  pd.DataFrame()
# df["chamfer"] = chamfers_gt + chamfers_pred_mp
# # df["method"] = ["bimanual with ground truth MP"]*len(chamfers) 
# df["method"] = ["with ground truth MP"]*len(chamfers_gt) + ["with predicted MP"]*len(chamfers_pred_mp)

# ax=sns.boxplot(y="chamfer",x='method', data=df, showfliers = False)


# # plt.title('Evaluate bimanual DeformerNet (with only position displacement and single box geometry)', fontsize=12)
# plt.title('Bimanual DeformerNet using ground truth vs predicted MP', fontsize=12)
# plt.xlabel('Method type',fontsize=12)
# plt.ylabel('Chamfer Distance (m)', fontsize=11)
# plt.show()