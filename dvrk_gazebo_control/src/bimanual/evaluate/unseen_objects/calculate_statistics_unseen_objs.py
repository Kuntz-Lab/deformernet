main_dir = "/home/baothach/shape_servo_data/rotation_extension/bimanual/unseen_objects/evaluate/chamfer_results/model_combined"
obj_name = "chicken_breast"

import numpy as np
import pickle5 as pickle
import os
from copy import deepcopy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import timeit
from itertools import product
np.random.seed(2023)

def get_results(range_data, unseen_obj_name=None):
       
    chamfer_data = []
    chamfer_data_avg = []
    for i in range_data:
        unseen_objects_main_path = "/home/baothach/shape_servo_data/rotation_extension/bimanual/unseen_objects/evaluate/chamfer_results/model_combined"
        file_name = os.path.join(unseen_objects_main_path, unseen_obj_name, f"{unseen_obj_name}_{i}.pickle")

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                # data = pickle.load(handle)
                # data = pickle.load(handle)["chamfer"]
                data_avg = pickle.load(handle)
                # data = data_avg["node"]
                data = data_avg["chamfer"]
                
                
          
            chamfer_data.extend(data)
            # chamfer_data.extend([d if d <= 1 else 999 for d in data])
            # chamfer_data.extend([d if d < 900 else d-999 for d in data])
            
            #     
            
            # chamfer_data_avg.extend(data_avg["node"])
            chamfer_data_avg.extend(list(np.array(data_avg["node"])/data_avg["num_nodes"]*1000))
    
    return np.array(chamfer_data), np.array(chamfer_data_avg) 

unseen_objects_list = ["chicken_breast"] 
all_filtered_idxs = []
nodes = []
chamfers = []

def filtering_condition(chamf_res):
    return np.where(chamf_res < 900)[0]
    # return np.where(chamf_res < 1.2)[0]

object_category = []
filtered_results = []
metrics = []

range_data = np.arange(100)

for unseen_obj_name in unseen_objects_list:
    print("=====================================")
    print(f"{unseen_obj_name}")

    # fig = plt.figure()

    chamfer_results = []
    chamfer_avg_results = []


    res, res_avg = get_results(range_data, unseen_obj_name)



    filtered_idxs = filtering_condition(res)
    print("filtered_idxs:", filtered_idxs.shape)
    
    filtered_idxs_2 = np.where(res_avg < 3)[0]
    filtered_idxs = np.array(list(set.intersection(set(filtered_idxs), set(filtered_idxs_2))))
    print("filtered_idxs:", filtered_idxs.shape)


    print("Filtered results +++++++")
    
    nodes = res_avg#[filtered_idxs]
    chamfers = res#[filtered_idxs]

    # print("Node:", np.mean(nodes), np.max(nodes), np.min(nodes))
    # print("Chamfer:", np.mean(chamfers), np.max(chamfers), np.min(chamfers))
    target_idx = round(filtered_idxs.shape[0]*0.75)
    idx = np.argsort(nodes)[max(target_idx-10,0):target_idx] 
    print("Idx of interest:", idx)
    print(nodes[idx])
    print(chamfers[idx])
    

    # print("Node quattiles:", np.sort(nodes)[np.array([0,round(filtered_idxs.shape[0]*0.25),round(filtered_idxs.shape[0]*0.5),\
    #                                     round(filtered_idxs.shape[0]*0.75), filtered_idxs.shape[0]-1])])
    # print("Chamfer quattiles:", np.sort(chamfers)[np.array([0,round(filtered_idxs.shape[0]*0.25),round(filtered_idxs.shape[0]*0.5),\
    #                                     round(filtered_idxs.shape[0]*0.75), filtered_idxs.shape[0]-1])])