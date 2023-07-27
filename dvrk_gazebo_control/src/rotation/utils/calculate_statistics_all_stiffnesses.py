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
from matplotlib.pyplot import figure
np.random.seed(2023)

def get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data):
    obj_type = f"{prim_name}_{stiffness}"
    if use_rot and use_mp_input:
        method_type = "w_rot_w_MP"
    elif use_rot and (not use_mp_input):    
        method_type = "w_rot_no_MP"
    elif (not use_rot) and use_mp_input:    
        method_type = "no_rot_w_MP"
    elif (not use_rot) and (not use_mp_input):    
        method_type = "no_rot_no_MP"

    object_category = f"{prim_name}_{stiffness}"
    main_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{object_category}Pa/evaluate"
    distribution_keyword = "inside" if inside else "outside"
    chamfer_path_kw = f"{distribution_keyword}_{mp_method}_{method_type}"
    

    chamfer_data = []
    chamfer_data_avg = []
    for i in range_data:
        file_name= os.path.join(main_path, "chamfer_results", object_category, chamfer_path_kw, f"{prim_name}_{str(i)}.pickle")

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

prim_names = ["box"] #["box", "cylinder", "hemis"]
stiffnesses = ["1k", "5k", "10k"] #["1k", "5k", "10k"]
mp_methods = ["dense_predictor"] #["ground_truth", "dense_predictor", "classifier", "keypoint"]
inside_options = [True]   #[True, False]
use_rot_options = [True]#[True, False]
use_mp_input_options = [True]#[True, False]

def swap_elements(idx_1, idx_2):
    chamfer_results[idx_1], chamfer_results[idx_2] = chamfer_results[idx_2], chamfer_results[idx_1]
    chamfer_avg_results[idx_1], chamfer_avg_results[idx_2] = chamfer_avg_results[idx_2], chamfer_avg_results[idx_1]

def filtering_condition(chamf_res):
    # return set(np.where(chamf_res < 900)[0])
    return set(np.where(chamf_res < 1.5)[0])


range_data = np.arange(10)

object_category = []
categories = []
filtered_results = []
metrics = []
total_data_length = 0
all_filtered_idxs = []
nodes = []
chamfers = []

for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    print("=====================================")
    print(f"{prim_name}_{stiffness}", inside)

    # fig = plt.figure()

    chamfer_results = []
    chamfer_avg_results = []

    for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
        mp_method = "dense_predictor"
        res, res_avg = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        chamfer_results.append(res)
        chamfer_avg_results.append(res_avg)
        # print(f"{mp_method} {use_rot} {use_mp_input}: Shape {res.shape} ; Mean {res.mean()}")


    filtered_idxs = []        
    # for res_avg in chamfer_avg_results:
    #     filtered_idxs.append(filtering_condition(res_avg))
    for res in chamfer_results:
        filtered_idxs.append(filtering_condition(res))


        
    filtered_idxs = np.array(list(set.intersection(*filtered_idxs)))
    all_filtered_idxs.append(deepcopy(len(filtered_idxs)))
    print(f"filtered_idxs {stiffness}:", filtered_idxs.shape)

    # filtered_results = []
    # print("Filtered results +++++++")
    
    nodes.extend(list(res_avg))
    chamfers.extend(list(res))

    # print("Node:", np.mean(nodes), np.max(nodes), np.min(nodes))
    # print("Chamfer:", np.mean(chamfers), np.max(chamfers), np.min(chamfers))
    # idx = np.argsort(nodes)[75]
    # print("Idx of interest:", idx, nodes[idx], chamfers[idx])

nodes = np.array(nodes)
chamfers = np.array(chamfers)

sum_filtered_len = sum(all_filtered_idxs)
target_idx = round(sum_filtered_len*0.75)-1
idx = np.argsort(chamfers)[max(target_idx-10,0):target_idx] 
# idx = np.argsort(chamfers)[max(target_idx-10,0):target_idx]

# # idx = np.argsort(nodes)[0:10] 
# idx = np.argsort(chamfers)[0:10] 
print("Idx of interest:", idx, nodes[idx], chamfers[idx])

final_idx = 103
print("final idx: {} {:.3f} {:.3f}".format(final_idx, nodes[final_idx], chamfers[final_idx]))


# print("Node quattiles:", np.sort(nodes)[np.array([0,round(filtered_idxs.shape[0]*0.25),round(filtered_idxs.shape[0]*0.5),\
#                                     round(filtered_idxs.shape[0]*0.75), filtered_idxs.shape[0]-1])])
# print("Chamfer quattiles:", np.sort(chamfers)[np.array([0,round(filtered_idxs.shape[0]*0.25),round(filtered_idxs.shape[0]*0.5),\
#                                     round(filtered_idxs.shape[0]*0.75), filtered_idxs.shape[0]-1])])
    