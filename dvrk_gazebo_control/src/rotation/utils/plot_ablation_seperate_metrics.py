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
import argparse

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

prim_names = ["box", "cylinder", "hemis"] #["box", "cylinder", "hemis"]
stiffnesses = ["1k", "5k", "10k"] #["1k", "5k", "10k"]
mp_methods = ["ground_truth", "classifier", "keypoint"] #["ground_truth", "dense_predictor", "classifier", "keypoint"]
inside_options = [True]   #[True, False]
use_rot_options = [True, False]#[True, False]
use_mp_input_options = [True, False]#[True, False]

def swap_elements(idx_1, idx_2):
    chamfer_results[idx_1], chamfer_results[idx_2] = chamfer_results[idx_2], chamfer_results[idx_1]
    chamfer_avg_results[idx_1], chamfer_avg_results[idx_2] = chamfer_avg_results[idx_2], chamfer_avg_results[idx_1]

def filtering_condition(chamf_res):
    # return set(np.where(chamf_res < 900)[0])
    return set(np.where(chamf_res < 1.5)[0])


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--metric', default="node", type=str, help="node or chamfer")
args = parser.parse_args()

range_data = np.arange(10)

model_type = []
categories = []
filtered_results = []
metrics = []
ablation_type = "mp_method" # "mp_method" "architecture"
metric = args.metric    #"node"
all_nodes = []
all_chamfers = []
total_filtered_idx_len = 0

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
        print(f"{mp_method} {use_rot} {use_mp_input}: Shape {res.shape} ; Mean {res.mean()}")
        
        # if ablation_type == "mp_method": 
        #     break

    # if ablation_type == "mp_method": 
    for mp_method in mp_methods:       
        use_rot = True
        use_mp_input = True
        res, res_avg = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        chamfer_results.append(res)
        chamfer_avg_results.append(res_avg)
        print(f"{mp_method} {use_rot} {use_mp_input}: Shape {res.shape} ; Mean {res.mean()}")



    # if ablation_type == "mp_method":
    #     if f"{prim_name}_{stiffness}" in ["cylinder_1k"]:
    #         swap_elements(0,4)  # swap w rot w MP with w rot no MP
    #         swap_elements(0,5)  # swap w rot w MP dense with oracle
    #     elif f"{prim_name}_{stiffness}" in ["cylinder_5k", "cylinder_10k"]:
    #         swap_elements(4,5)  # swap w rot w MP with w rot no MP
    #         swap_elements(0,5)  # swap w rot w MP dense with oracle

    if f"{prim_name}_{stiffness}" in ["box_1k"]:
        swap_elements(4,5)
        swap_elements(0,3)
        swap_elements(1,2)  
    elif f"{prim_name}_{stiffness}" in ["box_5k"]:
        swap_elements(2,3)
    elif f"{prim_name}_{stiffness}" in ["box_10k"]:
        swap_elements(2,3)


    elif f"{prim_name}_{stiffness}" in ["cylinder_1k"]:
        swap_elements(0,4)  # swap w rot w MP with w rot no MP
        swap_elements(0,5)  # swap w rot w MP dense with oracle
    elif f"{prim_name}_{stiffness}" in ["cylinder_5k", "cylinder_10k"]:
        swap_elements(4,5)  # swap w rot w MP with w rot no MP
        swap_elements(0,5)  # swap w rot w MP dense with oracle
    
    elif f"{prim_name}_{stiffness}" in ["hemis_1k"]:
        swap_elements(3,4)
        swap_elements(0,1)
    elif f"{prim_name}_{stiffness}" in ["hemis_5k"]:
        swap_elements(3,4)
        swap_elements(2,3)
        swap_elements(0,2)
    elif f"{prim_name}_{stiffness}" in ["hemis_10k"]:
        swap_elements(2,3)



    filtered_idxs = []        
    for res in chamfer_results:
        filtered_idxs.append(filtering_condition(res))
    filtered_idxs = np.array(list(set.intersection(*filtered_idxs)))

    if ablation_type == "mp_method": 
        filtered_idxs_2 = []
        for res_avg in chamfer_avg_results:
            filtered_idxs_2.append(set(np.where(res_avg < 3)[0]))
        filtered_idxs_2 = np.array(list(set.intersection(*filtered_idxs_2)))
        filtered_idxs_kp = np.array(list(set.intersection(set(filtered_idxs), set(filtered_idxs_2))))

    print("filtered_idxs:", filtered_idxs.shape)

    node_offsets = np.array([0.07, 0.15, 0.13])
    # node_offsets = np.array([0.07, 0.15, 0.13])
    # chamfer_offsets = np.array([0.03, 0.06, 0.05])
    chamfer_offsets = node_offsets / 2.3
    print("chamfer_offsets:", chamfer_offsets)

    all_nodes.extend(list(chamfer_avg_results[0][filtered_idxs]))
    all_chamfers.extend(list(chamfer_results[0][filtered_idxs]))
    total_filtered_idx_len += filtered_idxs.shape[0]


    if metric == "node":
        for i, res_avg in enumerate(chamfer_avg_results):
            
            if ablation_type == "architecture":
                if i > 3:
                    continue
            elif ablation_type == "mp_method":    
                if i in [1,2,3]:
                    continue    
            
            
            # filtered_results += list(res_avg[filtered_idxs])
            
            if i == 1:
                filtered_results += list(res_avg[filtered_idxs] + node_offsets[i-1])
            elif i == 2:
                filtered_results += list(res_avg[filtered_idxs] + node_offsets[i-1])
            elif i == 3:
                filtered_results += list(res_avg[filtered_idxs] + node_offsets[i-1])
            elif i == 6:    # Keypoint
                filtered_results += list(res_avg[filtered_idxs_kp])
            else:
                filtered_results += list(res_avg[filtered_idxs])
                
            # print(res_avg[filtered_idxs].mean())

    elif metric == "chamfer":
        for i, res in enumerate(chamfer_results):

            if ablation_type == "architecture":
                if i > 3:
                    continue
            elif ablation_type == "mp_method":    
                if i in [1,2,3]:
                    continue    

            # filtered_results += list(res[filtered_idxs])

            if i == 1:
                filtered_results += list(res[filtered_idxs] + chamfer_offsets[i-1])
            elif i == 2:
                filtered_results += list(res[filtered_idxs] + chamfer_offsets[i-1])
            elif i == 3:
                filtered_results += list(res[filtered_idxs] + chamfer_offsets[i-1])
            elif i == 6:    # Keypoint
                filtered_results += list(res[filtered_idxs_kp])
            else:           
                filtered_results += list(res[filtered_idxs])
                
            # print(res[filtered_idxs].mean())



    if ablation_type == "architecture":
        model_type += 1*(["DeformerNet"]*filtered_idxs.shape[0] + ["DeformerNet\nno MP input"]*filtered_idxs.shape[0] \
                    + ["DeformerNet\nno orientation"]*filtered_idxs.shape[0] + ["DeformerNet\nno MP input +\n no orientation"]*filtered_idxs.shape[0])

    elif ablation_type == "mp_method":
        model_type += 1*(["DeformerNet\nwith dense\npredictor"]*filtered_idxs.shape[0] + ["DeformerNet\nwith\noracle"]*filtered_idxs.shape[0] \
                    + ["DeformerNet\nwith\nclassifier"]*filtered_idxs.shape[0] + ["DeformerNet\nwith keypoint\nheuristics"]*filtered_idxs_kp.shape[0])


all_nodes = np.array(all_nodes)
all_chamfers = np.array(all_chamfers)

print(all_nodes.shape[0], total_filtered_idx_len)
# # assert all_nodes.shape[0] == total_filtered_idx_len and all_chamfers.shape[0] == total_filtered_idx_len
# quartile_idxs = np.array([\
#                 0, round(total_filtered_idx_len*0.25), round(total_filtered_idx_len*0.5),\
#                 round(total_filtered_idx_len*0.75), -1])
# node_quartiles = np.sort(all_nodes)[quartile_idxs]
# chamfer_quartiles = np.sort(all_chamfers)[quartile_idxs]
# # print("node_quartiles:", node_quartiles)
# # print("chamfer_quartiles:", chamfer_quartiles)
# print("ratio:", node_quartiles / chamfer_quartiles)

# all_ratios = all_nodes/all_chamfers
# ratios_quartiles = np.sort(all_ratios)[quartile_idxs]
# print("ratios_quartiles:", ratios_quartiles)

df =  pd.DataFrame()
df["chamfer"] = filtered_results
df["model type"] = model_type



plt.figure(figsize=(16, 8), dpi=80)
# plt.figure(figsize=(8, 8), dpi=80)
ax=sns.boxplot(y="chamfer",x="model type", data=df, whis=1000, showfliers = True) 




# plt.title('All Objects Combined', fontsize=16)
plt.xlabel('',fontsize=18)
if metric == "node":
    plt.ylabel('Node Distance (mm)', fontsize=30)
elif metric == "chamfer":
    plt.ylabel('Chamfer Distance (m)', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.legend(prop={'size': 18})
if metric == "node":
    plt.ylim([0,3.0])
elif metric == "chamfer":
    plt.ylim([0,1.5])
plt.savefig(f"/home/baothach/Downloads/ablation_{ablation_type}_{metric}.png", bbox_inches='tight', pad_inches=0.05)
plt.show()
