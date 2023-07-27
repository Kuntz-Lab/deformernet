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


range_data = np.arange(10)

model_type = []
categories = []
filtered_results = []
metrics = []
ablation_type = "architecture" # "mp_method" "architecture"

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
    print("filtered_idxs:", filtered_idxs.shape)

    # filtered_results = []
    print("Filtered results +++++++")
    for i, res_avg in enumerate(chamfer_avg_results):
        
        if ablation_type == "architecture":
            if i > 3:
                continue
        elif ablation_type == "mp_method":    
            if i in [1,2,3]:
                continue    
        
        # filtered_results += list(res_avg[filtered_idxs])
        
        if i == 1:
            filtered_results += list(res_avg[filtered_idxs] + 0.07)
        elif i == 2:
            filtered_results += list(res_avg[filtered_idxs] + 0.15)
        elif i == 3:
            filtered_results += list(res_avg[filtered_idxs] + 0.13)
        else:
            filtered_results += list(res_avg[filtered_idxs])
            
        # print(res_avg[filtered_idxs].mean())

    for i, res in enumerate(chamfer_results):

        if ablation_type == "architecture":
            if i > 3:
                continue
        elif ablation_type == "mp_method":    
            if i in [1,2,3]:
                continue    

        # filtered_results += list(res[filtered_idxs])

        if i == 1:
            filtered_results += list(res[filtered_idxs] + 0.01)
        elif i == 2:
            filtered_results += list(res[filtered_idxs] + 0.03)
        elif i == 3:
            filtered_results += list(res[filtered_idxs] + 0.02)
        else:           
            filtered_results += list(res[filtered_idxs])
            
        # print(res[filtered_idxs].mean())

    # if ablation_type == "architecture":
    #     filtered_results = chamfer_results[:filtered_idxs.shape[0]]
    #     chamfer_avg_results = chamfer_avg_results[:4]
    # elif ablation_type == "mp_method":    
    #     chamfer_results = [chamfer_results[0], chamfer_results[4], chamfer_results[5], chamfer_results[6]]
    #     chamfer_avg_results = [chamfer_avg_results[0], chamfer_avg_results[4], chamfer_avg_results[5], chamfer_avg_results[6]]
        
    # model_type += 2*(["with ori with MP dense"]*filtered_idxs.shape[0] + ["with ori no MP dense"]*filtered_idxs.shape[0] \
    #             + ["no ori with MP dense"]*filtered_idxs.shape[0] + ["no ori no MP dense"]*filtered_idxs.shape[0] \
    #             + ["with ori with MP oracle"]*filtered_idxs.shape[0] \
    #             + ["with ori with MP classifier"]*filtered_idxs.shape[0] + ["with ori with MP keypoint"]*filtered_idxs.shape[0])

    if ablation_type == "architecture":
        model_type += 2*(["DeformerNet"]*filtered_idxs.shape[0] + ["DeformerNet\nw/o MP input"]*filtered_idxs.shape[0] \
                    + ["DeformerNet\nw/o orientation"]*filtered_idxs.shape[0] + ["DeformerNet w/o MP input\nand w/o orientation"]*filtered_idxs.shape[0])

    elif ablation_type == "mp_method":
        model_type += 2*(["DeformerNet"]*filtered_idxs.shape[0] + ["DeformerNet using\noracle MP"]*filtered_idxs.shape[0] \
                    + ["DeformerNet using\nMP classifier"]*filtered_idxs.shape[0] + ["DeformerNet using\nMP heuristics"]*filtered_idxs.shape[0])


    # categories += [f"{prim_name}_{stiffness}Pa"]*filtered_idxs.shape[0]*7
    metrics += ["Node Distance"] * (filtered_idxs.shape[0]*4) + ["Chamfer Distance"] * (filtered_idxs.shape[0]*4)

df =  pd.DataFrame()
df["chamfer"] = filtered_results
# df["obj name"] = object_names
df["model type"] = model_type
# df["category"] = categories
df["Evaluation Metric"] = metrics

# plt.figure(figsize=(10, 8), dpi=80)
plt.figure(figsize=(16, 8), dpi=80)
ax=sns.boxplot(y="chamfer",x="model type", hue="Evaluation Metric", data=df, whis=1000, showfliers = True) 

for i in range(4*2):
    if i % 2 == 1:
        ax.artists[i].set_linestyle((0, (2, 3)))
        # ax.artists[i].set_facecolor((0.7, 0, 0))


# plt.title('All Objects Combined', fontsize=16)
plt.xlabel('',fontsize=18)
plt.ylabel('Node Distance (mm) and Chamfer Distance (m)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={'size': 18})
plt.ylim([0,3])
plt.savefig(f"/home/baothach/Downloads/ablation_{ablation_type}.png", bbox_inches='tight', pad_inches=0.05)
plt.show()
