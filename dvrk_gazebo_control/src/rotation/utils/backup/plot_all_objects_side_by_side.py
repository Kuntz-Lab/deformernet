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
total_data_length = 0

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
        break

    # for mp_method in mp_methods:       
    #     use_rot = True
    #     use_mp_input = True
    #     res, res_avg = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
    #     chamfer_results.append(res)
    #     chamfer_avg_results.append(res_avg)
    #     print(f"{mp_method} {use_rot} {use_mp_input}: Shape {res.shape} ; Mean {res.mean()}")

    # # swap_elements(2,3)
    # if f"{prim_name}_{stiffness}" in ["cylinder_5k", "cylinder_10k"]:
    #     swap_elements(0,1)  # swap w rot w MP with w rot no MP
        # swap_elements(0,4)  # swap w rot w MP dense with oracle


    filtered_idxs = []        
    for res in chamfer_results:
        filtered_idxs.append(filtering_condition(res))
    filtered_idxs = np.array(list(set.intersection(*filtered_idxs)))
    print("filtered_idxs:", filtered_idxs.shape)

    # filtered_results = []
    print("Filtered results +++++++")
    for res_avg in chamfer_avg_results:
        # filtered_results.append(list(res_avg[filtered_idxs]))
        filtered_results += list(res_avg[filtered_idxs])
        print(res_avg[filtered_idxs].mean())

    # model_type += ["with ori with MP dense"]*filtered_idxs.shape[0] + ["with ori no MP dense"]*filtered_idxs.shape[0] \
    #             + ["no ori with MP dense"]*filtered_idxs.shape[0] + ["no ori no MP dense"]*filtered_idxs.shape[0] \
    #             + ["with ori with MP oracle"]*filtered_idxs.shape[0] \
    #             + ["with ori with MP classifier"]*filtered_idxs.shape[0] + ["with ori with MP keypoint"]*filtered_idxs.shape[0]

    # model_type += ["Our DeformerNet"]*filtered_idxs.shape[0] + ["DeformerNet w/o MP input"]*filtered_idxs.shape[0] \
    #             + ["DeformerNet w/o orientation"]*filtered_idxs.shape[0] + ["DeformerNet w/o MP input and orientation"]*filtered_idxs.shape[0] 

    model_type += [f"{prim_name}_{stiffness}"]*filtered_idxs.shape[0] 

    # categories += [f"{prim_name}_{stiffness}Pa"]*filtered_idxs.shape[0]*7
    categories += [" "]*filtered_idxs.shape[0]*1
    
    total_data_length += filtered_idxs.shape[0]

model_type += [f"COMBINED"]*total_data_length
filtered_results = np.concatenate((filtered_results, filtered_results), axis=None)
categories += [" "]*total_data_length

df =  pd.DataFrame()
df["chamfer"] = filtered_results
# df["obj name"] = object_names
df["model type"] = model_type
# df["category"] = categories
df[" "] = categories

plt.figure(figsize=(10, 8), dpi=80)
# ax=sns.boxplot(y="chamfer",x='category', hue='model type', data=df, whis=100, showfliers = True) #, whis=3.5
ax=sns.boxplot(y="chamfer",x=' ', hue='model type', data=df, whis=100, showfliers = True) #, whis=3.5

# # plt.title('New model (with orientation) vs old model (w/o) usi}"|ng Node Dist', fontsize=16)

plt.title('All Objects Combined', fontsize=16)
# plt.xlabel('Category',fontsize=16)
plt.ylabel('Node Distance Average (mm)', fontsize=16)

plt.ylim([0,3])

plt.show()
