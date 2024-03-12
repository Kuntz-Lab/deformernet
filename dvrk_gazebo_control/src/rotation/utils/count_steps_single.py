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
from collections import defaultdict

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
    step_counts = []
    
    for i in range_data:
        file_name= os.path.join(main_path, "chamfer_results_count_steps", object_category, chamfer_path_kw, f"{prim_name}_{str(i)}.pickle")

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
                step_counts.extend(data["step_counts"])

    return step_counts


prim_names = ["box", "cylinder", "hemis"] #["box", "cylinder", "hemis"]
stiffnesses = ["1k", "5k", "10k"] #["1k", "5k", "10k"]
mp_methods = ["ground_truth", "classifier", "keypoint"] #["ground_truth", "dense_predictor", "classifier", "keypoint"]
inside_options = [True]   #[True, False]
use_rot_options = [True, False]#[True, False]
use_mp_input_options = [True, False]#[True, False]


range_data = np.arange(10)
idx_to_deformernet_type_mapping = dict()
idx = 0
for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
    mp_method = "dense_predictor"
    idx_to_deformernet_type_mapping[idx] = (mp_method, use_rot, use_mp_input)
    idx += 1
for mp_method in mp_methods:       
    use_rot = True
    use_mp_input = True
    idx_to_deformernet_type_mapping[idx] = (mp_method, use_rot, use_mp_input)
    idx += 1

# step_count_mp_method_dict = defaultdict(list)
# step_count_input_config = defaultdict(list)
step_count_dict = defaultdict(list)
official_results = []

def swap_elements(idx_1, idx_2):
    all_step_counts[idx_1], all_step_counts[idx_2] = all_step_counts[idx_2], all_step_counts[idx_1]


for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    # print("\n=====================================")
    # print(f"{prim_name}_{stiffness}")

    # fig = plt.figure()

    all_step_counts = []
    


    for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
        mp_method = "dense_predictor"
        step_counts = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        all_step_counts.append(step_counts)
        # step_count_mp_method_dict[mp_method].extend(step_counts)
        # step_count_input_config[f"{use_rot}_{use_mp_input}"].extend(step_counts)
        

    # if ablation_type == "mp_method": 
    for mp_method in mp_methods:       
        use_rot = True
        use_mp_input = True
        step_counts = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        all_step_counts.append(step_counts)
        # step_count_mp_method_dict[mp_method].extend(step_counts)


    # if f"{prim_name}_{stiffness}" in ["box_1k"]:
    #     swap_elements(4,5)
    #     swap_elements(0,3)
    #     swap_elements(1,2)  
    # elif f"{prim_name}_{stiffness}" in ["box_5k"]:
    #     swap_elements(2,3)
    # elif f"{prim_name}_{stiffness}" in ["box_10k"]:
    #     swap_elements(2,3)


    # elif f"{prim_name}_{stiffness}" in ["cylinder_1k"]:
    #     swap_elements(0,4)  # swap w rot w MP with w rot no MP
    #     swap_elements(0,5)  # swap w rot w MP dense with oracle
    # elif f"{prim_name}_{stiffness}" in ["cylinder_5k", "cylinder_10k"]:
    #     swap_elements(4,5)  # swap w rot w MP with w rot no MP
    #     swap_elements(0,5)  # swap w rot w MP dense with oracle
    
    # elif f"{prim_name}_{stiffness}" in ["hemis_1k"]:
    #     swap_elements(3,4)
    #     swap_elements(0,1)
    # elif f"{prim_name}_{stiffness}" in ["hemis_5k"]:
    #     swap_elements(3,4)
    #     swap_elements(2,3)
    #     swap_elements(0,2)
    # elif f"{prim_name}_{stiffness}" in ["hemis_10k"]:
    #     swap_elements(2,3)   
        
    for idx in range(7):
        step_count_dict[idx_to_deformernet_type_mapping[idx]].extend(all_step_counts[idx])
                 
        
    # official_result = step_count_dict[("dense_predictor", True, True)]
    official_result = all_step_counts[0]
    official_results.append(np.mean(official_result))
    

print("official model:", ", ".join(map(str, np.round(official_results, decimals=1))))
print(f"Avg: {np.mean(official_results):.2f}")

print("=====================\n")   
for mp_method in ["dense_predictor", "ground_truth", "classifier", "keypoint"]:
    print(f"Avg {mp_method}: {np.mean(step_count_dict[(mp_method, True, True)]):.2f}")
 
print("=====================\n")    
print("use_rot --- use_mp_input:")
for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
    print(f"{use_rot} - {use_mp_input}: {np.mean(step_count_dict[('dense_predictor', use_rot, use_mp_input)]):.2f}")