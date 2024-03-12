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

model_type = []
categories = []
filtered_results = []
metrics = []
ablation_type = "mp_method" # "mp_method" "architecture"
step_count_mp_method_dict = defaultdict(list)
step_count_input_config = defaultdict(list)


for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    # print("\n=====================================")
    # print(f"{prim_name}_{stiffness}")

    # fig = plt.figure()

    all_step_counts = []
    


    for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
        mp_method = "dense_predictor"
        step_counts = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        all_step_counts.extend(step_counts)
        if use_rot and use_mp_input:
            step_count_mp_method_dict[mp_method].extend(step_counts)
        step_count_input_config[f"{use_rot}_{use_mp_input}"].extend(step_counts)
        

    # if ablation_type == "mp_method": 
    for mp_method in mp_methods:       
        use_rot = True
        use_mp_input = True
        step_counts = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        all_step_counts.extend(step_counts)
        step_count_mp_method_dict[mp_method].extend(step_counts)
        
        
    # print(f"Average {prim_name}_{stiffness}: {np.mean(all_step_counts):.2f}")

print("=====================\n") 
for mp_method in ["dense_predictor", "ground_truth", "classifier", "keypoint"]:
    print(f"Avg {mp_method}: {np.mean(step_count_mp_method_dict[mp_method]):.2f}")
 
    
print("=====================\n")    
print("use_rot --- use_mp_input:")
for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
    print(f"{use_rot} - {use_mp_input}: {np.mean(step_count_input_config[f'{use_rot}_{use_mp_input}']):.2f}")