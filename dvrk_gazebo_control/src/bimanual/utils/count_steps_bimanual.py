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
import argparse
np.random.seed(2019)

def get_results(prim_name, stiffness, inside, range_data, unseen_obj_name=None):
    obj_type = f"{prim_name}_{stiffness}"

    object_category = f"{prim_name}_{stiffness}"
    main_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_{object_category}Pa/evaluate"
    distribution_keyword = "inside" if inside else "outside"
    
    step_counts = []
    
    for i in range_data:
        if unseen_obj_name is None:
            file_name= os.path.join(main_path, "chamfer_results_count_steps", object_category, distribution_keyword, f"{prim_name}_{str(i)}.pickle")
        else:
            unseen_objects_main_path = "/home/baothach/shape_servo_data/rotation_extension/bimanual/unseen_objects/evaluate/chamfer_results_2/model_combined"
            file_name = os.path.join(unseen_objects_main_path, unseen_obj_name, f"{unseen_obj_name}_{i}.pickle")
        # print(file_name)
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
                step_counts.append(data["step_count"])

    return step_counts
    
    

prim_names = ["box", "cylinder", "hemis"] #["box", "cylinder", "hemis"]
stiffnesses = ["1k", "5k", "10k"] #["1k", "5k", "10k"]
inside_options = [True]   #[True, False]


range_data = np.arange(100)
all_avg_step_counts = []


for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    # print("=====================================")
    # print(f"{prim_name}_{stiffness}", inside)

    step_counts = get_results(prim_name, stiffness, inside, range_data=range_data)
    # print(len(step_counts))
    all_avg_step_counts.append(np.round(np.mean(step_counts), decimals=1))
    
print(all_avg_step_counts)

print("\n============== Unseen objects:")
unseen_objects_list = ["chicken_breast"] 
for unseen_obj_name in unseen_objects_list:
    step_counts = get_results(prim_name, stiffness, inside, range_data=range_data)
    print(f"{unseen_obj_name}: {np.round(np.mean(step_counts), decimals=1)}")



