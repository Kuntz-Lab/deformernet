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

def get_results(prim_name, stiffness, inside, range_data):
    obj_type = f"{prim_name}_{stiffness}"

    object_category = f"{prim_name}_{stiffness}"
    main_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_{object_category}Pa/evaluate"
    distribution_keyword = "inside" if inside else "outside"
    

    chamfer_data = []
    chamfer_data_avg = []
    for i in range_data:
        file_name= os.path.join(main_path, "chamfer_results", object_category, distribution_keyword, f"{prim_name}_{str(i)}.pickle")
        # print(file_name)
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
inside_options = [True]   #[True, False]
all_filtered_idxs = []
nodes = []
chamfers = []

def filtering_condition(chamf_res):
    # return np.where(chamf_res < 900)[0]
    # return np.where(chamf_res < 1.2)[0]
    return np.where(chamf_res < 2)[0]

object_category = []
filtered_results = []
metrics = []

range_data = np.arange(100)

for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    print("=====================================")
    print(f"{prim_name}_{stiffness}", inside)

    # fig = plt.figure()

    chamfer_results = []
    chamfer_avg_results = []


    res, res_avg = get_results(prim_name, stiffness, inside, range_data=range_data)



    filtered_idxs = filtering_condition(res)
    print("filtered_idxs:", filtered_idxs.shape)
    
    filtered_idxs_2 = np.where(res_avg < 3)[0]
    filtered_idxs = np.array(list(set.intersection(set(filtered_idxs), set(filtered_idxs_2))))
    all_filtered_idxs.append(deepcopy(len(filtered_idxs)))
    print(f"filtered_idxs {stiffness}:", filtered_idxs.shape)

    nodes.extend(list(res_avg))
    chamfers.extend(list(res))
    
nodes = np.array(nodes)
chamfers = np.array(chamfers)

sum_filtered_len = sum(all_filtered_idxs)
# target_idx = round(sum_filtered_len*0.25)
# # idx = np.argsort(nodes)[max(target_idx-10,0):target_idx] 
# idx = np.argsort(chamfers)[max(target_idx-10,0):target_idx]

# idx = np.argsort(nodes)[0:10] 
idx = np.argsort(chamfers)[0:10] 
print("Idx of interest:", idx, nodes[idx], chamfers[idx])

# final_idx = 90
# print("final idx: {} {:.3f} {:.3f}".format(final_idx, nodes[final_idx], chamfers[final_idx]))
