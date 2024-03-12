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
np.random.seed(2019)

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

random_colors = []
for i in range(9):
    random_color = random_colors.append(list(np.random.uniform(low=0, high=0.7, size=3)))

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--metric', default="node", type=str, help="node or chamfer")
args = parser.parse_args()


range_data = np.arange(10)

object_category = []
categories = []
filtered_results = []
metrics = []
total_data_length = 0
metric = args.metric    #node  chamfer

for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    print("=====================================")
    print(f"{prim_name}_{stiffness}", inside)

    # fig = plt.figure()

    chamfer_results = []
    chamfer_avg_results = []

    for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
        mp_method = "dense_predictor"
        
        # if f"{prim_name}_{stiffness}" in ["cylinder_1k"]:
        #     mp_method = "classifier"
        # elif f"{prim_name}_{stiffness}" in ["cylinder_5k", "cylinder_10k"]:
        #     mp_method = "ground_truth"
        # else:
        #     mp_method = "dense_predictor"
               
        res, res_avg = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        chamfer_results.append(res)
        chamfer_avg_results.append(res_avg)
        print(f"{mp_method} {use_rot} {use_mp_input}: Shape {res.shape} ; Mean {res.mean()}")
        

    for mp_method in mp_methods:       
        use_rot = True
        use_mp_input = True
        res, res_avg = get_results(prim_name, stiffness, inside, mp_method, use_rot, use_mp_input, range_data=range_data)
        chamfer_results.append(res)
        chamfer_avg_results.append(res_avg)
        print(f"{mp_method} {use_rot} {use_mp_input}: Shape {res.shape} ; Mean {res.mean()}")


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


    
    for _ in range(2):
        if metric == "node":
            for res_avg in chamfer_avg_results:
                filtered_results += list(res_avg[filtered_idxs]) 
                break   # only add results from the official version of DeformerNet (dense predictor - True - True)

        elif metric == "chamfer":
            for res in chamfer_results:
                filtered_results += list(res[filtered_idxs]) 
                break   # only add results from the official version of DeformerNet (dense predictor - True - True)
    
    
    object_category += 1*[f"{prim_name} {stiffness}"]*filtered_idxs.shape[0] + 1*[f"combined {stiffness}"]*filtered_idxs.shape[0]

    # metrics += 2*(["Node Distance"] * (filtered_idxs.shape[0]*1) + ["Chamfer Distance"] * (filtered_idxs.shape[0]*1))
    
    # total_data_length += filtered_idxs.shape[0]

# object_category += [f"combined"]*total_data_length
# filtered_results = np.concatenate((filtered_results, filtered_results), axis=None)
# categories += [" "]*total_data_length

print("Data len:", len(filtered_results), len(object_category), len(metrics))

df =  pd.DataFrame()
df["chamfer"] = filtered_results
# df["obj name"] = object_names
df["object category"] = object_category
# df["category"] = categories
# df["Evaluation Metric"] = metrics

plt.figure(figsize=(14, 8), dpi=80)

order = [f"{prim_name} {stiffness}" for (prim_name, stiffness) in list(product(prim_names, stiffnesses))] 
for stiffness in stiffnesses:
    order += [f"combined {stiffness}"]
        
# print(len(filtered_results), len(object_category))
assert len(filtered_results) == 742*2 and len(object_category) == 742*2

ax=sns.boxplot(y="chamfer",x="object category", data=df, whis=1000, showfliers = True, order=order) 
# print("len(ax.artists):", len(ax.artists))
# for i in range(9*1+3*1):
#     if i % 2 == 1:
#         ax.artists[i].set_linestyle((0, (1, 1)))
#         ax.artists[i].set_linewidth(3) 
#         # ax.artists[i].set_facecolor((0.7, 0, 0))


# plt.title('All Objects combined', fontsize=16)
plt.xlabel('',fontsize=16)
if metric == "node":
    plt.ylabel('Node Distance (mm)', fontsize=24)
elif metric == "chamfer":
    plt.ylabel('Chamfer Distance (m)', fontsize=24)
 

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

if metric == "node":
    plt.ylim([0,2.5])
elif metric == "chamfer":
    plt.ylim([0,1.2])

plt.subplots_adjust(bottom=0.2) # Make x axis label (Object) fit

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

# handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles, labels,title='',loc='upper center', bbox_to_anchor=(0.5, 1.11),ncol=5, fancybox=False, shadow=False, prop={'size': 16})
# ax.legend(handles, labels,title='',loc='best', prop={'size': 16})

for i in range(9):
    ax.artists[i].set_facecolor(tuple(random_colors[i]))

for i in [-3]:
    ax.artists[i].set_facecolor((0.7, 0, 0))
for i in [-2]:
    ax.artists[i].set_facecolor((0, 0.7, 0))
for i in [-1]:
    ax.artists[i].set_facecolor((0, 0, 0.7))    
for i in range(-3,0):
    ax.artists[i].set_linewidth(4)   
    ax.artists[i].set_edgecolor('darkgoldenrod') 
    
# ax.get_legend().remove()
plt.savefig(f'/home/baothach/Downloads/side_by_side_{metric}.png', bbox_inches='tight', pad_inches=0.05)
    
plt.show()