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
    if unseen_obj_name is None:
        object_category = f"{prim_name}_{stiffness}"
        main_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_{object_category}Pa/evaluate"
        distribution_keyword = "inside" if inside else "outside"
        

    chamfer_data = []
    chamfer_data_avg = []
    for i in range_data:
        if unseen_obj_name is None:
            file_name = os.path.join(main_path, "chamfer_results", object_category, distribution_keyword, f"{prim_name}_{str(i)}.pickle")
        else:
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

prim_names = ["box", "cylinder", "hemis"] #["box", "cylinder", "hemis"]
stiffnesses = ["1k", "5k", "10k"] #["1k", "5k", "10k"]
inside_options = [True]   #[True, False]
unseen_objects_list = ["chicken_breast"] 

def filtering_condition(chamf_res):
    # return np.where(chamf_res < 900)[0]
    return np.where(chamf_res < 2)[0]

random_colors = []
for i in range(9):
    random_color = random_colors.append(list(np.random.uniform(low=0, high=0.7, size=3)))

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--metric', default="node", type=str, help="node or chamfer")
args = parser.parse_args()    
    
object_category = []
filtered_results = []
metrics = []
metric = args.metric  #node  chamfer

range_data = np.arange(100)

for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    print("=====================================")
    print(f"{prim_name}_{stiffness}", inside)

    # fig = plt.figure()

    chamfer_results = []
    chamfer_avg_results = []


    res, res_avg = get_results(prim_name, stiffness, inside, range_data=range_data)
    filtered_idxs = filtering_condition(res)
    # print("filtered_idxs:", filtered_idxs.shape)
    
    filtered_idxs_2 = np.where(res_avg < 3)[0]
    filtered_idxs = np.array(list(set.intersection(set(filtered_idxs), set(filtered_idxs_2))))
    # print("filtered_idxs:", filtered_idxs.shape)

    if metric == "node":
        new_filtered_result = list(res_avg[filtered_idxs]) 
    elif metric == "chamfer":
        new_filtered_result = list(res[filtered_idxs])


    for _ in range(2): # 1 for the specific object category, 1 for the combined.
        filtered_results += new_filtered_result


    object_category += 1*[f"{prim_name} {stiffness}"]*filtered_idxs.shape[0] + 1*[f"combined {stiffness}"]*filtered_idxs.shape[0]

# print(len(filtered_results))
# print(len(object_category))      

for unseen_obj_name in unseen_objects_list:
    res, res_avg = get_results(prim_name, stiffness, inside, range_data=range_data, unseen_obj_name=unseen_obj_name)
    filtered_idxs = filtering_condition(res)
    
    filtered_idxs_2 = np.where(res_avg < 2.5)[0]
    filtered_idxs = np.array(list(set.intersection(set(filtered_idxs), set(filtered_idxs_2))))
    
    # print(filtered_idxs)

    if metric == "node":
        new_filtered_result = list(res_avg[filtered_idxs]) 
    elif metric == "chamfer":
        new_filtered_result = list(res[filtered_idxs])    

    filtered_results += new_filtered_result
    object_category += 1*[f"{unseen_obj_name}"]*filtered_idxs.shape[0]

# print(len(filtered_results))  
# print(len(object_category))  

df =  pd.DataFrame()
df["chamfer"] = filtered_results
df["object category"] = object_category
# df["Evaluation Metric"] = metrics

plt.figure(figsize=(14, 10), dpi=80)

# ax=sns.boxplot(y="chamfer",x="object category", data=df, whis=1000, showfliers = True)


order = [f"{prim_name} {stiffness}" for (prim_name, stiffness) in list(product(prim_names, stiffnesses))] 
for stiffness in stiffnesses:
    order += [f"combined {stiffness}"]

for unseen_obj_name in unseen_objects_list:      
    order += [unseen_obj_name]  

ax=sns.boxplot(y="chamfer",x="object category", data=df, whis=1000, showfliers = True, order=order) 

        

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

for i in [-3 - len(unseen_objects_list)]:
    ax.artists[i].set_facecolor((0.7, 0, 0))
for i in [-2 - len(unseen_objects_list)]:
    ax.artists[i].set_facecolor((0, 0.7, 0))
for i in [-1 - len(unseen_objects_list)]:
    ax.artists[i].set_facecolor((0, 0, 0.7))    
for i in range(-3 - len(unseen_objects_list), 0 - len(unseen_objects_list)):
    ax.artists[i].set_linewidth(4)   
    ax.artists[i].set_edgecolor('darkgoldenrod') 


# for i in [-3]:
#     ax.artists[i].set_facecolor((0.7, 0, 0))
# for i in [-2]:
#     ax.artists[i].set_facecolor((0, 0.7, 0))
for i in [-1]:
    ax.artists[i].set_facecolor((0.7, 0, 0.7))    
    
# ax.get_legend().remove()
plt.savefig(f'/home/baothach/Downloads/bimanual_{metric}_unseen_obj.png', bbox_inches='tight', pad_inches=0.0)
    
plt.show()

