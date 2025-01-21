import os
import numpy as np
import open3d
from copy import deepcopy
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import sys
# sys.path.append("../../")
import pickle5 as pickle


def plot_violin(data, labels, metric):
    """
    This function takes a list of data accuracy lists and creates the violin plot.
    """
    fig, ax = plt.subplots(figsize=(9,5))
 
    v_parts = ax.violinplot(data, showmeans=True, showmedians=False)
    cmap = plt.get_cmap('ocean')
    # colors = cmap(np.linspace(0,1,len(labels)))
    # colors = colors
    colors = ['#4daf4a','#650021',
                '#f781bf', '#a65628', '#999999',
                '#e41a1c', '#dede00']
    
    i = 0
    for part , color in zip(v_parts['bodies'], colors):
        if i < 2:
            part.set_facecolor(color)
        else:
            part.set_facecolor('#377eb8')
        i+=1
        part.set_alpha(0.75)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = v_parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(2)
        
 
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=16)
    xlabel = "Point Cloud Frame"
    if metric == "chamfer":
        ylabel = "Chamfer Distance (mm)"
    else:
        ylabel = "Node Distance (m)"
    title = "DeformerNet Performance, on Object Frame vs World Frame"
    ax.set_xlabel(xlabel, fontsize=16)    
    ax.set_ylabel(ylabel, fontsize=16)    
    ax.set_title(title, fontsize=20)
    plt.yticks(fontsize=16)
    # Show the plot
    # plt.savefig(f"/home/baothach/Downloads/test_{metric}.png", bbox_inches='tight', pad_inches=0.1)   
    plt.show()

def get_result(result_path, frame_name):
    results = np.ones(range_data[-1]) * -1
    valid_idxs = []
    for i in range(*range_data):
        file_name= os.path.join(result_path, f"{prim_name}_{str(i)}.pickle")
        # print(file_name)
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
                if metric == "chamfer":
                    result = data["chamfer"][0]
                else:
                    result = (data["node"][0])/data["num_nodes"]*1000
                if result < 1999 and result > 999:
                    continue
                if result > 1999:
                    result -= 1999 
                # if result >= 1.0:
                #     continue
                results[i] = result
                valid_idxs.append(i)

    return results, valid_idxs
 
 
frame_names = ["object", "world"]   # ["object", "world"]
metric = "chamfer"  # options: "chamfer", "node"
prim_name = "box"
stiffness = "1k"

object_category = f"{prim_name}_{stiffness}"
main_result_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{object_category}Pa/evaluate/chamfer_results/" 
range_data = [0, 100]

all_chamfers = []
all_valid_idxs = []
for frame_name in frame_names:   
    result_path = os.path.join(main_result_path, f"{frame_name}_frame")
    chamfers, valid_idxs = get_result(result_path, frame_name)
    all_chamfers.append(chamfers)
    all_valid_idxs.append(valid_idxs)
    
# Filter only same valid indices
valid_idxs = list(set(all_valid_idxs[0]).intersection(all_valid_idxs[1]))
all_chamfers = [np.array(all_chamfers[0])[valid_idxs], np.array(all_chamfers[1])[valid_idxs]]
    
print("np.array(all_chamfers[0]).shape: ", np.array(all_chamfers[0]).shape)
print("np.array(all_chamfers[1]).shape: ", np.array(all_chamfers[1]).shape)
# print(np.array(all_chamfers).shape)
# print("all_chamfers: ", all_chamfers)
print("np.mean(all_chamfers[0]): ", np.mean(all_chamfers[0]))
print("np.mean(all_chamfers[1]): ", np.mean(all_chamfers[1]))
    
labels = ["Object Frame", "World Frame"]
plot_violin(all_chamfers, labels, metric)   



