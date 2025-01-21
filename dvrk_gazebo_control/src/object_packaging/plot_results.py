import os
import numpy as np
import open3d
from copy import deepcopy
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pickle5 as pickle

def read_pickle_data(data_path):
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)   

def plot_violin(data, labels):
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
    xlabel = "Models"
    ylabel = "Enclosure Percentage (%)"
    title = "DiffDef vs DefGoalNet, on Object Packaging task"
    ax.set_xlabel(xlabel, fontsize=16)    
    ax.set_ylabel(ylabel, fontsize=16)    
    ax.set_title(title, fontsize=20)
    plt.yticks(fontsize=16)
    # Show the plot
    plt.savefig(f"/home/baothach/Downloads/object_packaging_diffdef_vs_defgoalnet.png", bbox_inches='tight', pad_inches=0.1)   
    # plt.ylim(70, 100)
    plt.show()

def get_result(result_path):
    return read_pickle_data(result_path)["enclosed_percent"]

main_result_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_packaging_multimodal/evaluation/box_5k"

models = ["diffdef", "defgoalnet"]  # ["diffdef", "defgoalnet"]

all_enclosed_percents = []
for model in models:
    per_model_percents = []
    for i in range(0, 100):
        result_path = os.path.join(main_result_path, f"{model}/sample_{i}.pickle")
        if os.path.exists(result_path):
            per_model_percents.append(get_result(result_path))
    all_enclosed_percents.append(per_model_percents) 
        
print(np.array(all_enclosed_percents).shape)
    
labels = ["DiffDef", "DefGoalNet"]
plot_violin(all_enclosed_percents, labels)   



