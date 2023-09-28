import os
import numpy as np
import open3d
import pickle
from copy import deepcopy
import sys
sys.path.append("../../")
from utils.miscellaneous_utils import pcd_ize, read_pickle_data, write_pickle_data
import seaborn as sns
import matplotlib.pyplot as plt

# def plot_violin(data, labels, title="Easy Plane"):
#     """
#     This function takes a list of data accuracy lists and creates the violin plot.
#     """
#     fig, ax = plt.subplots(figsize=(20,20))

#     ax.violinplot(data, showmeans=True, showmedians=True)

#     ax.set_xticks(range(1, len(data) + 1))
#     ax.set_xticklabels(labels)
#     xlabel = "Models"
#     ylabel = "Tissue Coverage Percentage (%)"
#     title = "Tissue Coverage Accross Multiple Models"  
#     ax.set_xlabel(xlabel)    
#     ax.set_ylabel(ylabel)    
#     ax.set_title(title)
#     # Show the plot
#     # plt.ylim(0, 1)    
#     plt.show() 

def plot_violin(data, labels, t = ""):
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
        print(color)
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
    ylabel = "Tissue Coverage Percentage (%)"
    # title = "Chamfer Loss of Predicted Goal and Ground Truth Goal" + t  
    ax.set_xlabel(xlabel, fontsize=16)    
    ax.set_ylabel(ylabel, fontsize=16)    
    # ax.set_title(title, fontsize=30)
    plt.yticks(fontsize=16)
    # Show the plot
    plt.savefig("/home/baothach/Downloads/tissue_wrapping_coverage_percentage.png", bbox_inches='tight', pad_inches=0.05)   
    plt.show()

data_recording_path = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/evaluate"
model_names = ["pointconv_1000", "pointconv_100"] + [f"randomized/pointconv_10_random_{j}" for j in range(5)]
# model_names = [f"randomized/pointconv_100_random_{j}" for j in range(5)] + [f"randomized/pointconv_10_random_{j}" for j in range(5)]

all_percents = []

for model_name in model_names:
    percents = []
    print(f"Running model {model_name} ... \n")
    for i in range(0, 100):        
        save_path = os.path.join(data_recording_path, model_name, f"sample {i}.pickle")
        if not os.path.isfile(save_path):
            continue
        
        data = read_pickle_data(save_path)   

        percent = data["final_percent"]
        percents.append(percent*100)
        
        # if percent <= 0.2 and model_name == "pointconv_1000":
        #     print(i)
    
    all_percents.append(percents)



# # Create a figure and axis
# plt.figure(figsize=(8, 6))

# # Plot the violin plots for each item in the data list
# sns.violinplot(data=all_percents)

# # Customize labels and title
# item_labels = [f"Item {i+1}" for i in range(len(all_percents))]
# plt.xticks(range(len(all_percents)), item_labels)
# plt.xlabel("Items")
# plt.ylabel("Values")
# plt.title("Violin Plot of N Items")

# # Show the plot
# plt.show()
        
labels = ["1k", "100"] + [f"10({j})" for j in range(5)]            
# plot_violin(all_percents, model_names)    
plot_violin(all_percents, labels)   
   
        
