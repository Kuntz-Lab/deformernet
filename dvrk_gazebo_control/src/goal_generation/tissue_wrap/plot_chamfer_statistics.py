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
import torch
sys.path.append(f"/home/baothach/shape_servo_data/tanner/goal_generation_net")
from pointconv_model_bao import GoalGenNet

# def plot_violin(data, labels, title="Easy Plane"):
#     """
#     This function takes a list of data accuracy lists and creates the violin plot.
#     """
#     fig, ax = plt.subplots(figsize=(20,20))

#     ax.violinplot(data, showmeans=True, showmedians=True)

#     ax.set_xticks(range(1, len(data) + 1))
#     ax.set_xticklabels(labels)
#     xlabel = "Models"
#     ylabel = "Chamfer Distance (m)"
#     title = "Chamfer Distance Between Ground Truth and Predicted Goal Point Clouds" 
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
    ylabel = "Chamfer Distance (mm)"
    # title = "Chamfer Loss of Predicted Goal and Ground Truth Goal" + t  
    ax.set_xlabel(xlabel, fontsize=16)    
    ax.set_ylabel(ylabel, fontsize=16)    
    # ax.set_title(title, fontsize=30)
    plt.yticks(fontsize=16)
    # Show the plot
    plt.savefig("/home/baothach/Downloads/tissue_wrapping_chamfer_loss_norm.png", bbox_inches='tight', pad_inches=0.05)   
    plt.show()
    

device = torch.device("cuda")

data_processed_path = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/processed_eval_data"
goal_tensors = []
for i in range(100):
    goal_pc = read_pickle_data(os.path.join(data_processed_path, f"processed sample {i}.pickle"))["partial pcs"][1]
    goal_tensor = torch.from_numpy(goal_pc).unsqueeze(0).float()
    goal_tensors.append(goal_tensor)
goal_tensors = torch.cat(tuple(goal_tensors), dim=0).to(device)    

# model_names = ["pointconv_1000"]
# model_names = ["pointconv_1000", "pointconv_100"] + [f"randomized/pointconv_10_random_{j}" for j in range(5)]
model_names = ["pointconv_100"]
goal_models = []


goal_model = GoalGenNet(num_points=512, embedding_size=256).to(device)
goal_model.eval()


all_chamfers = []
for model_name in model_names:
    init_tensors = []
    cylinder_tensors = []
    weight_path = f"/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/weights/{model_name}"
    goal_model.load_state_dict(torch.load(f"{weight_path}/epoch 2000"))
    for i in range(100):
        data = read_pickle_data(os.path.join(data_processed_path, f"processed sample {i}.pickle"))
        init_pc = data["partial pcs"][0]
        init_tensor = torch.from_numpy(init_pc).unsqueeze(0).float()
        init_tensors.append(init_tensor)

        cylinder_pc = data["cylinder_pc"]
        cylinder_tensor = torch.from_numpy(cylinder_pc).unsqueeze(0).float()
        cylinder_tensors.append(cylinder_tensor)
        
    init_tensors = torch.cat(tuple(init_tensors), dim=0).to(device)   
    cylinder_tensors = torch.cat(tuple(cylinder_tensors), dim=0).to(device)      
    # print(init_tensors.shape, cylinder_tensors.shape)    

    pred_goal_tensors = goal_model(init_tensors, cylinder_tensors)
    # print(pred_goal_tensors.shape, goal_tensors.shape)
    
    chamfers = []
    for k in range(goal_tensors.shape[0]):    
        chamfer = goal_model.get_chamfer_loss(goal_tensors.permute(0,2,1)[k:k+1], 
                                              pred_goal_tensors.permute(0,2,1)[k:k+1]).detach().cpu().numpy() / 512 * 1000
        chamfers.append(chamfer)
        
    # print(chamfers)
    all_chamfers.append(chamfers)
    
    del init_tensors
    del cylinder_tensors
    del pred_goal_tensors

# labels = ["1k", "100"] + [f"10({j})" for j in range(5)]            
# # plot_violin(all_chamfers, model_names)    
# plot_violin(all_chamfers, labels)        

print(all_chamfers)
print(np.argmax(all_chamfers))
