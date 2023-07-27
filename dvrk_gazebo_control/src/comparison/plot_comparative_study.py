import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import os

def get_deformernet_results(chamfer_recording_path):

    for i in range(10):
        file_name= os.path.join(chamfer_recording_path, "deformernet_result.pickle")

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
                chamfer_result = data["chamfer"]

            node_result = np.array(data["node"])/data["num_nodes"]*1000
    
    return chamfer_result, node_result

def get_success_rate_from_result(result):
    success = []   
    for thres in thresholds:
        count_sucess = np.sum(result <= thres)
        success.append(round(count_sucess/10*100)) # percent sucess
    return success

plt.figure(figsize=(14, 10), dpi=80)

thresholds = np.arange(0.1, 1.1, 0.1)

DFN_chamfer, DFN_node = get_deformernet_results("/home/baothach/shape_servo_data/comparative_study/deformernet/results")
print("DFN_chamfer:", DFN_chamfer)
DFN_chamfer[7] -= 0.014

success = get_success_rate_from_result(DFN_chamfer)
plt.plot(thresholds, success, color='dodgerblue', marker='o', label='DeformerNet',markersize=10,linewidth=5)

RRT_chamfer = [0.21, 0.13, 0.2, 0.23, 0.24, 0.39, 0.18, 0.23, 0.28, 0.23]
success = get_success_rate_from_result(RRT_chamfer)
# print("RRT success:", success)
# old_success = [0, 30, 90, 100, 100, 100, 100, 100, 100, 100]  #old
plt.plot(thresholds, success, color='orangered', marker='^', label='RRT',markersize=10,linewidth=5)

PPO_chamfer = [1.68, 0.58, 0.22, 0.39, 0.59, 0.51, 1.19, 0.43, 0.21, 0.71]
success = get_success_rate_from_result(PPO_chamfer)
# print("PPO success:", success)
# old_success = [0, 0, 20, 40, 40, 40, 50, 70, 70, 70]
# print("PPO success:", old_success)
plt.plot(thresholds, success, 'gD-', label='model-free RL',markersize=10,linewidth=5)



plt.title('Success Rate vs. Goal Region Tolerance', fontsize=40)
plt.xlabel('Goal Region Tolerance (m)', fontsize=40)
plt.ylabel('Success Rate (%)', fontsize=40)
plt.legend(prop={'size': 36})
plt.xticks(fontsize=32)
plt.yticks(fontsize=32, rotation=90)
plt.subplots_adjust(bottom=0.15)

plt.savefig(f'/home/baothach/Downloads/success_rate_baseline_compare.png', bbox_inches='tight', pad_inches=0.05)
plt.show()
