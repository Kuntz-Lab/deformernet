import numpy as np
import pickle
import os
from copy import deepcopy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import open3d


def print_stat(data, category):
    print("------------")
    print(category)
    print("mean:", np.mean(data))
    print("max:", np.max(data))
    print("min:", np.min(data))
    

results_recording_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinders_1000Pa/evaluate/"
# results_recording_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinders_1000Pa/evaluate/results_no_rot"


def get_results(path):
    chamfers = []
    node_dists = []

    for i in [20]:#range(86):
        with open(os.path.join(path, f"cylinder_{i}.pickle"), 'rb') as handle:
            data = pickle.load(handle)

        chamfer_dist = data["chamfer"][0]
        final_full_pc = data["final full pc"]
        goal_full_pc = data["goal full pc"]
        node_dist = np.linalg.norm(goal_full_pc-final_full_pc)

        if chamfer_dist >= 999:
            # continue
            chamfer_dist -= 999

        chamfers.append(chamfer_dist)
        node_dists.append(node_dist)

        # print("chamfer_dist:", chamfer_dist)
        # print("node_dist:", node_dist)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(final_full_pc)   
        pcd.paint_uniform_color([0,0,0])
        pcd_goal = open3d.geometry.PointCloud()
        pcd_goal.points = open3d.utility.Vector3dVector(goal_full_pc)   
        pcd_goal.paint_uniform_color([1,0,0])
        open3d.visualization.draw_geometries([pcd, pcd_goal])

    return np.array(chamfers), node_dists

chamfers, node_dists = get_results(os.path.join(results_recording_path, "results"))
# print_stat(chamfers, category="chamfer")
# print_stat(node_dists, category="node_dist")

chamfers_no_rot, node_dists_no_rot = get_results(os.path.join(results_recording_path, "results_no_rot"))
# print_stat(chamfers_no_rot, category="chamfer")
# print_stat(node_dists_no_rot, category="node_dist")

# df =  pd.DataFrame()
# df["Chamfer"] = list(chamfers) + list(chamfers_no_rot)
# length = len(list(chamfers))
# length_no_rot = len(list(chamfers_no_rot))
# df["Method"] = ["with orientation"]*length + ["no orientation"]*length_no_rot

# ax=sns.boxplot(y="Chamfer",x='Method', data=df, showfliers = False)
# plt.title("Chamfer Distance Comparison")
# plt.show()


# df["Node distance"] = list(node_dists) + list(node_dists_no_rot)
# ax=sns.boxplot(y="Node distance",x='Method', data=df, showfliers = False)
# plt.title("Node Distance Comparison")
# plt.show()

chamfer_diff = chamfers - chamfers_no_rot
print_stat(chamfer_diff, category="chamfer")
min_idx = np.argmin(chamfer_diff)
print(min_idx, chamfer_diff[min_idx])





