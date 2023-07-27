import numpy as np
import pickle
import os
from copy import deepcopy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_mean_std(result_dir, prim_name, obj_type, inside, range_data, method_type='w_rot'):
    chamfer_data = []
    for i in range(range_data[0], range_data[1]):
        if inside:
            file_name = os.path.join(result_dir, obj_type, f"inside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        else:
            file_name = os.path.join(result_dir, obj_type, f"outside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
            # chamfer_data.extend(data)
            # chamfer_data.extend([d for d in data if d <= 1])
            # chamfer_data.extend([d if d < 900 else d-999 for d in data])
            # chamfer_data.extend([d for d in data if d < 900])
    # print(len([d for d in chamfer_data if d >900]))
    mean = np.mean(chamfer_data)
    std = np.std(chamfer_data)
    return mean, std, len(chamfer_data)

def get_results(result_dir, prim_name, obj_type, inside, range_data, method_type='w_rot'):
    chamfer_data = []
    for i in range(range_data[0], range_data[1]):
        if inside:
            file_name = os.path.join(result_dir, obj_type, f"inside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        else:
            file_name = os.path.join(result_dir, obj_type, f"outside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
            chamfer_data.extend(data)
            # chamfer_data.extend([d if d <= 1 else 999 for d in data])
            # chamfer_data.extend([d if d < 900 else d-999 for d in data])

    return np.array(chamfer_data)


result_dir = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/evaluate/chamfer_results"
# prim_name = "cylinder"
# obj_type = "box_1k"
prim_name = "box"
# obj_type = "box_5k"

for obj_type in ['1k']:
    print("==========")
    print(f"{prim_name}_{obj_type}")
    # mean, std, data_length = get_mean_std(result_dir, prim_name, f"{prim_name}_{obj_type}", \
    #                                     inside=True, range_data=[0,10], method_type='w_rot')
    # print(mean, std, data_length)

    chamf_w_rot_ori = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                            inside=True, range_data=[0,10], method_type='w_rot' )
    
    chamf_no_rot_ori = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                            inside=True, range_data=[0,10], method_type='no_rot' )

    chamf_w_rot_ori = np.array(chamf_w_rot_ori) 
    chamf_no_rot_ori = np.array(chamf_no_rot_ori)

    maintain_contact_idxs = np.array(list(set(np.where(chamf_w_rot_ori < 999)[0]) & \
                                      set(np.where(chamf_no_rot_ori < 999)[0])))
    chamf_w_rot = deepcopy(chamf_w_rot_ori[maintain_contact_idxs])
    chamf_no_rot = deepcopy(chamf_no_rot_ori[maintain_contact_idxs])

    mean = np.mean(chamf_w_rot)
    std = np.std(chamf_w_rot)
    max = np.max(chamf_w_rot)
    print(mean, std, max)
    mean = np.mean(chamf_no_rot)
    std = np.std(chamf_no_rot)
    max = np.max(chamf_no_rot)
    print(mean, std, max)  


    diffs = chamf_w_rot - chamf_no_rot
    test1 = [diff for diff in diffs if diff > 0 ]
    print(f"{len(test1)}/{diffs.shape[0]}")
    # print(test1)

    # largest_idx = np.argmin(chamf_w_rot)
    # largest_idx = np.argsort(deepcopy(chamf_w_rot-chamf_no_rot))[-1]
    largest_idx = np.argsort(deepcopy(chamf_w_rot-chamf_no_rot))[(chamf_w_rot.shape[0]*2)//4]
    # # largest_idx = np.argsort(deepcopy(chamf_w_rot-chamf_no_rot))[0]
    # # print(chamf_w_rot[largest_idx]-chamf_no_rot[largest_idx])
    # print(largest_idx)
    # print(chamf_no_rot[largest_idx])
    # print(chamf_w_rot[largest_idx])
    largest_idx = maintain_contact_idxs[largest_idx]
    print(largest_idx)
    print(chamf_no_rot_ori[largest_idx])
    print(chamf_w_rot_ori[largest_idx])

