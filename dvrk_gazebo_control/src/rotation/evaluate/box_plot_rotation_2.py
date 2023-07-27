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
    for i in range_data:
        if inside:
            file_name = os.path.join(result_dir, obj_type, f"inside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        else:
            file_name = os.path.join(result_dir, obj_type, f"outside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                # data = pickle.load(handle)
                # data = pickle.load(handle)["chamfer"]
            #     data = pickle.load(handle)["node"]
                
          
            # chamfer_data.extend(data)
            # chamfer_data.extend([d if d <= 1 else 999 for d in data])
            # chamfer_data.extend([d if d < 900 else d-999 for d in data])
            
                data = pickle.load(handle)
            chamfer_data.extend(list(np.array(data["node"])/data["num_nodes"]*10000))

    return np.array(chamfer_data)


result_dir = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/evaluate/chamfer_results"
# prim_name = "cylinder"
# obj_type = "box_1k"
prim_name = "box"
# obj_type = "box_5k"

object_meshes_path = "/home/baothach/shape_servo_data/evaluation/meshes/box_1k/inside_sample_ratio"
for t in range(1):
    print("======================================", t)
    object_name = "box_" + str(t)
    with open(os.path.join(object_meshes_path, object_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
    print(f'height: {data["height"]}, width: {data["width"]}, ratio:{data["height"]/data["width"]}')

    chamfers = []
    model_type = []
    categories = []

    last_idx = 10

    for obj_type in ['1k']:
        # for method_type in ['', 'only_rot_', 'only_pos_']:
        # for method_type in ['normal_']:        
        # for method_type in ['sample_ratio_']:    
        for method_type in ['gt_']:     
            print("==========")
            print(f"{method_type}")

            if method_type == "difficult_":
                last_idx = 50
            elif method_type == "normal_":
                last_idx = 100

            exclude_list = []
            # print([k for k in range(4) if k not in exclude_list])
            chamf_w_rot_ori = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}w_rot' )
            
            chamf_no_rot_ori = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}no_rot' )

            # chamf_w_rot_ori = np.delete(chamf_w_rot_ori, [1,9,4])
            # chamf_no_rot_ori = np.delete(chamf_no_rot_ori, [1,9,4])

            # print(chamf_w_rot_ori, chamf_no_rot_ori)

            chamf_w_rot_ori = np.array(chamf_w_rot_ori) 
            chamf_no_rot_ori = np.array(chamf_no_rot_ori)
            
            print(chamf_w_rot_ori.shape, chamf_no_rot_ori.shape)
            print(chamf_w_rot_ori.mean(), chamf_no_rot_ori.mean())

            chamf_w_rot = chamf_w_rot_ori#[:10]
            chamf_no_rot = chamf_no_rot_ori#[:10]


            # maintain_contact_idxs = np.array(list(set(np.where(chamf_w_rot_ori < 999)[0]) & \
            #                                 set(np.where(chamf_no_rot_ori < 999)[0])))


            # maintain_contact_idxs = np.array(list(set(np.where(chamf_w_rot_ori < 0.4)[0]) & \
            #                                 set(np.where(chamf_no_rot_ori < 999)[0])))

            # maintain_contact_idxs = np.array(np.where(chamf_w_rot_ori < 0.8)[0])
                                            

            # chamf_w_rot = deepcopy(chamf_w_rot_ori[maintain_contact_idxs])
            # chamf_no_rot = deepcopy(chamf_no_rot_ori[maintain_contact_idxs])
            print(chamf_w_rot.mean(), chamf_no_rot.mean())




            chamfers += list(chamf_w_rot) + list(chamf_no_rot)
            model_type += ["with orientation"]*len(chamf_w_rot) + ["without orientation"]*len(chamf_no_rot)
            if method_type == '':
                categories += ["both orientation and position"]*(len(chamf_w_rot)+len(chamf_no_rot))
            elif method_type == 'only_rot_':
                categories += ["only orientation"]*(len(chamf_w_rot)+len(chamf_no_rot))
            elif method_type == 'only_pos_':
                categories += ["only position"]*(len(chamf_w_rot)+len(chamf_no_rot))
            elif method_type == "long_":
                categories += ["long box"]*(len(chamf_w_rot)+len(chamf_no_rot))       
            elif method_type == "normal_":     
                categories += ["random boxes"]*(len(chamf_w_rot)+len(chamf_no_rot))   
            else:     
                categories += ["random boxes"]*(len(chamf_w_rot)+len(chamf_no_rot))

    chamf_diff = chamf_w_rot - chamf_no_rot
    print(np.where(chamf_diff<0))
    # print(chamf_diff)
    count = sum(1 for i in chamf_diff if i < 0)
    print(f"New model outperforms {count}/{chamf_diff.shape[0]} or {count/chamf_diff.shape[0]}%")

# # plt.plot(chamf_w_rot, "go")
# # plt.plot(chamf_no_rot, "ro")
# # plt.show()

df =  pd.DataFrame()
df["chamfer"] = chamfers
# df["obj name"] = object_names
df["model type"] = model_type
df["category"] = categories

ax=sns.boxplot(y="chamfer",x='category', hue='model type', data=df, whis=1000, showfliers = True) #, whis=3.5


# # plt.title('New model (with orientation) vs old model (w/o) using Node Dist', fontsize=16)
plt.title('Long boxes', fontsize=16)
plt.xlabel('Category',fontsize=16)
plt.ylabel('Node Distance (m)', fontsize=16)

plt.ylim([0,25])

plt.show()


