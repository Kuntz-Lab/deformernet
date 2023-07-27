import numpy as np
import pickle5 as pickle
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
    chamfer_data_avg = []
    for i in range_data:
        if inside:
            file_name = os.path.join(result_dir, obj_type, f"inside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        else:
            file_name = os.path.join(result_dir, obj_type, f"outside_{method_type}", f"{prim_name}_{str(i)}.pickle")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                # data = pickle.load(handle)
                # data = pickle.load(handle)["chamfer"]
                data_avg = pickle.load(handle)
                data = data_avg["node"]
                
                
          
            chamfer_data.extend(data)
            # chamfer_data.extend([d if d <= 1 else 999 for d in data])
            # chamfer_data.extend([d if d < 900 else d-999 for d in data])
            
            #     
            
            # chamfer_data_avg.extend(data_avg["node"])
            # chamfer_data_avg.extend(data_avg["chamfer"])
            chamfer_data_avg.extend(list(np.array(data_avg["node"])/data_avg["num_nodes"]*10000))

    return np.array(chamfer_data), np.array(chamfer_data_avg) 


result_dir = "/home/baothach/shape_servo_data/rotation_extension/multi_hemis_1000Pa_2/evaluate/chamfer_results"
# prim_name = "cylinder"
# obj_type = "hemis_1k"
prim_name = "hemis"
# obj_type = "hemis_5k"

object_meshes_path = "/home/baothach/shape_servo_data/evaluation/meshes/hemis_1k/inside_sample_ratio"
for t in range(1):
    # print("======================================", t)
    # object_name = "hemis_" + str(t)
    # with open(os.path.join(object_meshes_path, object_name + ".pickle"), 'rb') as handle:
    #     data = pickle.load(handle)
    # print(f'height: {data["height"]}, width: {data["width"]}, ratio:{data["height"]/data["width"]}')

    chamfers = []
    model_type = []
    categories = []

    last_idx = 10

    for obj_type in ['1k']:
        # for method_type in ['', 'only_rot_', 'only_pos_']:
        # for method_type in ['normal_']:        
        for method_type in ['']:    
            print("==========")
            print(f"{method_type}")

            if method_type == "difficult_":
                last_idx = 50
            elif method_type == "normal_":
                last_idx = 100

            exclude_list = []
            # print([k for k in range(4) if k not in exclude_list])
            chamf_w_rot_ori, chamf_w_rot_ori_avg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}gt_w_rot' )
            
            chamf_no_rot_ori, chamf_no_rot_ori_avg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}gt_no_rot' )


            a, a_avg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}dense_w_MP_w_rot' )

            b, b_avg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}dense_no_MP_no_rot' )

            c, c_avg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}dense_no_MP_w_rot' )

            d, d_avg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}dense_w_MP_no_rot' )

            e, e_avg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", \
                                    inside=True, range_data=[k for k in range(10) if k not in exclude_list], method_type=f'{method_type}classifier_w_rot' )

            num_print = 10
            max_dists = np.sort(a)
            max_idxs = np.argsort(a)
            print("max_dists:", max_dists[-num_print:])
            print("max_idxs:", max_idxs[-num_print:])

            # chamf_w_rot_ori = np.delete(chamf_w_rot_ori, [1,9,4])
            # chamf_no_rot_ori = np.delete(chamf_no_rot_ori, [1,9,4])

            # print(chamf_w_rot_ori, chamf_no_rot_ori)

            # chamf_w_rot_ori = np.array(chamf_w_rot_ori) 
            # chamf_no_rot_ori = np.array(chamf_no_rot_ori)
            
            print(chamf_w_rot_ori.shape, chamf_no_rot_ori.shape, a.shape, b.shape, c.shape, d.shape, e.shape)
            print(chamf_w_rot_ori.mean(), chamf_no_rot_ori.mean(), a.mean(), b.mean(), c.mean(), d.mean(), e.mean())

            # chamf_w_rot = chamf_w_rot_ori#[:10]
            # chamf_no_rot = chamf_no_rot_ori#[:10]


            # maintain_contact_idxs = np.array(list(set(np.where(chamf_w_rot_ori < 999)[0]) & \
            #                                 set(np.where(chamf_no_rot_ori < 999)[0])))


            maintain_contact_idxs = np.array(list(set(np.where(chamf_w_rot_ori < 999)[0]) & \
                                            set(np.where(chamf_no_rot_ori < 999)[0]) & set(np.where(a < 999)[0]) & set(np.where(b < 999)[0]) & set(np.where(c < 999)[0]) & set(np.where(d < 999)[0]) & set(np.where(e < 999)[0])))

            # maintain_contact_idxs = np.array(np.where(chamf_w_rot_ori < 1)[0])
            print(maintain_contact_idxs.shape)                            

            chamf_w_rot = deepcopy(chamf_w_rot_ori_avg[maintain_contact_idxs])
            chamf_no_rot = deepcopy(chamf_no_rot_ori_avg[maintain_contact_idxs])
            a = deepcopy(a_avg[maintain_contact_idxs])
            b = deepcopy(b_avg[maintain_contact_idxs])
            c = deepcopy(c_avg[maintain_contact_idxs])
            d = deepcopy(d_avg[maintain_contact_idxs])
            e = deepcopy(e_avg[maintain_contact_idxs])
            # print(chamf_w_rot.mean(), chamf_no_rot.mean())
            print(chamf_w_rot.mean(), chamf_no_rot.mean(), a.mean(), b.mean(), c.mean(), d.mean(), e.mean())




            chamfers += list(chamf_w_rot) + list(chamf_no_rot) + list(a) + list(b) + list(c) + list(d) + list(e)
            model_type += ["with ori with MP oracle"]*len(chamf_w_rot) + ["no ori no MP oracle"]*len(chamf_no_rot) + ["with ori with MP dense"]*len(a) \
                        + ["no ori no MP dense"]*len(b) + ["with ori no MP dense"]*len(c) + ["no ori with MP dense"]*len(d) + ["classifier"]*len(e)
            if method_type == 'a':
                categories += ["both orientation and position"]*(len(chamf_w_rot)+len(chamf_no_rot))
            elif method_type == 'only_rot_':
                categories += ["only orientation"]*(len(chamf_w_rot)+len(chamf_no_rot))
            elif method_type == 'only_pos_':
                categories += ["only position"]*(len(chamf_w_rot)+len(chamf_no_rot))
            elif method_type == "long_":
                categories += ["long hemis"]*(len(chamf_w_rot)+len(chamf_no_rot))       
            elif method_type == "normal_":     
                categories += ["random hemis"]*(len(chamf_w_rot)+len(chamf_no_rot))   
            else:     
                categories += ["random hemis"]*(len(chamf_w_rot)+len(chamf_no_rot)+len(a)+len(b)+len(c)+len(d)+len(e))

    # chamf_diff = chamf_w_rot - chamf_no_rot
    # print(np.where(chamf_diff<0))
    # print(chamf_diff)
    # count = sum(1 for i in chamf_diff if i < 0)
    # print(f"New model outperforms {count}/{chamf_diff.shape[0]} or {count/chamf_diff.shape[0]}%")


# # plt.plot(chamf_w_rot, "go")
# # plt.plot(chamf_no_rot, "ro")
# # plt.show()

df =  pd.DataFrame()
df["chamfer"] = chamfers
# df["obj name"] = object_names
df["model type"] = model_type
df["category"] = categories

ax=sns.boxplot(y="chamfer",x='category', hue='model type', data=df, whis=3.5, showfliers = True) #, whis=3.5


# # plt.title('New model (with orientation) vs old model (w/o) using Node Dist', fontsize=16)
plt.title('Long hemis', fontsize=16)
plt.xlabel('Category',fontsize=16)
plt.ylabel('Node Distance (m)', fontsize=16)

plt.ylim([0,25])

plt.show()


