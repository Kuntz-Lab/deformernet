import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_results(result_dir, prim_name, obj_type, inside, range_data, mp_type):
    chamfer_data = []
    for i in range(range_data[0], range_data[1]):
        if mp_type=="seg_pointconv":
            file_name = os.path.join(result_dir, obj_type, f"inside", f"{prim_name}_{str(i)}.pickle")
        else:
            if inside:
                file_name = os.path.join(result_dir, obj_type, f"inside_{mp_type}", f"{prim_name}_{str(i)}.pickle")
            else:
                file_name = os.path.join(result_dir, obj_type, f"outside_{mp_type}", f"{prim_name}_{str(i)}.pickle")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
            chamfer_data.extend(data)
            # chamfer_data.extend([d if d <= 1 else 999 for d in data])
            # chamfer_data.extend([d if d < 999 else -999 for d in data])
            # chamfer_data.extend([d if d < 999 else d-999 for d in data])
            # chamfer_data.extend([d for d in data if d < 999])
    # mean = np.mean(chamfer_data)
    # std = np.std(chamfer_data)
    return np.array(chamfer_data)

def compute_and_display_statistics(chamfer_results):
    for res in chamfer_results:
        print(np.mean(res))

result_dir = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/evaluate_success/chamfer_results"
# prim_name = "cylinder"
# obj_type = "cylinder_1k"
prim_name = "box"
# obj_type = "box_5k"

# for obj_type in ['1k', '5k', '10k']:
result_range = [0,10]
for obj_type in ['1k']:
    print("==========")
    print(f"{prim_name}_{obj_type}")
    chamfer_gt = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=result_range, mp_type="gt")
    chamfer_seg = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=result_range, mp_type="seg")
    chamfer_classifier = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=result_range, mp_type="classifier")
    chamfer_keypoint = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=result_range, mp_type="keypoint")
    chamfer_seg_pointconv = get_results(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=result_range, mp_type="seg_pointconv")


# largest_idx = np.argsort(chamfer_keypoint)[-5:]
# print(largest_idx)
# print(chamfer_keypoint[largest_idx])

# assert chamfer_gt != [] and chamfer_seg != [] and chamfer_classifier != [] 

# lost_contact_gt = np.where(chamfer_gt < 999)[0]
# lost_contact_seg = np.where(chamfer_seg < 999)[0]
# lost_contact_classifier = np.where(chamfer_classifier < 999)[0]
# lost_contact_keypoint = np.where(chamfer_keypoint < 999)[0]
# lost_contact_seg_pointconv = np.where(chamfer_seg_pointconv < 999)[0]
lost_contact_gt = np.where(chamfer_gt < 999)[0]
lost_contact_seg = np.where(chamfer_seg < 999)[0]
lost_contact_classifier = np.where(chamfer_classifier < 999)[0]
lost_contact_keypoint = np.where(chamfer_keypoint < 999)[0]
lost_contact_seg_pointconv = np.where(chamfer_seg_pointconv < 0.6)[0]
# print(lost_contact_gt, lost_contact_seg, lost_contact_classifier)
# valid_idxs = np.array(list(set(lost_contact_gt) & set(lost_contact_seg) & set(lost_contact_classifier)))
# valid_idxs = np.array(list(set(lost_contact_gt) & set(lost_contact_seg)\
#                  & set(lost_contact_classifier) & set(lost_contact_keypoint)))
# valid_idxs = np.array(list( \
#                  set(lost_contact_classifier) & set(lost_contact_keypoint)))
valid_idxs = np.array(list(set(lost_contact_gt) & set(lost_contact_seg)\
                 & set(lost_contact_classifier) & set(lost_contact_keypoint) & set(lost_contact_seg_pointconv)))
print(valid_idxs.shape)
chamfer_gt = chamfer_gt[valid_idxs]
chamfer_seg = chamfer_seg[valid_idxs]
chamfer_classifier = chamfer_classifier[valid_idxs]
chamfer_keypoint = chamfer_keypoint[valid_idxs]
chamfer_seg_pointconv = chamfer_seg_pointconv[valid_idxs]
compute_and_display_statistics([chamfer_gt, chamfer_seg, chamfer_seg_pointconv, chamfer_classifier, chamfer_keypoint])
    

# # plot = plt.figure(1)
# # plt.plot(chamfer_gt, 'ro')
# # plt.plot(chamfer_seg, 'bo')
# # plt.legend(['gt', 'seg'])
# # plt.title('Final chamfer distance -- ground truth MP vs predicted MP from segmentation method')

# # plot = plt.figure(2)
# # plt.plot(chamfer_gt, 'ro')
# # plt.plot(chamfer_classifier, 'go')
# # plt.legend(['gt', 'classifier'])
# # plt.title('Final chamfer distance -- ground truth MP vs predicted MP from classification method')

# # plot = plt.figure(3)
# # plt.plot(chamfer_seg, 'bo')
# # plt.plot(chamfer_classifier, 'go')
# # plt.legend(['seg', 'classifier'])
# # plt.title('Final chamfer distance -- segmentation vs classifier method')

# # plot = plt.figure(7)
# # plt.plot(chamfer_seg, 'bo')
# # plt.plot(chamfer_keypoint, 'go')
# # plt.legend(['seg', "kp"])
# # plt.title('Final chamfer distance -- segmentation vs keypoint method')

# plot = plt.figure(8)
# zero_line_x = [0,60]
# zero_line_y = [0,0]
# plt.plot(np.array(chamfer_seg)-np.array(chamfer_keypoint), 'ko')
# plt.plot(zero_line_x, zero_line_y, 'r-', linewidth=3)
# plt.title('chamfer dense predictor - chamfer keypoint')

# # plot = plt.figure(9)
# # zero_line_x = [0,60]
# # zero_line_y = [0,0]
# # plt.plot(np.array(chamfer_classifier)-np.array(chamfer_keypoint), 'ko')
# # plt.plot(zero_line_x, zero_line_y, 'r-', linewidth=3)
# # plt.title('chamfer classifier - chamfer keypoint')

# # plot = plt.figure(4)
# # zero_line_x = [0,80]
# # zero_line_y = [0,0]
# # plt.plot(np.array(chamfer_gt)-np.array(chamfer_seg), 'ko')
# # plt.plot(zero_line_x, zero_line_y, 'r-', linewidth=3)
# # plt.title('chamfer ground truth - chamfer seg')


# # plot = plt.figure(5)
# # plt.plot(np.array(chamfer_gt)-np.array(chamfer_classifier), 'ko')
# # plt.plot(zero_line_x, zero_line_y, 'r-', linewidth=3)
# # plt.title('chamfer ground truth - chamfer classifier')

# # plot = plt.figure(6)
# # zero_line_x = [0,80]
# # zero_line_y = [0,0]
# # plt.plot(np.array(chamfer_seg)-np.array(chamfer_classifier), 'ko')
# # plt.plot(zero_line_x, zero_line_y, 'r-', linewidth=3)
# # plt.title('chamfer seg - chamfer classifier')

# plt.xlabel('goal pc sample index')
# plt.ylabel('Chamfer distance (m)')
# plt.show()


chamfer_gt_tolerance = np.array(chamfer_gt)*1.2
# success_seg = np.where(chamfer_gt_tolerance >= np.array(chamfer_seg))[0]
success_seg = list(set(np.where(np.array(chamfer_seg) <= 1.2*np.array(chamfer_gt))[0]) \
                    | set(np.where(np.array(chamfer_seg) <= 0.20)[0]))
success_seg_count = sum([1 if succ > 0 else 0 for succ in success_seg])
print(success_seg_count)

# success_classifier = np.where(chamfer_gt_tolerance >= np.array(chamfer_classifier))[0]
success_classifier = list(set(np.where(np.array(chamfer_classifier) <= 1.2*np.array(chamfer_gt))[0]) \
                    | set(np.where(np.array(chamfer_classifier) <= 0.20)[0]))
success_classifier_count = sum([1 if succ > 0 else 0 for succ in success_classifier])
print(success_classifier_count)

df =  pd.DataFrame()
df["chamfer"] = list(chamfer_gt) + list(chamfer_seg) + list(chamfer_classifier) + list(chamfer_keypoint) + list(chamfer_seg_pointconv) 
length = len(list(chamfer_gt))
df["method"] = ["ground truth"]*length + ["dense predictor PointNet++"]*length + ["classifier"]*length + ["keypoint"]*length + ["dense predictor (PointConv)"]*length

ax=sns.boxplot(y="chamfer",x='method', data=df, whis=10, showfliers = True)


plt.title('Evaluate DeformerNet using ground truth MP vs predicted MP with multiple methods', fontsize=16)
plt.xlabel('Method type',fontsize=16)
plt.ylabel('Chamfer Distance (m)', fontsize=16)
plt.show()
