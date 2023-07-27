from matplotlib import pyplot as plt
import numpy as np
import pickle
import os


path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/results"

final_errors = []
final_frechet_1 = []
final_frechet_2 = []

start_errors = []
start_frechet_1 = []
start_frechet_2 = []

for i in range(100):
    with open(os.path.join(path, f"sample {i}.pickle"), 'rb') as handle:
        data = pickle.load(handle)

    goal_bb_1 = data["goal_bb_1"]
    goal_bb_2 = data["goal_bb_2"]
    backbone_data_1 = data["backbone_data_1"]
    backbone_data_2 = data["backbone_data_2"]
    frechets_1 = data["frechets_1"]
    frechets_2 = data["frechets_2"]
    end_pos_1 = np.array(data["end_pos_1"])
    end_pos_2 = np.array(data["end_pos_2"])

    err_x = end_pos_2[:,0] - end_pos_1[:,0]
    err_y = end_pos_2[:,1] - end_pos_1[:,1]
    err_z = end_pos_2[:,2] - end_pos_1[:,2]
    err = np.sqrt(err_x**2 + err_y**2 + err_z**2)
    min_idx = np.argmin(err)    #-1 #np.argmin(err)
    
    # print(end_pos_1.shape, err.shape)
    
    # final_errors.append(min(err))
    final_errors.append(err[min_idx])
    final_frechet_1.append(frechets_1[min_idx])
    final_frechet_2.append(frechets_2[min_idx])

    # print("results:", err, frechets_1, frechets_2)    
    # print("final end pos error:", min(err))   
    # print("final frechet dists:", frechets_1[min_idx], frechets_2[min_idx])

    start_errors.append(err[0])
    start_frechet_1.append(frechets_1[0])
    start_frechet_2.append(frechets_2[0])



# print(np.sort(final_frechet_2)[-10:])
# print(np.argsort(final_frechet_2)[-10:])
# print(len(list(np.where(np.array(final_frechet_1)<=0.12)[0])))
bad_idxs = set.union(set([np.argmax(final_errors)]), \
        set(list(np.where(np.array(final_frechet_1)>=0.12)[0])),set(list(np.where(np.array(final_frechet_2)>=0.12)[0])))
bad_idxs = list(bad_idxs)

print("len(bad_idxs):", len(bad_idxs))

for bad_idx in bad_idxs:
    final_errors = [final_errors[i] for i in range(len(final_errors)) if i not in bad_idxs]
    final_frechet_1 = [final_frechet_1[i] for i in range(len(final_frechet_1)) if i not in bad_idxs]
    final_frechet_2 = [final_frechet_2[i] for i in range(len(final_frechet_2)) if i not in bad_idxs]
    # final_errors.pop(bad_idx)
    # final_frechet_1.pop(bad_idx)
    # final_frechet_2.pop(bad_idx)
final_frechet = [final_frechet_1[i]+final_frechet_2[i] for i in range(len(final_frechet_1))]



print("====== FINAL RESULT ======")
# # print("Total data point:", len(final_errors), len(final_frechet_1), len(final_frechet_2))
print("End pos error min, avg, max:", min(final_errors), np.mean(final_errors), max(final_errors), np.std(final_errors))
print("Frechet 1 min, avg, max:", min(final_frechet_1), np.mean(final_frechet_1), max(final_frechet_1), np.std(final_frechet_1))
print("Frechet 2 min, avg, max:", min(final_frechet_2), np.mean(final_frechet_2), max(final_frechet_2), np.std(final_frechet_2))
print("Total Frechet min, avg, max:", min(final_frechet), np.mean(final_frechet), max(final_frechet), np.std(final_frechet))

print("====== START ======")
print("End pos error min, avg, max:", min(start_errors), np.mean(start_errors), max(start_errors))
print("Frechet 1 min, avg, max:", min(start_frechet_1), np.mean(start_frechet_1), max(start_frechet_1))
print("Frechet 2 min, avg, max:", min(start_frechet_2), np.mean(start_frechet_2), max(start_frechet_2))