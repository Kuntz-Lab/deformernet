from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/results"
with open(os.path.join(path, f"sample {0}.pickle"), 'rb') as handle:
    data = pickle.load(handle)

goal_bb_1 = data["goal_bb_1"]
goal_bb_2 = data["goal_bb_2"]
backbone_data_1 = data["backbone_data_1"]
backbone_data_2 = data["backbone_data_2"]
frechets_1 = data["frechets_1"]
frechets_2 = data["frechets_2"]
end_pos_1 = np.array(data["end_pos_1"])
end_pos_2 = np.array(data["end_pos_2"])

print(len(end_pos_1))
# for i in range(len(backbone_data_1)):
#     plt.plot(backbone_data_1[i][0], backbone_data_1[i][1], 'g')
#     plt.plot(goal_bb_1[0], goal_bb_1[1], 'ro')    
    
#     plt.plot(backbone_data_2[i][0], backbone_data_2[i][1], 'b')
#     plt.plot(goal_bb_2[0], goal_bb_2[1], 'yo')    
    
# plt.legend(['cylinder 1', 'goal 1', 'cylinder 2', 'goal 2'])
# plt.title("Visualization of 2 cylinders' backbones")
# plt.xlabel('x position (m)')
# plt.ylabel('y position (m)')


# plt.plot(frechets_1)
# plt.plot(frechets_2)
# plt.legend(['cylinder 1', 'cylinder 2'])
# plt.title("Frechet distance cylinders' backbones vs goals' backbones")
# plt.xlabel('Time')
# plt.ylabel('Frechet distance (m)')


# for i in range(len(backbone_data_1)):
#     print(end_pos_1[i][0])
#     plt.plot(end_pos_1[i][0], end_pos_1[i][1])
    

# # print(np.array(end_pos_2))
# plt.plot(end_pos_1[:,0], end_pos_1[:,1])
# plt.plot(end_pos_2[:,0], end_pos_2[:,1])
# # plt.plot(end_pos_1[:-3,0], end_pos_1[:-3,1])
# # plt.plot(end_pos_2[:-3,0], end_pos_2[:-3,1])
# plt.legend(['cylinder 1', 'cylinder 2'])
# plt.title("x-y trajectory of the ends of two cylinders")
# plt.xlabel('x position (m)')
# plt.ylabel('y position (m)')

err_x = end_pos_2[:,0] - end_pos_1[:,0]
err_y = end_pos_2[:,1] - end_pos_1[:,1]
err = np.sqrt(err_x**2 + err_y**2)
plt.plot(err)
plt.ylabel('Errors (m)')
plt.title("Errors vs Time")

plt.show()

# if t == 0:                    
#     plt.plot(xs, ys, 'g')
#     plt.plot(goal_backbones[t][:,0], goal_backbones[t][:,1], 'ro')
#     backbone_data_1.append([xs, ys])
#     frechets_1.append(df)
#     end_pos_1.append(pc_1[backbone_idxs][center_1_idx])
# else:
#     plt.plot(xs, ys, 'g')
#     plt.plot(goal_backbones[t][:,0], goal_backbones[t][:,1], 'bo')
#     backbone_data_2.append([xs, ys])
#     frechets_2.append(df)
#     end_pos_2.append(pc_2[backbone_idxs][center_2_idx])

