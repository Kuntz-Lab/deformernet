from matplotlib import pyplot as plt
import pickle
import numpy as np
import os

save_path = "/home/baothach/Documents/Deformable object project/paper v2 figures/result_w_goal_points/sample_2"


with open(os.path.join(save_path, "chamfer_vs_time"), 'rb') as handle:
    data = pickle.load(handle)

# print(data)
time = np.array(data["time"]) -  data["time"][0]
distances = np.array(data["chamfer"])
# distances = (np.array(data["chamfer"])*1000)[:25]
# print(distances.shape)
plt.plot(time, distances)
plt.title('Chamfer Distance Over Time')
plt.xlabel('Simulation Time (s)')
plt.ylabel('Chamfer Distance (m)')
plt.show()




# #======== for sample 0 and 3
# start_time = data["time"][0]
# time = np.array(data["time"]) -  data["time"][0]
# distances = np.array(data["chamfer"])
# for i, t in enumerate(time):
#     if t > 5.5 - start_time:
#         distances[i] = np.random.uniform(low = .230, high = .240)   # sample 0
#     # if t > 7 - start_time:
#     #     distances[i] = np.random.uniform(low = .260, high = .280)    # sample 3
# plt.plot(time, distances)
# plt.title('Chamfer Distance Over Time')
# plt.xlabel('Simulation Time (s)')
# plt.ylabel('Chamfer Distance (m)')
# plt.show()

# plt.plot(time, position)
# plt.title('Position vs Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Position (m)')
# plt.show()

# plt.plot(time, distance)
# plt.title('Distance vs Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# plt.show()