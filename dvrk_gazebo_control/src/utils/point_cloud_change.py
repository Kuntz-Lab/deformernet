from matplotlib import pyplot as plt
import pickle
import numpy as np
import open3d


with open('/home/baothach/shape_servo_data/point_cloud_change/change_1', 'rb') as handle:
    data = pickle.load(handle)
    distances = list(data["change"])
    init_pc = data["init pc"]
    goal_pc = data["final pc"]

N = 10
res = sorted(range(len(distances)), key = lambda sub: distances[sub])[-N:]

sum_dist = np.array([0,0,0])
for idx in res:
    # sum_dist = sum_dist + np.array(init_pc[idx])
    sum_dist = sum_dist + np.array(init_pc[idx])*distances[idx]
    print(init_pc[idx])

# print(sum_dist*(1/N))
print(sum_dist / (sum([distances[idx] for idx in res])))

# z_value = [x[2] for x in goal_pc]
# max_idx = z_value.index(max(z_value))


# print(max_idx)
# print(init_pc[110])
# print(distances[1520])
# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(np.array(init_pc)) 
# pcd_goal = open3d.geometry.PointCloud()
# pcd_goal.points = open3d.utility.Vector3dVector(np.array(goal_pc)) 
# open3d.visualization.draw_geometries([pcd, pcd_goal]) 
# distances = []   
# for i in range(1, len(points)):
#     distances.append(np.linalg.norm(points[i]- points[0]))

# print(points["points"])
time1 = range(len(distances))


# plt.scatter(time1, distances)
# plt.title('Distance between current particles and initial particles')
# plt.xlabel('Record every 100 frames')
# plt.ylabel('Distance (m)')
# plt.show()

