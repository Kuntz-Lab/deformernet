from matplotlib import pyplot as plt
import pickle
import numpy as np
import open3d


with open('/home/baothach/shape_servo_data/uncertainty/uncertainty_vis_no_touching', 'rb') as handle:
    vis_uncertainty = pickle.load(handle) 

points = vis_uncertainty["points"]
final_vtc_30 = vis_uncertainty["30"]
final_vtc_135 = vis_uncertainty["135"]
pcds = []
for point in points:
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(point))
    pcds.append(pcd)
difference_points = [np.linalg.norm(np.asarray(pcds[0].compute_point_cloud_distance(pcds[i]))) for i in range(1,len(points))]    

difference_30 = [np.linalg.norm(final_vtc_30[0] - final_vtc_30[i]) for i in range(1,len(final_vtc_30))] 
difference_each_pair = []
for i in range(len(pcds)):
    temp = i
    for j in range(temp+1, len(pcds)):
        diff = np.linalg.norm(np.asarray(pcds[i].compute_point_cloud_distance(pcds[j])))
        difference_each_pair.append(diff)
        if j > 50:
            break
    if i > 50:
        break   

difference_adjacent_pair = []
for i in range(len(pcds)):
    if i == 0:
        continue
    diff = np.linalg.norm(np.asarray(pcds[i].compute_point_cloud_distance(pcds[i-1])))    
    difference_adjacent_pair.append(diff) 
    

time1 = range(len(difference_points))
time2 = range(len(difference_each_pair))
time3 = range(len(difference_adjacent_pair))
time4 = range(len(difference_30))

plt.plot(time1, difference_points)
plt.title('Difference with respect to the initial point cloud')
plt.show()

plt.plot(time4, difference_30)
plt.title('Difference with respect to the initial feature vector')
plt.show()

plt.scatter(time2, difference_each_pair)
plt.title('Difference between each pair')
plt.show()


plt.plot(time3, difference_adjacent_pair)
plt.title('Difference between two adjacent point clouds')
plt.show()