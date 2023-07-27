from matplotlib import pyplot as plt
import pickle
import numpy as np
import open3d


with open('/home/baothach/shape_servo_data/uncertainty/on_ground_2', 'rb') as handle:
    distances = pickle.load(handle)


# distances = []   
# for i in range(1, len(points)):
#     distances.append(np.linalg.norm(points[i]- points[0]))

# print(points["points"])
time1 = range(len(distances))


plt.plot(time1, distances)
plt.title('Distance between current particles and initial particles')
plt.xlabel('Record every 100 frames')
plt.ylabel('Distance (m)')
plt.show()

# plt.plot(time4, difference_30)
# plt.title('Difference with respect to the initial feature vector')
# plt.show()

# plt.scatter(time2, difference_each_pair)
# plt.title('Difference between each pair')
# plt.show()


# plt.plot(time3, difference_adjacent_pair)
# plt.title('Difference between two adjacent point clouds')
# plt.show()