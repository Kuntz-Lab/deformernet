import matplotlib.pyplot as plt
import numpy as np

chamfer = np.arange(0.1, 1.1, 0.1)
success = [20, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# plt.plot(chamfer, success, 'bo-', label='DeformerNet')
plt.plot(chamfer, success, color='dodgerblue', marker='o', label='DeformerNet',markersize=10,linewidth=5)

success = [0, 30, 90, 100, 100, 100, 100, 100, 100, 100]
# plt.plot(chamfer, success, 'g^-', label='RRT')
plt.plot(chamfer, success, color='orangered', marker='^', label='RRT',markersize=10,linewidth=5)

success = [0, 0, 20, 40, 40, 40, 50, 70, 70, 70]
# plt.plot(chamfer, success, 'rD-', label='model-free RL')
plt.plot(chamfer, success, 'gD-', label='model-free RL',markersize=10,linewidth=5)

# plt.title('Success Rate vs. Goal Region Tolerance', fontsize=32)
# plt.xlabel('Goal Region Tolerance (m)', fontsize=32)
# plt.ylabel('Success Rate (%)', fontsize=32)
# plt.legend(prop={'size': 24})
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24, rotation=90)

plt.title('Success Rate vs. Goal Region Tolerance', fontsize=40)
plt.xlabel('Goal Region Tolerance (m)', fontsize=40)
plt.ylabel('Success Rate (%)', fontsize=40)
plt.legend(prop={'size': 40})
plt.xticks(fontsize=32)
plt.yticks(fontsize=32, rotation=90)
plt.subplots_adjust(bottom=0.15)
plt.show()
