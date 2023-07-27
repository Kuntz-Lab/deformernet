import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import os


save_path = "/home/baothach/shape_servo_data/stress_field_prediction/test"

with open(os.path.join(save_path, "sample 0.pickle"), 'rb') as handle:
    data = np.array(pickle.load(handle))


plt.figure(figsize=(14, 10), dpi=80)

# xs = np.arange(data.shape[0])
plt.plot(data[:,0], label='Desired Left',linewidth=5)
plt.plot(data[:,1], label='Desired Right',linewidth=5)
plt.plot(data[:,2], label='Measured Left',linewidth=5)
plt.plot(data[:,3], label='Measured Right',linewidth=5)



plt.title('Desired and Measured Force Over Time', fontsize=40)
# plt.xlabel('Force (N)', fontsize=40)
plt.ylabel('Force (N)', fontsize=40)
plt.legend(prop={'size': 36})
plt.xticks(fontsize=32)
plt.yticks(fontsize=32, rotation=90)
# plt.subplots_adjust(bottom=0.15)

# plt.savefig(f'/home/baothach/Downloads/success_rate_baseline_compare.png', bbox_inches='tight', pad_inches=0.05)
plt.show()
