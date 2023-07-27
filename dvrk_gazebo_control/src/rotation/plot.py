import matplotlib.pyplot as plt
import numpy as np

# chamfer = np.arange(0.1, 1.1, 0.1)
# success = [20, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# plt.plot(chamfer, success, 'bo-', label='DeformerNet')

# success = [0, 30, 90, 100, 100, 100, 100, 100, 100, 100]
# plt.plot(chamfer, success, 'g^-', label='RRT')

# success = [0, 0, 20, 40, 40, 40, 50, 70, 70, 70]
# plt.plot(chamfer, success, 'rD-', label='model-free RL')

# plt.title('Success Rate vs Goal Region Tolerance')
# plt.xlabel('Goal region tolerance (m)')
# plt.ylabel('Success Rate (%)')
# plt.legend()
# plt.show()

new_no_rot = [0.26, 0.48, 0.45, 0.268, 0.22, 0.26, 0.3, 0.26, 0.18, 0.15]
old_no_rot = [0.08, 0.19, 0.34, 0.17, 0.3, 0.2, 0.5, 0.26, 0.25, 0.14]

new_rot = [0.28, 0.42, 0.21, 0.28, 0.37, 0.44, 0.4, 0.56, 0.17, 0.44]
old_rot = [0.13, 0.41, 0.2, 0.24, 0.35, 0.43, 0.18, 0.61, 0.17, 0.65]

plt.plot(old_no_rot, 'bo', label='old model + simple goals')
plt.plot(new_no_rot, 'y*', label='new model + simple goals', markersize=10)
plt.title('Chamfer distance over 10 samples - simple goals')
# plt.xlabel('Goal region tolerance (m)')
plt.ylabel('Chamfer distance')
plt.legend()
plt.show()

plt.plot(old_rot, 'bo', label='old model + complex goals')
plt.plot(new_rot, 'y*', label='new model + complex goals', markersize=10)
plt.title('Chamfer distance over 10 samples - complex goals')
# plt.xlabel('Goal region tolerance (m)')
plt.ylabel('Chamfer distance')
plt.legend()
plt.show()


