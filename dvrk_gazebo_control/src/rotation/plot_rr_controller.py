import matplotlib.pyplot as plt
import numpy as np
import pickle

# x = np.random.rand(100)
# ys = np.random.rand(2,100)
# for y in ys:
#   plt.plot(x, y, 'bo-', label='DeformerNet')
# plt.show()

with open('debug/combine.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = np.array(data).T
plot1 = plt.figure(1)
sub_data = data[:6,:]
x = np.arange(sub_data.shape[1])
for y in sub_data:
  plt.plot(x, y)
  plt.legend(['x', 'y', 'z'])
# plt.show()

plot2 = plt.figure(2)
sub_data = data[6:12,:]
x = np.arange(sub_data.shape[1])
for y in sub_data:
  plt.plot(x, y)
  plt.legend(['alpha', 'beta', 'gamma'])
# plt.show()

plot3 = plt.figure(3)
sub_data = data[12:18,:]
x = np.arange(sub_data.shape[1])
for y in sub_data:
  plt.plot(x, y)
  plt.legend(['x', 'y', 'z', 'alpha', 'beta', 'gamma'])
# plt.show()

plot3 = plt.figure(4)
sub_data = data[18:23,:]
x = np.arange(sub_data.shape[1])
for y in sub_data:
  plt.plot(x, y)
  plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
# plt.show()

plot3 = plt.figure(5)
sub_data = data[23:,:]
x = np.arange(sub_data.shape[1])
for y in sub_data:
  plt.plot(x, y)
  plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
plt.show()


