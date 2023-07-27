from matplotlib import pyplot as plt
import pickle


with open('/home/baothach/shape_servo_data/visualization/shape_servo_vis', 'rb') as handle:
    vis_data = pickle.load(handle) 


time = vis_data["time"]
velocity = vis_data["velocity"]
position = vis_data["position"]
distance = vis_data["distance"]

plt.plot(time, velocity)
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.show()


plt.plot(time, position)
plt.title('Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.show()

plt.plot(time, distance)
plt.title('Distance vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.show()