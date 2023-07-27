
import sys
import os

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True

# Cylinders:
for i in range(0, 100):
    os.system("source devel/setup.bash")

    if headless:
        os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation.py --flex --headless True --obj_name cylinder_" + str(i))
    else:
        os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation.py --flex --obj_name cylinder_" + str(i))





