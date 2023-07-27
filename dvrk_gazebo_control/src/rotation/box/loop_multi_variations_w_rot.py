
import sys
import os

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True

## Cylinders:
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation.py --flex --headless True --obj_name cylinder_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation.py --flex --obj_name cylinder_" + str(i))



## boxs:
for i in range(29, 100):
    os.system("source devel/setup.bash")

    if headless:
        os.system("rosrun dvrk_gazebo_control box_collect_data_w_rotation.py --flex --headless True --obj_name box_" + str(i))
    else:
        os.system("rosrun dvrk_gazebo_control box_collect_data_w_rotation.py --flex --obj_name box_" + str(i))

# # Hemis
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --headless True --obj_name hemis_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --obj_name hemis_" + str(i))