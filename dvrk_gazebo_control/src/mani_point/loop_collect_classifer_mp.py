#!/usr/bin/env python3
import sys
import os
import numpy as np


pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True

# # Cylinder
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_multi_shapes.py --flex --headless True --obj_name cylinder_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_multi_shapes.py --flex --obj_name cylinder_" + str(i))



# # Boxes:
# for _ in range(0, 350):
#     i = np.random.randint(low=0, high=100)

#     if headless:
#         os.system("rosrun dvrk_gazebo_control multi_boxes_collect_gt_data.py --flex --headless True --obj_name box_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control multi_boxes_collect_gt_data.py --flex --obj_name box_" + str(i))

for _ in range(0, 1000000):
    i = np.random.randint(low=0, high=100)
    os.system("source devel/setup.bash")

    if headless:
        os.system("rosrun dvrk_gazebo_control multi_boxes_collect_classifier_data.py --flex --headless True --obj_name box_" + str(i))
    else:
        os.system("rosrun dvrk_gazebo_control multi_boxes_collect_classifier_data.py --flex --obj_name box_" + str(i))


# # Hemis
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --headless True --obj_name hemis_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --obj_name hemis_" + str(i))