#!/usr/bin/env python3
import sys
import os
import numpy as np

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True

## Cylinders:
# # DeformerNet
# for i in range(0, 100):
# # for i in range(96, -1, -1):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation_2.py --flex --headless True --obj_name cylinder_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation_2.py --flex --obj_name cylinder_" + str(i))

# # MP
for _ in range(0, 500):
    i = np.random.randint(low=0, high=100)

    if headless:
        os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation_2.py --flex --headless True --obj_name cylinder_" + str(i))
    else:
        os.system("rosrun dvrk_gazebo_control cylinder_collect_data_w_rotation_2.py --flex --obj_name cylinder_" + str(i))



# ## boxes:
# # DeformerNet
# for i in range(0, 100):
# # for i in range(96, -1, -1):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control box_collect_data_w_rotation_2.py --flex --headless True --obj_name box_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control box_collect_data_w_rotation_2.py --flex --obj_name box_" + str(i))

# # MP
# for _ in range(0, 350):
#     i = np.random.randint(low=0, high=100)

#     if headless:
#         os.system(f"rosrun dvrk_gazebo_control box_collect_data_w_rotation.py --flex --headless True --obj_name box_{i} --data_category MP")
#     else:
#         os.system(f"rosrun dvrk_gazebo_control box_collect_data_w_rotation.py --flex --obj_name box_{i} --data_category MP")      


##### Hemis
# # DeformerNet
# for i in range(0, 100):
# # for i in range(96, -1, -1):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_w_rotation.py --flex --headless True --obj_name hemis_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_w_rotation.py --flex --obj_name hemis_" + str(i))

# # MP
# for _ in range(0, 350):
#     i = np.random.randint(low=0, high=100)

#     if headless:
#         os.system(f"rosrun dvrk_gazebo_control hemis_collect_data_w_rotation.py --flex --headless True --obj_name hemis_{i} --data_category MP")
#     else:
#         os.system(f"rosrun dvrk_gazebo_control hemis_collect_data_w_rotation.py --flex --obj_name hemis_{i} --data_category MP")      