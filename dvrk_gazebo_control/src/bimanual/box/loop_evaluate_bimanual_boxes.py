#!/usr/bin/env python3
import sys
import os
import timeit
import numpy as np

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True  #False
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     os.system(f"rosrun dvrk_gazebo_control bimanual_multi_box_collect_goals.py --flex --headless {str(headless)}")
#     # os.system(f"rosrun dvrk_gazebo_control bimanual_single_box_evaluate_goals.py --flex --headless {str(headless)}")

for _ in range(0, 100):

    # i = np.random.randint(0,10)
    # os.system(f"rosrun dvrk_gazebo_control bimanual_multi_boxes_collect_goals.py --flex --headless {str(headless)} --obj_name box_{i}")
    
    os.system(f"rosrun dvrk_gazebo_control bimanual_multi_boxes_evaluate_goals.py --flex --headless {str(headless)}")





