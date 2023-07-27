import sys
import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True #True
start_time = timeit.default_timer()


for i in range(0, 100):
    # os.system(f"rosrun dvrk_gazebo_control collect_goals_cylinder_rotation.py --flex --headless {str(headless)}")


    os.system(f"rosrun dvrk_gazebo_control evaluate_cylinder_rotation.py --flex --headless {str(headless)}")