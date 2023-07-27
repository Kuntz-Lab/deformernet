
import sys
import os
import numpy as np

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = False


# for _ in range(0, 100000):
#     i = np.random.randint(low = 0, high = 100)
#     os.system(f"rosrun dvrk_gazebo_control collect_data_stress_prediction_multi_box.py --headless {str(headless)} --obj_name box_{i}")


for i in range(0, 10):
    os.system(f"rosrun dvrk_gazebo_control get_eval_data_stress_prediction.py --headless {str(headless)} --obj_name box_{i}")