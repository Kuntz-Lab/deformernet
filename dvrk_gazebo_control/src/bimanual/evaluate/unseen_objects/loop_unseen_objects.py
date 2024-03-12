
import sys
import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True
start_time = timeit.default_timer()
obj_name = "chicken_breast"

# # Unseen objects
# for _ in range(200):
    
#     os.system(f"rosrun dvrk_gazebo_control bimanual_unseen_objects_collect_goals.py " \
#             f"--headless {str(headless)} --obj_name {obj_name}")
    

for obj_idx in range(0, 100):

    os.system(f"rosrun dvrk_gazebo_control bimanual_unseen_objects_collect_goals.py " \
            f"--headless {str(headless)} --obj_name {obj_name} --obj_idx {obj_idx}")
    
    os.system(f"rosrun dvrk_gazebo_control evaluate_unseen_objects_bimanual.py " \
            f"--headless {str(headless)} --obj_name {obj_name} --obj_idx {obj_idx} --model_category combined")


print("Elapsed time", (timeit.default_timer() - start_time)/3600)