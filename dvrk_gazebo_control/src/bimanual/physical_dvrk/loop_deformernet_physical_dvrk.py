
import sys
import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True
start_time = timeit.default_timer()

# prim_names = ["box", "cylinder", "hemis"] #["box", "cylinder", "hemis"]
stiffnesses = ["1k"]  #["1k", "5k", "10k"] 

prim_name = "box"

# # Data collection
# for stiffness in stiffnesses:
#     for _ in range(0, 300):  # 500
#         i = np.random.randint(0,100)
#         os.system(f"rosrun dvrk_gazebo_control collect_data_bimanual_physical_dvrk.py --flex --headless {str(headless)} "
#                 f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {i}")
    
#         # os.system(f"rosrun dvrk_gazebo_control collect_data_single_physical_dvrk.py --flex --headless {str(headless)} "
#         #         f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {i}")

# Evaluation
for stiffness in stiffnesses:
    for i in range(0,100):  # 500
        # os.system(f"rosrun dvrk_gazebo_control collect_goals_physical_dvrk.py --flex --headless {str(headless)} "
        #         f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {i}")
        os.system(f"rosrun dvrk_gazebo_control evaluate_object_frame.py --flex --headless {str(headless)} "
                f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {i}")    

print(f"Elapsed time (hours): {(timeit.default_timer() - start_time)/3600:.3f}")