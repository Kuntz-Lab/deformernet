
import sys
import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True
start_time = timeit.default_timer()

prim_names = ["box"] #["box", "cylinder", "hemis"]
# stiffnesses = ["5k"]  #["1k", "5k", "10k"] 
goal_models = ["defgoalnet"] #["diffdef", "defgoalnet"]

stiffness = "5k"
for goal_model in goal_models:
    for _ in range(0, 100):
        i = np.random.randint(0,100)
        for prim_name in prim_names:
            os.system(f"rosrun dvrk_gazebo_control evaluate_object_packaging_multimodal_object_frame.py --headless {str(headless)} "
                    f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {i} --goal_model {goal_model}")    


# for goal_model in goal_models:
#     for _ in range(100):  
#         for datasize in [100, 10]: 
#             if datasize == 10:
#                 seeds = [0, 1, 2, 3, 4]
#             else:
#                 seeds = [0]
#             for seed in seeds:
#                 object_idx = np.random.randint(0,100)

#                 for prim_name in prim_names:
#                     os.system(f"rosrun dvrk_gazebo_control evaluate_object_packaging_multimodal_object_frame.py --headless {str(headless)} "
#                             f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {object_idx} --goal_model {goal_model} "
#                             f"--datasize {datasize} --model_seed {seed}")       

#                     print(f"\n\n*** Elaspsed time: {(timeit.default_timer() - start_time)/3600} hours")

print(f"\n\n*** TOTAL Elaspsed time: {(timeit.default_timer() - start_time)/3600} hours")
