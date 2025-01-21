
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
goal_models = ["diffdef", "defgoalnet"]

stiffness = "5k"
for goal_model in goal_models:
    for _ in range(0, 200):
        i = np.random.randint(0,100)
        for prim_name in prim_names:
            os.system(f"rosrun dvrk_gazebo_control evaluate_object_packaging_multimodal.py --headless {str(headless)} "
                    f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {i} --goal_model {goal_model}")    

print(f"Elapsed time (hours): {(timeit.default_timer() - start_time)/3600:.3f}")