#!/usr/bin/env python3
import sys
import os
import numpy as np
import timeit
from itertools import product

start_time = timeit.default_timer() 

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True

prim_names = ["cylinder"] #["box", "cylinder", "hemis"]
stiffnesses = ["5k", "10k"]


for (prim_name, stiffness) in list(product(prim_names, stiffnesses)):
    # for _ in range(0, 200):
    # print(f"========== Object: {prim_name}_{stiffness}")
    # if not (prim_name != "box" and stiffness != "1k"):
    for _ in range(0, 200):    
        i = np.random.randint(0,100)

        os.system(f"rosrun dvrk_gazebo_control bimanual_multi_box_collect_data_w_rot.py --flex --headless {str(headless)} \
                    --prim_name {prim_name} --stiffness {stiffness} --obj_name {i}")
    # else:
    #     print("========== Already collected data for this object category")
    
print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )

  