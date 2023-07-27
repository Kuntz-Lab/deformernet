#!/usr/bin/env python3
import sys
import os
import timeit
from itertools import product

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

def run_evaluate_loop(headless, prim_name, stiffness, inside, num_obj=10):
    for i in range(0, num_obj):

        os.system(f"rosrun dvrk_gazebo_control evaluate_all_objects_bimanual.py --flex --headless {str(headless)} --prim_name {prim_name}\
                    --stiffness {stiffness} --obj_name {i} --inside {str(inside)}")

def run_collect_goals_loop(headless, prim_name, stiffness, inside, num_obj=10):
    for i in range(0, num_obj):

        os.system(f"rosrun dvrk_gazebo_control bimanual_all_collect_goals_w_rot.py --flex --headless {str(headless)} --obj_name {i}\
                     --prim_name {prim_name} --stiffness {stiffness} --inside {str(inside)}")



headless = True
num_obj = 100
start_time = timeit.default_timer() 


prim_names = ["cylinder"] #["box", "cylinder", "hemis"]
stiffnesses = ["10k"] #["1k", "5k", "10k"]

# ### Collect goals
# for (prim_name, stiffness) in list(product(prim_names, stiffnesses)):
#     obj_type = f"{prim_name}_{stiffness}"

#     run_collect_goals_loop(headless, prim_name, stiffness, inside=True, num_obj=num_obj)
#     # run_collect_goals_loop(headless, prim_name, obj_type, inside=False, num_obj=num_obj)




### Evaluate
inside_options = [True]   #[True, False]

for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    print("=====================================")
    print(prim_name, stiffness, inside)
    
    run_evaluate_loop(headless, prim_name, stiffness, inside, num_obj=num_obj)


                
            
        






print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )





