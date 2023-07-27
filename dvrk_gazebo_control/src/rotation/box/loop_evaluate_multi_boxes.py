#!/usr/bin/env python3
import sys
import os
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

def run_evaluate_loop(headless, prim_name, obj_type, inside, num_obj=10):
    for i in range(8, num_obj):
        os.system("source devel/setup.bash")

        os.system(f"rosrun dvrk_gazebo_control evaluate_{prim_name}.py --flex --headless {str(headless)} --obj_name {prim_name}_{i}\
                    --obj_type {obj_type} --inside {str(inside)}")

def run_collect_goals_loop(headless, prim_name, obj_type, inside, num_obj=10):
    for i in range(0, num_obj):
        os.system("source devel/setup.bash")

        os.system(f"rosrun dvrk_gazebo_control collect_goals_{prim_name}.py --flex --headless {str(headless)} --obj_name {prim_name}_{i}\
                    --obj_type {obj_type} --inside {str(inside)}")


headless = False
start_time = timeit.default_timer() 

#### Box
prim_name = "box"
# obj_type = "box_1k"
num_obj = 10


for obj_type in ["box_5k"]:
    # collect goal data

    # run_collect_goals_loop(headless, prim_name, obj_type, inside=True, num_obj=num_obj)
    # run_collect_goals_loop(headless, prim_name, obj_type, inside=False, num_obj=num_obj)


    # evaluate
    # run_evaluate_loop(headless, prim_name, obj_type, inside=True, num_obj=num_obj)
    run_evaluate_loop(headless, prim_name, obj_type, inside=False, num_obj=num_obj)




print("Elapsed time evaluate ", timeit.default_timer() - start_time)





