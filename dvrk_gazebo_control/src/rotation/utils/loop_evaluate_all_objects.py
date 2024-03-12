#!/usr/bin/env python3
import sys
import os
import timeit
from itertools import product

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

def run_evaluate_loop(headless, prim_name, stiffness, inside, use_rot, use_mp_input, num_obj=10):
    for i in range(0, num_obj):

        # os.system(f"rosrun dvrk_gazebo_control evaluate_all_objects_w_predicted_mp.py --flex --headless {str(headless)} --prim_name {prim_name}\
        #             --stiffness {stiffness} --obj_name {i} --inside {str(inside)} --use_rot {str(use_rot)} --use_mp_input {str(use_mp_input)} --mp_method {mp_method}")
        os.system(f"rosrun dvrk_gazebo_control evaluate_all_objects_w_predicted_mp_count_steps.py --flex --headless {str(headless)} --prim_name {prim_name}\
                    --stiffness {stiffness} --obj_name {i} --inside {str(inside)} --use_rot {str(use_rot)} --use_mp_input {str(use_mp_input)} --mp_method {mp_method}")


def run_collect_goals_loop(headless, prim_name, stiffness, inside, num_obj=10):
    for i in range(0, num_obj):

        os.system(f"rosrun dvrk_gazebo_control collect_goals_{prim_name}_rotation_2.py --flex --headless {str(headless)} --obj_name {i}\
                     --prim_name {prim_name} --stiffness {stiffness} --inside {str(inside)}")

# def run_evaluate_loop(headless, prim_name, stiffness, inside, use_rot, use_mp_input, num_obj=10):
#     for i in range(0, num_obj):

#         os.system(f"rosrun dvrk_gazebo_control evaluate_all_objects_w_predicted_mp_modified_loss_ratio.py --flex --headless {str(headless)} --prim_name {prim_name}\
#                     --stiffness {stiffness} --obj_name {i} --inside {str(inside)} --use_rot {str(use_rot)} --use_mp_input {str(use_mp_input)} --mp_method {mp_method}")

# def run_collect_goals_loop(headless, prim_name, stiffness, inside, num_obj=10):
#     for i in range(0, num_obj):

#         os.system(f"rosrun dvrk_gazebo_control collect_goals_{prim_name}_rotation_modified_loss_ratio.py --flex --headless {str(headless)} --obj_name {i}\
#                      --prim_name {prim_name} --stiffness {stiffness} --inside {str(inside)}")


headless = True
num_obj = 10 #10
start_time = timeit.default_timer() 

### Collect goals
prim_names = ["box", "cylinder", "hemis"] #["box", "cylinder", "hemis"]
stiffnesses = ["10k"] #["1k", "5k", "10k"] 


# for (prim_name, stiffness) in list(product(prim_names, stiffnesses)):
#     obj_type = f"{prim_name}_{stiffness}"

#     run_collect_goals_loop(headless, prim_name, stiffness, inside=True, num_obj=num_obj)
#     # run_collect_goals_loop(headless, prim_name, obj_type, inside=False, num_obj=num_obj)




### Evaluate
# prim_names = ["hemis"] #["hemis", "cylinder", "hemis"]
# stiffnesses = ["10k"] #["1k", "5k", "10k"]
mp_methods = ["ground_truth", "classifier", "keypoint"] # ["ground_truth", "classifier", "keypoint"] #["ground_truth", "dense_predictor", "classifier", "keypoint"]
inside_options = [True]   #[True, False]
use_rot_options = [True, False]#[True, False]
use_mp_input_options = [True, False]#[True, False]



for (prim_name, stiffness, inside) in list(product(prim_names, stiffnesses, inside_options)):
    print("=====================================")
    print(prim_name, stiffness, inside)
    

    for (use_rot, use_mp_input) in list(product(use_rot_options, use_mp_input_options)):
        mp_method = "dense_predictor"
        run_evaluate_loop(headless, prim_name, stiffness, inside, use_rot, use_mp_input, num_obj=num_obj)
                
 
    for mp_method in mp_methods:       
        use_rot = True
        use_mp_input = True
        run_evaluate_loop(headless, prim_name, stiffness, inside, use_rot, use_mp_input, num_obj=num_obj)            
        


print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )





