
import sys
import os
import numpy as np
import timeit
sys.path.append("../")
from util.retraction_cutting_utils import count_matches

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

# data_recording_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/data/retraction_cutting"
# os.makedirs(data_recording_path, exist_ok=True)
# filenames = os.listdir(data_recording_path)


headless = True  #True
start_time = timeit.default_timer()
categoies = ["cylinder", "ellipsoid"]    #["cylinder", "ellipsoid"]
num_object_per_category = 100


# for _ in range(1000):   
#     for category in categoies:
#         for object_idx in range(num_object_per_category):
#             obj_name = f"{category}_{object_idx}"

#             os.system(f"rosrun dvrk_gazebo_control collect_data_retraction_cutting.py " \
#             f"--headless {str(headless)} --obj_name {obj_name}")   

#         print("Elapsed time", (timeit.default_timer() - start_time)/3600)


# # Collect data
# for _ in range(100000):   
#     object_idx = np.random.randint(num_object_per_category)
#     obj_name = f"cylinder_{object_idx}"

#     # os.system(f"rosrun dvrk_gazebo_control collect_data_retraction_tool.py " \
#     # f"--headless {str(headless)} --obj_name {obj_name}")

#     os.system(f"rosrun dvrk_gazebo_control collect_data_retraction_tool_deformernet.py " \
#     f"--headless {str(headless)} --obj_name {object_idx}")            

#     print(f"\n\n*** Elaspsed time: {(timeit.default_timer() - start_time)/3600} hours")


# Evaluate jointly DiffDef + DeformerNet on retraction tool task
goal_models = ["diffdef"] #["diffdef", "defgoalnet"]

# for goal_model in goal_models:
#     for _ in range(150):   
#         object_idx = np.random.randint(num_object_per_category)
#         obj_name = f"cylinder_{object_idx}"

#         os.system(f"rosrun dvrk_gazebo_control evaluate_data_retraction_tool.py " \
#         f"--headless {str(headless)} --obj_name {obj_name} --goal_model {goal_model}")          

#         print(f"\n\n*** Elaspsed time: {(timeit.default_timer() - start_time)/3600} hours")

for goal_model in goal_models:
    for _ in range(150):  
        for datasize in [100, 1000, 10]: 
            if datasize == 10:
                seeds = [0, 1, 2, 3, 4]
            else:
                seeds = [0]
            for seed in seeds:
                object_idx = np.random.randint(num_object_per_category)
                obj_name = f"cylinder_{object_idx}"

                os.system(f"rosrun dvrk_gazebo_control evaluate_data_retraction_tool.py " \
                f"--headless {str(headless)} --obj_name {obj_name} --goal_model {goal_model} --datasize {datasize} --model_seed {seed}")       

                print(f"\n\n*** Elaspsed time: {(timeit.default_timer() - start_time)/3600} hours")

print(f"\n\n*** TOTAL Elaspsed time: {(timeit.default_timer() - start_time)/3600} hours")