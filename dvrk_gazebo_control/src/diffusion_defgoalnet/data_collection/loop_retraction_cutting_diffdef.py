
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
# num_object_per_category = 50     #50


# for _ in range(1000):   
#     for category in categoies:
#         for object_idx in range(num_object_per_category):
#             obj_name = f"{category}_{object_idx}"

#             os.system(f"rosrun dvrk_gazebo_control collect_data_retraction_cutting.py " \
#             f"--headless {str(headless)} --obj_name {obj_name}")
                
        

#         print("Elapsed time", (timeit.default_timer() - start_time)/3600)

num_object_per_category = 100 
for _ in range(100000):   
    object_idx = np.random.randint(num_object_per_category)
    obj_name = f"cylinder_{object_idx}"

    # os.system(f"rosrun dvrk_gazebo_control collect_data_retraction_tool.py " \
    # f"--headless {str(headless)} --obj_name {obj_name}")

    os.system(f"rosrun dvrk_gazebo_control collect_data_retraction_tool_deformernet.py " \
    f"--headless {str(headless)} --obj_name {object_idx}")            

    print(f"\n\n*** Elaspsed time: {(timeit.default_timer() - start_time)/3600} hours")