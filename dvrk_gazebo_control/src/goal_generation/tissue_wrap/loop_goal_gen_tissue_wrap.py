import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True

start_time = timeit.default_timer() 


for _ in range(0, 100000):  # 100000
    object_idx = np.random.randint(low = 0, high = 100)
    os.system(f"rosrun dvrk_gazebo_control collect_data_goal_generation_tissue_wrap.py --object_idx {object_idx} --headless {str(headless)}")

# for object_idx in range(0, 200):    
#     os.system(f"rosrun dvrk_gazebo_control collect_data_goal_generation_tissue_wrap.py --object_idx {object_idx} --headless {str(headless)}")


# for i in range(0, 100000):

#     os.system(f"rosrun dvrk_gazebo_control collect_data_goal_generation_tissue_wrap.py --flex --headless {str(headless)}")
    

# # model_names = ["pointconv_1000", "pointconv_100"] + [f"randomized/pointconv_10_random_{j}" for j in range(1)]
# model_names = [f"randomized/pointconv_100_random_{j}" for j in range(0,5)]

# for model_name in model_names:
#     for eval_sample_idx in range(0, 100):
#         os.system(f"rosrun dvrk_gazebo_control evaluate_goal_generation_tissue_wrap.py --headless {str(headless)} --model_name {model_name} --eval_sample_idx {eval_sample_idx}")
        
        
print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )