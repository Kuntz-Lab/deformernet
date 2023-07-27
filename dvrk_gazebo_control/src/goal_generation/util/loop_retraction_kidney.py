import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True

start_time = timeit.default_timer() 

for i in range(0, 10000):
    kidney_idx = np.random.randint(low=0, high=100)

    os.system(f"rosrun dvrk_gazebo_control collect_data_goal_generation_retraction_kidney.py \
            --kidney_idx {kidney_idx} --headless {str(headless)}")
        
print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )