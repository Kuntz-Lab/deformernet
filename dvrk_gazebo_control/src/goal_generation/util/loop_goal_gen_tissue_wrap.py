import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True#False

start_time = timeit.default_timer() 

for i in range(0, 10000):
    os.system("source devel/setup.bash")

    os.system(f"rosrun dvrk_gazebo_control collect_data_goal_generation_tissue_wrap.py --flex --headless {str(headless)}")
        
print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )