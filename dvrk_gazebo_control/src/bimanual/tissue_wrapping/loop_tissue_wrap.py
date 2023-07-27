import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True#False

start_time = timeit.default_timer() 

for i in range(0, 125):
    os.system("source devel/setup.bash")

    os.system(f"rosrun dvrk_gazebo_control tissue_wrap_multi_cylinders.py --flex --headless {str(headless)}")
        
print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )