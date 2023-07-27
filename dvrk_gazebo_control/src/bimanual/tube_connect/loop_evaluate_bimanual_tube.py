
import sys
import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True
start_time = timeit.default_timer()

# # Cylinder
# for i in range(10, 100):
    
#     # os.system(f"rosrun dvrk_gazebo_control bimanual_multi_cylinders_collect_data.py --flex --headless {str(headless)} --obj_name cylinder_{i}")
    
#     i = np.random.randint(0,10)
#     os.system(f"rosrun dvrk_gazebo_control bimanual_multi_cylinders_collect_goals.py --flex --headless {str(headless)} --obj_name cylinder_{i}")
    
#     os.system(f"rosrun dvrk_gazebo_control bimanual_multi_cylinders_evaluate_goals.py --flex --headless {str(headless)}")



## Boxes:
for _ in range(0, 15000):
    # i = np.random.randint(0,100)
    # os.system(f"rosrun dvrk_gazebo_control bimanual_multi_cylinders_tube_collect_data_rot.py --flex --headless {str(headless)} --obj_name cylinder_{0}")
    os.system(f"rosrun dvrk_gazebo_control bimanual_multi_cylinders_tube_collect_data_rot_attached.py --flex --headless {str(headless)} --obj_name cylinder_{0}")

    # i = np.random.randint(0,10)
    # os.system(f"rosrun dvrk_gazebo_control bimanual_multi_cylinders_collect_goals.py --flex --headless {str(headless)} --obj_name cylinder_{i}")
    
    # os.system(f"rosrun dvrk_gazebo_control bimanual_multi_boxes_evaluate_goals_w_mp.py --flex --headless {str(headless)}")


# # Hemis
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --headless True --obj_name hemis_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --obj_name hemis_" + str(i))


print("Elapsed time", timeit.default_timer() - start_time)