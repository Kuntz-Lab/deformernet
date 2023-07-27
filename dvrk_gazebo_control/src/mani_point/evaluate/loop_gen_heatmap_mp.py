
import sys
import os
import numpy as np


pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = True#False

# # Cylinder
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_multi_shapes.py --flex --headless True --obj_name cylinder_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control cylinder_collect_data_multi_shapes.py --flex --obj_name cylinder_" + str(i))



# Boxes:
for goal_idx in range(1, 9):
    # i = 0#np.random.randint(low=0, high=100)
    os.system("source devel/setup.bash")

    if headless:
        os.system(f"rosrun dvrk_gazebo_control gen_heatmap_mp.py --flex --headless True --goal_idx {goal_idx}")
    else:
        os.system(f"rosrun dvrk_gazebo_control gen_heatmap_mp.py --flex --goal_idx {goal_idx} ")


# # Hemis
# for i in range(0, 100):
#     os.system("source devel/setup.bash")

#     if headless:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --headless True --obj_name hemis_" + str(i))
#     else:
#         os.system("rosrun dvrk_gazebo_control hemis_collect_data_multi_shapes.py --flex --obj_name hemis_" + str(i))