

import os
import sys
sys.path.append('/home/baothach/dvrk_grasp_pipeline_issac/src/dvrk_env/dvrk_gazebo_control/src')
import GraspDataCollectionClient as dc_class

a = sorted(os.listdir("/home/baothach/sim_data/BigBird/BigBird_mesh"))
print(a.index("advil_liqui_gels"))
# dc_client = dc_class.GraspDataCollectionClient()

# dc_client.get_last_object_id_name()

# print(dc_client.last_object_name.decode('UTF-8')=="custom_box0")
# print(dc_client.cur_object_id)