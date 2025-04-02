import numpy as np
import trimesh
import os
import pickle
from copy import deepcopy
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
from utils.mesh_utils import create_tet_mesh, apply_euler_rotation_trimesh, simplify_mesh_pymeshlab
from utils.miscellaneous_utils import read_pickle_data, write_pickle_data


save_dir = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_tool"
os.makedirs(os.path.join(save_dir, "mesh_tool"), exist_ok=True)

object_name = "tool_1"

height = 0.4   #np.random.uniform(0.05, 0.15)
radius = 0.2 / 40  

# Create the cylinder
cylinder = trimesh.creation.cylinder(radius=radius, height=height)

# # Translate the mesh to center it at the origin
# cylinder.apply_translation(-cylinder.centroid)
cylinder.apply_translation([0, 0, height / 2])

# Export mesh or visualize
coordinate_frame = trimesh.creation.axis()  
coordinate_frame.apply_scale(0.2)
# cylinder.show()
trimesh.Scene([cylinder, coordinate_frame]).show()

cylinder.export(os.path.join(save_dir, "mesh_tool", f"{object_name}.obj")) 
