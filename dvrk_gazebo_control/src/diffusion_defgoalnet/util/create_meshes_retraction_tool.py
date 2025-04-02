import numpy as np
import trimesh
import os
import pickle
from copy import deepcopy
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
from utils.mesh_utils import create_tet_mesh, apply_euler_rotation_trimesh
from utils.miscellaneous_utils import read_pickle_data, write_pickle_data


save_dir = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_tool"
os.makedirs(os.path.join(save_dir, "mesh"), exist_ok=True)

export_tri_mesh = True#False
export_tet_mesh = True
vis = False
num_object_per_category = 100

categoies = ["cylinder"]

for category in categoies:
    for object_idx in range(num_object_per_category):
        object_name = f"{category}_{object_idx}"
        print(f"\n========= Object: {object_name}")
        if export_tri_mesh:
            if category == "cylinder":
                # height = np.random.uniform(0.05, 0.15)
                height = np.random.uniform(0.05, 0.12)
                ratio = np.random.uniform(2.5, 5)
                radius = height / ratio  
                
                plane_origin=[0,0.0,-height * 0.2]

                mesh = trimesh.creation.cylinder(radius=radius, height=height)

            slicing_angles = [0, 0, 0]
            plane_normal=[0,0,1]
            apply_euler_rotation_trimesh(mesh, *slicing_angles, degrees=True)
            mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)

            # Shift the object such that the bottom of the mesh is aligned with the z=0 plane
            lowest_point_z = mesh.bounds[0, 2]
            mesh.apply_translation([0, 0, -lowest_point_z])

            coordinate_frame = trimesh.creation.axis()  
            coordinate_frame.apply_scale(0.2)
            meshes = [mesh, coordinate_frame]

            # Save the mesh and other information
            if category == "cylinder":
                mesh.export(os.path.join(save_dir, "mesh", f"{object_name}.obj")) 
                info = {"radius": radius, "height": height}
                write_pickle_data(info, os.path.join(save_dir, "mesh", f"{object_name}_info.pickle"))

            if vis:
                trimesh.Scene(meshes).show()



        if export_tet_mesh:
            print("Generating tetrahedral mesh ...")
            create_tet_mesh(os.path.join(save_dir, "mesh"), 
                            object_name, mesh_extension='.obj',
                            coarsen=False, verbose=False)

