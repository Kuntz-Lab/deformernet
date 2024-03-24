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


save_dir = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_cutting"
os.makedirs(os.path.join(save_dir, "mesh"), exist_ok=True)

export_tri_mesh = True#False
export_tet_mesh = True

object_name = "cylinder_0"
category = "ellipsoid"   # ellipsoid cylinder

if export_tri_mesh:
    radius = 0.02
    height = 0.1
    # radius = 0.03
    # height = 0.1


    slicing_angles = [0, 0, 0] 
    plane_normal=[0,0,1]
    plane_origin=[0,0.0,-height * 0.2]

    mesh1 = trimesh.creation.cylinder(radius=radius, height=height)
    mesh1 = trimesh.intersections.slice_mesh_plane(mesh=mesh1, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)

    radius = 0.05
    ratio = 4/5#10/4.
    mesh = trimesh.creation.icosphere(radius = radius) 


    vertices_transformed = mesh.vertices * np.array([ratio,ratio,1])
    mesh2 = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh.faces)
    
    slicing_angles = [0, 0, 0]
    plane_normal=[0,0,1]
    plane_origin=[0,0,-radius * 0.4]
    mesh2 = trimesh.intersections.slice_mesh_plane(mesh=mesh2, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)

    mesh2.apply_translation([0.1, 0, 0])

    coordinate_frame = trimesh.creation.axis()  
    coordinate_frame.apply_scale(0.1)

    meshes = [mesh1, mesh2, coordinate_frame]
    trimesh.Scene(meshes).show()


    
   
