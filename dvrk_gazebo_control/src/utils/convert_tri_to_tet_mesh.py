from mesh_utils import create_tet_mesh, simplify_mesh_pymeshlab
from miscellaneous_utils import write_pickle_data
import os
import trimesh
import numpy as np

mesh_main_path = "/home/baothach/sim_data/Custom/Custom_objects/random_stuff/mesh"

obj_name = "chicken_breast" #"meat"

mesh_dir = mesh_main_path   #os.path.join(mesh_main_path)

tri_mesh_fname = os.path.join(mesh_dir, f"{obj_name}.stl")
tri_mesh = trimesh.load(tri_mesh_fname)
tri_mesh.vertices = tri_mesh.vertices * np.array([1,0.8,1.2])
tri_mesh.visual.face_colors = [125, 125, 125, 255]
tri_mesh.show()

tri_mesh = trimesh.convex.convex_hull(tri_mesh.vertices)
tri_mesh = tri_mesh.apply_scale(0.001)   # 0.05
# tri_mesh = trimesh.smoothing.filter_laplacian(tri_mesh, lamb=0.1, iterations=10)
tri_mesh.vertices = tri_mesh.vertices * np.array([1,0.8,1.2])
tri_mesh = simplify_mesh_pymeshlab(tri_mesh, target_num_vertices=100)

# tri_mesh.export(os.path.join(mesh_dir, f"simplified_{obj_name}.stl")) 

coordinate_frame = trimesh.creation.axis(axis_length=0.1, origin_size=0.01)  
# coordinate_frame.apply_scale(0.1)

# print("tri_mesh.vertices.shape:", tri_mesh.vertices.shape)
print("tri_mesh.extents (cm):", tri_mesh.extents * 100)
# trimesh.Scene([tri_mesh, coordinate_frame]).show()
# tri_mesh.show()
  
# create_tet_mesh(mesh_dir, f"simplified_{obj_name}", output_tet_mesh_name=obj_name, coarsen=True, verbose=True, mesh_extension='.stl') 
# create_tet_mesh(mesh_dir, f"{obj_name}", coarsen=True, verbose=True, mesh_extension='.obj') 


# data = {"extents": tri_mesh.extents}
# write_pickle_data(data, os.path.join(mesh_main_path, f"{obj_name}.pickle"))

