from mesh_utils import create_tet_mesh, simplify_mesh_pymeshlab
import os
import trimesh

mesh_main_path = "/home/baothach/sim_data/Custom/Custom_objects/random_stuff/mesh"

obj_name = "kidney" #"meat"

mesh_dir = mesh_main_path   #os.path.join(mesh_main_path)

tri_mesh_fname = os.path.join(mesh_dir, f"{obj_name}.stl")
tri_mesh = trimesh.load(tri_mesh_fname)
# tri_mesh = simplify_mesh_pymeshlab(tri_mesh, target_num_vertices=100)
tri_mesh = trimesh.convex.convex_hull(tri_mesh.vertices)
tri_mesh = tri_mesh.apply_scale(0.05)   # 0.05
# tri_mesh = trimesh.smoothing.filter_laplacian(tri_mesh, lamb=0.1, iterations=10)

tri_mesh.export(os.path.join(mesh_dir, f"simplified_{obj_name}.stl")) 

coordinate_frame = trimesh.creation.axis()  
# coordinate_frame.apply_scale(0.2)

print(tri_mesh.vertices.shape)
trimesh.Scene([tri_mesh, coordinate_frame]).show()
# tri_mesh.show()
  
# create_tet_mesh(mesh_dir, f"simplified_{obj_name}", coarsen=True, verbose=True, mesh_extension='.stl') 
create_tet_mesh(mesh_dir, f"{obj_name}", coarsen=True, verbose=True, mesh_extension='.obj') 

