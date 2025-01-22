import numpy as np
import trimesh
import os
import pickle
import random
import sys
sys.path.append('../')
from utils.mesh_utils import create_tet_mesh, simplify_mesh_pymeshlab
from utils.miscellaneous_utils import write_pickle_data


def random_process_mesh(file_path, scale, output_path, vis=False):
    """
    Processes a 3D mesh by centering it at its centroid and scaling it.
    
    Args:
        file_path (str): Path to the input .obj file.
        scale (float): The target scale of the mesh.
        output_path (str): Path to save the processed mesh.
        vis (bool, optional): If True, visualizes the mesh before and after processing in the same scene. Default is False.
    
    Returns:
        None
    """
    # Load the mesh
    mesh = trimesh.load(file_path)
    mesh = simplify_mesh_pymeshlab(mesh, target_num_vertices=300)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The provided file does not contain a valid mesh.")

    # Keep a copy of the original mesh for visualization
    original_mesh = mesh.copy()
    
    # Center the mesh at its centroid
    centroid = mesh.centroid
    mesh.vertices -= centroid
    
    # Scale the mesh
    # max_dimension = max(mesh.bounding_box.extents)
    # scale_factor = scale / max_dimension
    mesh.vertices *= scale
    
    if vis:
        # Create a scene to visualize both meshes
        scene = trimesh.Scene()
        # # Add original mesh (in red for contrast)
        # original_mesh.visual.vertex_colors = [255, 0, 0, 150]  # Red with transparency
        # scene.add_geometry(original_mesh)

        # Add processed mesh (in green)
        mesh.visual.vertex_colors = [0, 255, 0, 150]  # Green with transparency
        scene.add_geometry(mesh)
        
        print("Visualizing original (red) and processed (green) meshes...")
        scene.show()
    
    # Save the processed mesh to the output path
    mesh.export(output_path)
    print(f"Processed mesh saved to {output_path}")

    mesh_dims = mesh.extents
    return mesh_dims

    
main_dir = "/home/baothach/sim_data/Custom/Custom_objects/CAO_tissue"
num_mesh = 10#0
seed = 0
mesh_info_dict = {}
np.random.seed(seed) 
    
for i in range(num_mesh):
    object_name = f"tissue_{i}"
    print(f"Processing {object_name} ...")

    tri_mesh_dir = os.path.join(main_dir, "tri_mesh")
    tet_mesh_dir = os.path.join(main_dir, "tet_mesh")
    os.makedirs(tri_mesh_dir, exist_ok=True)
    os.makedirs(tet_mesh_dir, exist_ok=True)

    obj_path = os.path.join(main_dir, "original_CAO_tissue.obj")
    random_scale = np.random.uniform(0.002, 0.005)

    mesh_dims = random_process_mesh(obj_path, random_scale, os.path.join(tri_mesh_dir, f"{object_name}.obj"), vis=False)
    mesh_info_dict[object_name] = {"scale": random_scale, "mesh_dims": mesh_dims}
    # print(f"\n*** Mesh dimensions: {mesh_dims}")

    create_tet_mesh(tet_mesh_dir, object_name, mesh_extension='.obj', tri_mesh_dir=tri_mesh_dir, verbose=True)

# Save mesh_info_dict to a pickle file
data = mesh_info_dict
write_pickle_data(data, os.path.join(main_dir, "mesh_info_dict.pickle"))   




