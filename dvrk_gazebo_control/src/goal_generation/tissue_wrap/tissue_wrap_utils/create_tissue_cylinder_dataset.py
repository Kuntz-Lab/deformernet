import trimesh
import numpy as np
import os
from util_functions import create_tet_mesh, write_pickle_data
import pickle

np.random.seed(0)

### Original cylinder and tissue
# trimesh.creation.cylinder(radius=0.015, height=0.1) # cylinder
# mesh = trimesh.creation.box((0.2, 0.15, 0.018))  # tissue

main_path = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/objects_dataset_eval"
cylinder_path = os.path.join(main_path, "cylinder")
cylinder_dict_path = os.path.join(cylinder_path, "specification")
cylinder_mesh_path = os.path.join(cylinder_path, "mesh")

tissue_path = os.path.join(main_path, "tissue")
tissue_dict_path = os.path.join(tissue_path, "specification")
tissue_mesh_path = os.path.join(tissue_path, "mesh")

os.makedirs(cylinder_path, exist_ok=True)
os.makedirs(tissue_path, exist_ok=True)
os.makedirs(cylinder_dict_path, exist_ok=True)
os.makedirs(tissue_dict_path, exist_ok=True)
os.makedirs(cylinder_mesh_path, exist_ok=True)
os.makedirs(tissue_mesh_path, exist_ok=True)

num_objects = 200     #200
min_width_val, max_width_val = 0.10, 0.20

tissue_length_to_width_ratio = 0.25/0.19

# tissue_widths = np.ones(num_objects) * 0.20
tissue_widths = np.random.uniform(low=min_width_val, high=max_width_val, size=num_objects)
tissue_lengths = np.random.uniform(low=0.8, high=1.2, size=num_objects) * tissue_widths * \
                tissue_length_to_width_ratio 
tissue_thickness = 0.015                

cylinder_lengths = tissue_widths
cylinder_radii = np.random.uniform(low=0.01, high=0.015, size=num_objects)

for i in range(tissue_widths.shape[0]):
    
    print(f"Object {i+1}/{tissue_widths.shape[0]}")
    
    cylinder_mesh = trimesh.creation.cylinder(radius=cylinder_radii[i], height=cylinder_lengths[i])
    
    tissue_mesh = trimesh.creation.box((tissue_lengths[i], 
                                        tissue_widths[i], 
                                        tissue_thickness))     # 0.015 0.02
        
    cylinder_fname = os.path.join(cylinder_mesh_path, f"cylinder_{i}.obj")
    tissue_fname = os.path.join(tissue_mesh_path, f"tissue_{i}.stl")
    
    tissue_mesh.export(tissue_fname)        
    create_tet_mesh(tissue_mesh_path, f"tissue_{i}", coarsen=False, verbose=False)
    
    cylinder_mesh.export(cylinder_fname)  
    
    
    cylinder_data = {"radius": cylinder_radii[i], "length": cylinder_lengths[i]}
    cylinder_dict_fname = os.path.join(cylinder_dict_path, f"cylinder_{i}.pickle")
    write_pickle_data(cylinder_data, cylinder_dict_fname)
    
    tissue_data = {"width": tissue_widths[i], "length": tissue_lengths[i], "thickness": tissue_thickness}
    tissue_dict_fname = os.path.join(tissue_dict_path, f"tissue_{i}.pickle")
    write_pickle_data(tissue_data, tissue_dict_fname)    