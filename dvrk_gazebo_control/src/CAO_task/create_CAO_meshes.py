import numpy as np
import trimesh
import os
import pickle
import random
import sys
sys.path.append('../')
from utils.mesh_utils import create_tet_mesh, simplify_mesh_pymeshlab


def create_box_mesh_datatset(save_mesh_dir, type, num_mesh=100, save_pickle=True, seed=0):
    np.random.seed(seed) 
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")

        thickness = np.random.uniform(low = 0.02, high = 0.03)
        # # sample = np.random.uniform(low = 0.075, high = 0.25, size=2)
        # sample = np.random.uniform(low = 0.075, high = 0.20, size=2)
        # height, width = max(sample), min(sample)

        # min_dim, max_dim = 0.075, 0.20
        # min_ratio = 1.5 
        # width = np.random.uniform(low = min_dim, high = max_dim/min_ratio)
        # height = np.random.uniform(low = width * min_ratio, high = max_dim)

        min_dim, max_dim = 0.12, 0.24
        scale = 0.3
        min_dim, max_dim = min_dim, max_dim
        
        candidates = np.random.uniform(low = min_dim, high = max_dim, size=2)
        height, width = max(candidates), min(candidates)
        thickness = np.random.uniform(low = min_dim, high = max_dim)

        height, width, thickness = 0.24, 0.24, 0.24

        height, width, thickness = height*scale, width*scale, thickness*scale
        mesh = trimesh.creation.box((height, width, thickness))
        # mesh = simplify_mesh_pymeshlab(mesh, target_num_vertices=300)
        # coor = trimesh.creation.axis(
        #     origin_size=0.01,
        #     axis_length=0.5,
        #     # axis_radius=axis_radius
        # )
        # trimesh.Scene([mesh, coor]).show()
        # # mesh.show()

        if type == '1k':
            youngs_mean = 1000
            youngs_std = 200        
        elif type == '5k':
            youngs_mean = 5000
            youngs_std = 1000  
        elif type == '10k':    
            youngs_mean = 10000
            youngs_std = 1000  
        else:
            raise ValueError("type must be either '1k', '5k', or '10k'")

        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "box"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet_mesh(save_mesh_dir, object_name, mesh_extension='.stl', verbose=True)
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}

        data = primitive_dict
        with open(os.path.join(save_mesh_dir, "primitive_dict_box.pickle"), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_box.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


# ## 1000-200, 5000-1000, 10000-1000
stiffness = "1k"
mesh_dir = f"/home/baothach/sim_data/Custom/Custom_mesh/physical_CAO/multi_box_{stiffness}Pa"
os.makedirs(mesh_dir, exist_ok=True)
create_box_mesh_datatset(mesh_dir, type=stiffness, num_mesh=1, seed=None) # seed=0






