import numpy as np
import trimesh
import os
import pickle
import random
import sys
sys.path.append('../../')
from utils.mesh_utils import create_tet_mesh


def create_cylinder_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True, seed=0):
    np.random.seed(seed)
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")
        radius = np.random.uniform(low = 0.03, high = 0.06)
        height = np.random.uniform(low = 0.23, high = 0.40)
        mesh = trimesh.creation.cylinder(radius=radius, height=height)

        youngs_mean = 10000
        youngs_std = 1000        
        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "cylinder"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)
        
        primitive_dict[object_name] = {'radius': radius, 'height': height, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_cylinder.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def create_box_mesh_datatset(save_mesh_dir, type, num_mesh=100, save_pickle=True, seed=0):
    np.random.seed(seed)
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")

        thickness = np.random.uniform(low = 0.02, high = 0.03)
        # sample = np.random.uniform(low = 0.075, high = 0.25, size=2)
        sample = np.random.uniform(low = 0.075, high = 0.20, size=2)
        height, width = max(sample), min(sample)
        

        mesh = trimesh.creation.box((height, width, thickness))

        if type == '1k':
            youngs_mean = 1000
            youngs_std = 200        
        elif type == '5k':
            youngs_mean = 5000
            youngs_std = 1000  
        elif type == '10k':    
            youngs_mean = 10000
            youngs_std = 1000  

        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "box"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet_mesh(save_mesh_dir, object_name, mesh_extension='.stl')
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_box.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def create_hemis_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True, seed=0):
    np.random.seed(seed)
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")
        radius = np.random.uniform(low = 0.2, high = 0.3)
        origin = radius/0.2 * 0.1 
        ratio = np.random.uniform(low = 1.5, high = 4)

        # youngs_mean = 5000
        # youngs_std = 1000       
        youngs_mean = 10000
        youngs_std = 1000 

        youngs = np.random.normal(youngs_mean, youngs_std)

        mesh = trimesh.creation.icosphere(radius = radius)   # hemisphere
        mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0,origin], cap=True)

        vertices_transformed = mesh.vertices * np.array([1./ratio,1,1])
        mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh.faces)
                       

        shape_name = "hemis"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)

        primitive_dict[object_name] = {'radius': radius, 'origin': origin, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_hemis.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


## 1000-200, 5000-1000, 10000-1000

mesh_dir = "/home/baothach/sim_data/Custom/Custom_mesh/physical_dvrk/multi_box_5kPa_eval"
os.makedirs(mesh_dir, exist_ok=True)
create_box_mesh_datatset(mesh_dir, type='5k', num_mesh=100, seed=2024) # seed=0

# mesh_dir = "/home/baothach/sim_data/Custom/Custom_mesh/multi_cylinders_1000Pa"
# create_cylinder_mesh_datatset(mesh_dir)

# mesh_dir = "/home/baothach/sim_data/Custom/Custom_mesh/multi_hemis_10kPa"
# create_hemis_mesh_datatset(mesh_dir)




