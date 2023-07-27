import numpy as np
import trimesh
import os
import pickle
import random
import open3d
from copy import deepcopy

def create_tet(mesh_dir, object_name):
    # STL to mesh
    import os
    os.chdir('/home/baothach/fTetWild/build') 
    mesh_path = os.path.join(mesh_dir, object_name+'.stl')
    save_fTetwild_mesh_path = os.path.join(mesh_dir, object_name + '.mesh')
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path + " >/dev/null")


    # Mesh to tet:
    mesh_file = open(os.path.join(mesh_dir, object_name + '.mesh'), "r")
    tet_output = open(
        os.path.join(mesh_dir, object_name + '.tet'), "w")

    # Parse .mesh file
    mesh_lines = list(mesh_file)
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    vertices_start = mesh_lines.index('Vertices')
    num_vertices = mesh_lines[vertices_start + 1]

    vertices = mesh_lines[vertices_start + 2:vertices_start + 2
                        + int(num_vertices)]

    tetrahedra_start = mesh_lines.index('Tetrahedra')
    num_tetrahedra = mesh_lines[tetrahedra_start + 1]
    tetrahedra = mesh_lines[tetrahedra_start + 2:tetrahedra_start + 2
                            + int(num_tetrahedra)]

    # print("# Vertices, # Tetrahedra:", num_vertices, num_tetrahedra)

    # Write to tet output
    tet_output.write("# Tetrahedral mesh generated using\n\n")
    tet_output.write("# " + num_vertices + " vertices\n")
    for v in vertices:
        tet_output.write("v " + v + "\n")
    tet_output.write("\n")
    tet_output.write("# " + num_tetrahedra + " tetrahedra\n")
    for t in tetrahedra:
        line = t.split(' 0')[0]
        line = line.split(" ")
        line = [str(int(k) - 1) for k in line]
        l_text = ' '.join(line)
        tet_output.write("t " + l_text + "\n")


def create_box_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True, seed=0, vis=False):
    np.random.seed(seed)

    
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")
        
        # Sample box dimensions
        height = np.random.uniform(low = 0.2, high = 0.5)
        width = np.random.uniform(low = 0.2, high = 0.5)
        thickness = np.random.uniform(low = 0.04, high = 0.08)
        youngs = 1000

        # Sampling attachment point conditions
        x = np.random.uniform(low = -1.0, high = 0.0)   # how steep the slope is
        y = np.sqrt(1-x**2) * np.random.choice([-1,1])  # -1 or 1 are the two opposite directions of the cut
        slice_origin = np.random.uniform(low = -0.9*height/2, high = 0.9*height/2)

        # Create and save object and base meshes
        mesh_obj = trimesh.creation.box((height, width, thickness))  
        mesh_base = trimesh.creation.box((height, width, 0.005))       
        mesh_base = trimesh.intersections.slice_mesh_plane(mesh=mesh_base, plane_normal=[x,y,0], plane_origin=[slice_origin,0,0], cap=True)

        if vis:
            copied_mesh_obj = deepcopy(mesh_obj)
            T = trimesh.transformations.translation_matrix([0., 0, -0.1])
            copied_mesh_obj.apply_transform(T)
            meshes = [copied_mesh_obj, mesh_base]
            trimesh.Scene(meshes).show()

      
        object_name = f"box_{i}"
        base_name = f"base_{i}"
        mesh_obj.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)
        mesh_base.export(os.path.join(save_mesh_dir, base_name+'.obj'))
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def create_handcrafted_box_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True, seed=None, vis=False):
    if seed is not None:
        np.random.seed(seed)

    
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")
        
        youngs = 1000
        height, width, thickness = (0.45, 0.3, 0.06)
        if i == 0:            
            x = -1
            y = np.sqrt(1-x**2)
            slice_origin = 0*height/2
        elif i== 1:
            x = -1
            y = np.sqrt(1-x**2)
            slice_origin = -0.9 * height/2

        # Create and save object and base meshes
        mesh_obj = trimesh.creation.box((height, width, thickness))  
        mesh_base = trimesh.creation.box((height, width, 0.005))       
        mesh_base = trimesh.intersections.slice_mesh_plane(mesh=mesh_base, plane_normal=[x,y,0], plane_origin=[slice_origin,0,0], cap=True)

        if vis:
            copied_mesh_obj = deepcopy(mesh_obj)
            T = trimesh.transformations.translation_matrix([0., 0, -0.1])
            copied_mesh_obj.apply_transform(T)
            meshes = [copied_mesh_obj, mesh_base]
            trimesh.Scene(meshes).show()

      
        object_name = f"box_{i}"
        base_name = f"base_{i}"
        mesh_obj.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)
        mesh_base.export(os.path.join(save_mesh_dir, base_name+'.obj'))
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

# mesh_dir = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting/multi_box"
# os.makedirs(mesh_dir,exist_ok=True)
# create_box_mesh_datatset(mesh_dir, num_mesh=100, vis=False)

# mesh_dir = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/Custom_mesh"
# os.makedirs(mesh_dir,exist_ok=True)
# create_box_mesh_datatset(mesh_dir, num_mesh=10, vis=False, seed=2023)

mesh_dir = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/Custom_mesh_2"
os.makedirs(mesh_dir,exist_ok=True)
create_handcrafted_box_mesh_datatset(mesh_dir, num_mesh=2, vis=True)