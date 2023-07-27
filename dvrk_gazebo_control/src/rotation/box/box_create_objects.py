import numpy as np
import trimesh
import os
import pickle
import random

def create_tet(mesh_dir, object_name):
    # STL to mesh
    import os
    os.chdir('/home/baothach/fTetWild/build') 
    mesh_path = os.path.join(mesh_dir, object_name+'.stl')
    save_fTetwild_mesh_path = os.path.join(mesh_dir, object_name + '.mesh')
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path)


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

    print("# Vertices, # Tetrahedra:", num_vertices, num_tetrahedra)

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


def create_evaluation_box_mesh_datatset(save_mesh_dir, type, start_idx=0, num_mesh=10, inside=True, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    for i in range(start_idx, num_mesh):

        if inside:        
            width = np.random.uniform(low = 0.1, high = 0.2)
            height = np.random.uniform(low = width, high = 0.3)
            thickness = np.random.uniform(low = 0.04, high = 0.06)
            
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
        else:
            width = np.random.uniform(low = 0.2, high = 0.4)
            height = np.random.uniform(low = width, high = 0.43)
            thickness = np.random.uniform(low = 0.04, high = 0.06)            
                         
            
            if type == '1k':
                youngs = random.uniform(*random.choice([(400, 600), (1400, 1800)]))
            elif type == '5k':
                youngs = random.uniform(*random.choice([(1000, 3000), (7000, 9000)]))   
            elif type == '10k':  
                youngs = random.uniform(*random.choice([(6000, 8000), (12000, 14000)]))

        mesh = trimesh.creation.box((height, width, thickness))

        shape_name = "box"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)
        
        primitive_dict = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}
   

        data = primitive_dict
        with open(os.path.join(save_mesh_dir, object_name + ".pickle"), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 



## 1000-200, 5000-1000, 10000-1000  
# intial seed: 0

# mesh_dir = '/home/baothach/shape_servo_data/evaluation/meshes/box_5k/inside'
# create_evaluation_box_mesh_datatset(mesh_dir, type = '5k', num_mesh=10, inside=True)

# mesh_dir = '/home/baothach/shape_servo_data/evaluation/meshes/box_5k/outside'
# create_evaluation_box_mesh_datatset(mesh_dir, type = '5k', num_mesh=10, inside=False)


for obj_type in ['1k', '5k', '10k']:
    mesh_dir = f'/home/baothach/shape_servo_data/evaluation/meshes/box_{obj_type}/inside'
    create_evaluation_box_mesh_datatset(mesh_dir, type = obj_type, num_mesh=10, inside=True)

    mesh_dir = f'/home/baothach/shape_servo_data/evaluation/meshes/box_{obj_type}/outside'
    create_evaluation_box_mesh_datatset(mesh_dir, type = obj_type, num_mesh=10, inside=False)