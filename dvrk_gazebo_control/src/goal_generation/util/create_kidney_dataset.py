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


def create_kidney_urdf_dataset(num_obj, urdf_path, seed=137):
    np.random.seed(seed)
    
    for i in range(num_obj):
        file_name = f"kidney_{i}.urdf"
        f = open(os.path.join(urdf_path, file_name), 'w')
        scale = np.random.uniform(low=0.6, high=1.0) 
        
        urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>            
    <robot name="kidney">
    <link name="kidney">
        <visual>
        <origin xyz="0.0 0.0 0.0"/>
        <geometry>
            <mesh filename="/home/baothach/sim_data/Custom/Custom_mesh/kidney/Ginjal New splified_3.obj" scale="{scale:.3f} {scale:.3f} {scale:.3f}"/>
        </geometry>
        </visual>
        <collision>
        <origin xyz="0.0 0.0 0.0"/>
        <geometry>
            <mesh filename="/home/baothach/sim_data/Custom/Custom_mesh/kidney/Ginjal New splified_3.obj" scale="{scale:.3f} {scale:.3f} {scale:.3f}"/>
        </geometry>
        </collision>
        <inertial>
        <mass value="5000"/>
        <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
        </inertial>
    </link>
    </robot>
        """

        f.write(urdf_str)
        f.close()

urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/goal_generation/muti_kidneys"
create_kidney_urdf_dataset(num_obj=100, urdf_path=urdf_path)