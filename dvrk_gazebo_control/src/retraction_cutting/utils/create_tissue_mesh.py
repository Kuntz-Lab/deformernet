import numpy as np
import trimesh
import os
import pickle
import random
import open3d

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


# mesh1 = trimesh.creation.box((0.45, 0.3, 0.06))     # original
mesh1 = trimesh.creation.box((0.45, 0.3, 0.06))        #let's vary 0.2-0.5 for H,W; 0.04-0.1 for thickness
mesh_dir = '/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting'
object_name = "box_2"

meshes = [mesh1]
trimesh.Scene(meshes).show()
mesh1.export(os.path.join(mesh_dir, object_name+'.stl'))
create_tet(mesh_dir, object_name)


# mesh1 = trimesh.creation.box((0.45, 0.3, 0.005))  # original
mesh1 = trimesh.creation.box((0.45, 0.3, 0.06))     #let's vary x from 0 to -1; plane_origin[0] from -0.9*height to 0.8*height
x = -0.9#-0.75
y = -np.sqrt(1-x**2)
mesh2 = trimesh.intersections.slice_mesh_plane(mesh=mesh1, plane_normal=[x,y,0], plane_origin=[0,0,0], cap=True)
T = trimesh.transformations.translation_matrix([0., 0, -0.1])
mesh1.apply_transform(T)

mesh_dir = '/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting'
object_name = "base_2"

meshes = [mesh1, mesh2]
trimesh.Scene(meshes).show()
# mesh2.export(os.path.join(mesh_dir, object_name+'.obj'))



