import numpy as np
import trimesh
import os
import pickle
from copy import deepcopy
# import roslib.packages as rp
# import sys
# pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
# sys.path.append(pkg_path + '/src/utils')
# from mesh_utils import create_tet_mesh

def create_tet(mesh_dir, object_name):
    # STL to mesh
    import os
    os.chdir('/home/baothach/fTetWild/build') 
    mesh_path = os.path.join(mesh_dir, object_name+'.obj')
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



object_name = "cylinder"
radius = 0.0128
height = 0.129
mesh = trimesh.creation.cylinder(radius=radius, height=height)


save_dir = "/home/baothach/shape_servo_data/TAMP/object_data"
# mesh.export(os.path.join(save_dir, "mesh", f"{object_name}.obj"))
# create_tet(save_dir, "cylinder")

info = {"height": height, "radius": radius}
with open(os.path.join(os.path.join(save_dir, "info"), f"{object_name}.pickle"), 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)