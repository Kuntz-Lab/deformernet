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
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path + " --coarsen >/dev/null")


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


# # mesh1 = trimesh.creation.box((0.078, 0.039, 0.026))     
# mesh1 = trimesh.creation.box((0.07, 0.07, 0.025))  
# mesh1.apply_scale(1./3)
# # print("Number of faces:", len(mesh1.faces))
# # mesh1 = mesh1.simplify_quadratic_decimation(face_count=len(mesh1.faces)//2)
# # # mesh1=trimesh.remesh.subdivide(mesh1, order=2)

# for _ in range(3):
#     v, f = trimesh.remesh.subdivide(vertices=mesh1.vertices,faces=mesh1.faces)
#     mesh1.vertices = v
#     mesh1.faces = f
#     print("Number of faces:", len(mesh1.faces), len(f))
# mesh1 = trimesh.smoothing.filter_laplacian(mesh1, lamb=1, iterations=20)
# # mesh1.show()
# # # mesh1 = mesh1.smoothed()
# mesh1 = mesh1.simplify_quadratic_decimation(face_count=200)
# print("Number of faces:", len(mesh1.faces))
# print(mesh1.extents)
# mesh1.show()

# obj_name = "square"
# mesh_dir = f'/home/baothach/sim_data/stress_prediction_data/objects/{obj_name}'
# os.makedirs(mesh_dir, exist_ok=True)
# mesh1.export(os.path.join(mesh_dir, obj_name+'.stl'))
# create_tet(mesh_dir, obj_name)


obj_name = "rectangle"
# object_scale = 1.3
fname_object = f'/home/baothach/stress_field_prediction/examples/{obj_name}/{obj_name}.stl'
mesh2 = trimesh.load(fname_object)
# print("Number of vertices:", len(mesh2.vertices))
# mesh2.apply_scale(object_scale)
# # mesh2.apply_translation([0,0.07,0])

# ori_rectangle_dims = np.array([0.078, 0.039, 0.026])
# desired_dims = np.array([0.07, 0.07, 0.025]) 
# meshes = [deepcopy(mesh2).apply_translation([0,0.07,0])]
# vertices_transformed = mesh2.vertices * desired_dims/ori_rectangle_dims
# mesh_transformed = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh2.faces)
# # mesh_transformed = mesh_transformed.simplify_quadratic_decimation(face_count=len(mesh_transformed.faces)//3)
# print("Number of vertices transformed:", len(mesh_transformed.vertices))
# mesh_transformed.apply_scale(1/5)

# # meshes.append(mesh_transformed)
# # trimesh.Scene(meshes).show()

mesh2.show()
obj_name = "rectangle_test"
mesh_dir = f'/home/baothach/sim_data/stress_prediction_data/objects/{obj_name}'
os.makedirs(mesh_dir, exist_ok=True)
mesh2.export(os.path.join(mesh_dir, obj_name+'.stl'))
create_tet(mesh_dir, obj_name)






