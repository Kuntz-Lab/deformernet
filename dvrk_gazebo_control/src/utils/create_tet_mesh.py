import numpy as np
import trimesh
import os

# mesh_dir = '/home/baothach/sim_data/Custom/Custom_mesh/hemis'
mesh_dir = '/home/baothach/Downloads/'
object_name = "tissue"
# #square_rod, round_rod, cylinder, hemis, thin_tisue_layer
# # mesh = trimesh.creation.box((0.2, 0.15, 0.015))   # thin layer official
# # mesh = trimesh.creation.box((0.2, 0.15, 0.018))
# # mesh = trimesh.creation.cylinder(radius=0.06, height=0.23)    # cylinder
# # mesh = trimesh.creation.cylinder(radius=0.03, height=0.40)
# # mesh = trimesh.creation.box((0.40, 0.04, 0.04))

mesh = trimesh.creation.box((2, 1.5, 0.15))   # thin layer for wrapping


# mesh = trimesh.creation.icosphere(radius = 0.2)
# mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0,0.1], cap=True)
# trimesh.Scene(mesh).show()
mesh.export(os.path.join(mesh_dir, object_name+'.stl'))


create_tet = True
if create_tet:
    # STL to mesh
    import os
    os.chdir('/home/baothach/fTetWild/build') 
    mesh_path = os.path.join(mesh_dir, object_name+'.stl')
    save_fTetwild_mesh_path = os.path.join(mesh_dir, 'simplified_mesh.mesh')
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path)


    # Mesh to tet:
    mesh_file = open(os.path.join(mesh_dir, 'simplified_mesh.mesh'), "r")
    tet_output = open(
        os.path.join(mesh_dir, 'simplified_mesh.tet'), "w")

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
