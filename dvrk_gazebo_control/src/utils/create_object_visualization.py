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

# mesh_dir = "/home/baothach/shape_servo_data/rotation_extension/visualization/display_objects/mesh"
# # object_name = "long_box"

# mesh1 = trimesh.creation.box((0.15, 0.15, 0.04))



# meshes = [mesh1]
# trimesh.Scene(meshes).show()
# mesh1.export(os.path.join(mesh_dir, object_name+'.stl'))

# create_tet(mesh_dir, object_name)





def create_cylinder_mesh_datatset(save_mesh_dir):


    # radius = np.random.uniform(low = 0.03, high = 0.06)
    # height = np.random.uniform(low = 0.23, high = 0.40)
    
    radius =  0.06  #0.06  #0.05  
    height =  0.18  #0.18  #0.40  
    
    youngs = 1000
    
    mesh = trimesh.creation.cylinder(radius=radius, height=height)
    shape_name = "cylinder"        
    object_name = shape_name + "_" + str(0)
    mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
    create_tet(save_mesh_dir, object_name)
    
    primitive_dict = {'radius': radius, 'height': height, 'youngs': youngs}
    

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, object_name + ".pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_box_mesh_datatset(save_mesh_dir):

    # ratio = np.random.uniform(low = 1, high = 3.5)
    # if ratio <= 1.5:
    #     width = np.random.uniform(low = 0.15, high = 0.2)
    # else:
    #     width = np.random.uniform(low = 0.1, high = 0.2)
    # height = min(0.4, width * ratio)
    # thickness = np.random.uniform(low = 0.04, high = 0.06)
    
    ratio = 3#1
    width = 0.15#0.2
    height = width * ratio#min(0.4, width * ratio)
    thickness = 0.06

    mesh = trimesh.creation.box((height, width, thickness))


    youngs = 1000

    shape_name = "box"        
    object_name = shape_name + "_" + str(1)
    mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
    create_tet(save_mesh_dir, object_name)
    
    primitive_dict = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}


    data = primitive_dict
    with open(os.path.join(save_mesh_dir, object_name + ".pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_hemis_mesh_datatset(save_mesh_dir):


    # radius = np.random.uniform(low = 0.2, high = 0.3)
    # origin = radius/0.2 * 0.1 
    # ratio = np.random.uniform(low = 1.5, high = 4)


    radius = 0.3    #0.2
    ratio = 4   #1
    origin = radius/0.2 * 0.1
    

    youngs = 1000



    mesh = trimesh.creation.icosphere(radius = radius)   # hemisphere
    mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0,origin], cap=True)

    vertices_transformed = mesh.vertices * np.array([1./ratio,1,1])
    mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh.faces)
                    

    shape_name = "hemis"        
    object_name = shape_name + "_" + str(1)
    mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
    create_tet(save_mesh_dir, object_name)

    primitive_dict = {'radius': radius, 'origin': origin, 'youngs': youngs}

    


    data = primitive_dict
    with open(os.path.join(save_mesh_dir, object_name + ".pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


## 1000-200, 5000-1000, 10000-1000

# mesh_dir = "/home/baothach/shape_servo_data/rotation_extension/visualization/display_objects/mesh"
mesh_dir = "/home/baothach/shape_servo_data/dynamics/mesh"
# create_box_mesh_datatset(mesh_dir)


create_cylinder_mesh_datatset(mesh_dir)


# create_hemis_mesh_datatset(mesh_dir)




