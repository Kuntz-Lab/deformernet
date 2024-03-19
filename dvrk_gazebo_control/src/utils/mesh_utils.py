import numpy as np
import trimesh
import open3d


def sample_points_from_tet_mesh(mesh, k):

    """ 
    Sample points from the tetrahedral mesh by executing the following procedure:
    1) Sampling k points from each tetrahedron by computing the weighted average location of the 4 vertices with random weights.
    
    """

    def generate_random_weights(k):
        # Generate random weights for k points
        weights = np.random.rand(k, 4)
        # Normalize the weights so that they sum up to 1 for each point
        normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)
        return normalized_weights

    def compute_weighted_average(vertices, weights):
        # Compute the weighted average for each set of weights and vertices
        return np.einsum('ijk,ij->ik', vertices, weights)
    

    num_tetrahedra = mesh.shape[0]
    vertices = mesh.reshape(num_tetrahedra, 4, -1)

    points = []

    for _ in range(k):
        # Generate random weights for all tetrahedra and points at once
        weights_list = generate_random_weights(k=1)

        # Compute the weighted average location for all points and vertices
        sampled_points = compute_weighted_average(vertices, weights_list)
        
        points.append(sampled_points)
        
    sampled_points = np.concatenate((points), axis=0)

    return sampled_points


def create_tet_mesh(mesh_dir, intput_tri_mesh_name, output_tet_mesh_name=None, mesh_extension='.stl', 
                    coarsen=True, verbose=False, fTetWild_dir='/home/baothach/fTetWild/build'):
    
    if output_tet_mesh_name is None:
        output_tet_mesh_name = intput_tri_mesh_name
    
    # surface mesh (.stl, .obj, etc.) to volumetric mesh (.mesh)
    import os
    os.chdir(fTetWild_dir) 
    mesh_path = os.path.join(mesh_dir, intput_tri_mesh_name + mesh_extension)
    save_fTetwild_mesh_path = os.path.join(mesh_dir, output_tet_mesh_name + '.mesh')
    
    if coarsen:
        os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path + " --coarsen >/dev/null")
    else:
        os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path + " >/dev/null")

    # .mesh to .tet:
    mesh_file = open(os.path.join(mesh_dir, output_tet_mesh_name + '.mesh'), "r")
    tet_output = open(
        os.path.join(mesh_dir, output_tet_mesh_name + '.tet'), "w")

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

    if verbose:
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
        
        
def simplify_mesh_pymeshlab(mesh, target_num_vertices):
    
    import pymeshlab as ml
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Invalid mesh type. Must be a Trimesh object") 
    
	# https://stackoverflow.com/questions/65419221/how-to-use-pymeshlab-to-reduce-vertex-number-to-a-certain-number/65424578#65424578
    numFaces = 100 + 2*target_num_vertices

    ms = ml.MeshSet()
    pymeshlab_mesh = ml.Mesh(mesh.vertices, mesh.faces)
    ms.add_mesh(pymeshlab_mesh)    
  
    while (ms.current_mesh().vertex_number() > target_num_vertices):
        ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
        numFaces = numFaces - (ms.current_mesh().vertex_number() - target_num_vertices)
    
    return trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix()) 


def trimesh_to_open3d_mesh(trimesh_mesh):
    # Access vertices and faces of the trimesh mesh
    vertices = np.array(trimesh_mesh.vertices)
    faces = np.array(trimesh_mesh.faces)

    # Create an open3d TriangleMesh from the vertices and faces
    open3d_mesh = open3d.geometry.TriangleMesh()
    open3d_mesh.vertices = open3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = open3d.utility.Vector3iVector(faces)

    return open3d_mesh

def open3d_to_trimesh_mesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return trimesh_mesh


def apply_euler_rotation_trimesh(mesh, roll, pitch, yaw, degrees=False):
    """
    Apply a rotation transformation to a trimesh object based on given Euler angles.
    
    Parameters:
    - mesh: trimesh.Trimesh object
    - roll: Rotation around the X-axis in degrees or radians
    - pitch: Rotation around the Y-axis in degrees or radians
    - yaw: Rotation around the Z-axis in degrees or radians
    - degrees: If True, interpret the angles as degrees; otherwise as radians
    
    Returns:
    - mesh: Transformed trimesh.Trimesh object with the applied rotation
    """
    
    # Convert degrees to radians if necessary
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    # Create rotation matrices for each axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combine the rotations in ZYX order
    R = R_z @ (R_y @ R_x)  # Note: @ operator is used for matrix multiplication

    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R

    # Apply the transformation to the mesh
    mesh.apply_transform(transformation_matrix)
    
    return mesh