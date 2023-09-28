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


def visualize_voxel(voxel_grid):
    import matplotlib.pyplot as plt
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of occupied voxels
    x, y, z = np.where(voxel_grid)

    # Plot the occupied voxels as points
    ax.scatter(x, y, z, c='b', marker='o', s=10)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def voxelize_point_cloud(point_cloud, grid_size=[32]*3, vis=False):
    # max_point = 2*np.max(np.abs(point_cloud))
    # grid_size = int(np.ceil(2 * max_point / voxel_size)) + 1  # Calculate grid size and add 1

    # # Initialize voxel grid
    # voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    # # Iterate over points and fill the voxel grid
    # for point in point_cloud:
    #     voxel_coordinates = np.floor((point + max_point) / voxel_size).astype(int)
    #     if (0 <= voxel_coordinates).all() and (voxel_coordinates < grid_size).all():
    #         voxel_grid[voxel_coordinates[0], voxel_coordinates[1], voxel_coordinates[2]] = True

    from pyntcloud import PyntCloud

    n_x, n_y, n_z = grid_size[0], grid_size[1], grid_size[2]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud)   
    cloud = PyntCloud.from_instance('OPEN3D', pcd)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n_x, n_y=n_x, n_z=n_y)
    # voxelgrid_id = cloud.add_structure("voxelgrid", size_x=grid_size[0], size_y=grid_size[1], size_z=grid_size[2])
    voxelgrid = cloud.structures[voxelgrid_id]


    voxel_array = np.zeros((n_x, n_y, n_z))

    for (x,y,z) in zip(voxelgrid.voxel_x, voxelgrid.voxel_y, voxelgrid.voxel_z):
        voxel_array[x,y,z] = 1

    if vis:
        visualize_voxel(voxel_array)

    return voxel_array

def point_cloud_to_mesh_mcubes(point_cloud, grid_size, use_mcubes_smooth=False, vis_voxel=False, vis_mesh=False):
    import mcubes

    voxel_grid = voxelize_point_cloud(point_cloud, grid_size, vis=vis_voxel)
    
    if use_mcubes_smooth:
        voxel_grid = mcubes.smooth(voxel_grid)
    
    # Extract the 0-isosurface
    vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    if vis_mesh:
        mesh.show()

    return mesh

def is_occupied(x, y, z, pre_voxelized):
    '''
    x,y,z is bottom left.
    '''

    # Check if all are on.
    return (pre_voxelized[x,y,z] == 1.0 and
            pre_voxelized[x,y,z+1] == 1.0 and
            pre_voxelized[x,y+1,z] == 1.0 and
            pre_voxelized[x,y+1,z+1] == 1.0 and 
            pre_voxelized[x+1,y,z] == 1.0 and
            pre_voxelized[x+1,y,z+1] == 1.0 and
            pre_voxelized[x+1,y+1,z] == 1.0 and
            pre_voxelized[x+1,y+1,z+1] == 1.0)

def is_active(x, y, z, pre_voxelized):
    '''
    Check if given x,y,z in given voxel grid is active.
    '''
    voxel_occupancy = pre_voxelized[x,y,z]

    base = pre_voxelized[x,y,z]
    return (pre_voxelized[x,y,z+1] != base or
            pre_voxelized[x,y+1,z] != base or
            pre_voxelized[x,y+1,z+1] != base or 
            pre_voxelized[x+1,y,z] != base or
            pre_voxelized[x+1,y,z+1] != base or
            pre_voxelized[x+1,y+1,z] != base or
            pre_voxelized[x+1,y+1,z+1] != base)

def get_grid_points(active_voxels, current_voxel_resolution, bound):
    grid_pts = set()
    voxel_size = (2*bound) / float(current_voxel_resolution)
    for x,y,z in active_voxels:
        x_ = -bound + (((2*bound) / float(current_voxel_resolution)) * x)
        y_ = -bound + (((2*bound) / float(current_voxel_resolution)) * y)
        z_ = -bound + (((2*bound) / float(current_voxel_resolution)) * z)
        grid_pts.add((x_,y_,z_))
        grid_pts.add((x_,y_,z_+voxel_size))
        grid_pts.add((x_,y_+voxel_size,z_))
        grid_pts.add((x_,y_+voxel_size,z_+voxel_size))
        grid_pts.add((x_+voxel_size,y_,z_))
        grid_pts.add((x_+voxel_size,y_,z_+voxel_size))
        grid_pts.add((x_+voxel_size,y_+voxel_size,z_))
        grid_pts.add((x_+voxel_size,y_+voxel_size,z_+voxel_size))
    return np.array(list(grid_pts))

def mise_voxel(point_cloud, initial_voxel_resolution, final_voxel_resolution, voxel_size, centroid_diff=None):
    import mcubes
    '''
    get_sdf: map from query points to SDF (assume everything else already embedded in func (i.e., point cloud/embedding)).
    bound: sample within [-bound, bound] in x,y,z.
    initial/final_voxel_resolution: powers of two representing voxel resolution to evaluate at.
    voxel_size: size of each voxel (in final res) determined by view.
    centroid_diff: offset if needed.
    '''

    centered_pc = point_cloud - np.mean(point_cloud, axis=0)
    bounds = trimesh.PointCloud(centered_pc).bounds
    bound = max(bounds[1])

    # Active voxels: voxels we want to evaluate grid points of.
    active_voxels = []
    # Full voxelization.
    voxelized = np.zeros((final_voxel_resolution,final_voxel_resolution,final_voxel_resolution), dtype=np.float32)
    # Intermediate voxelization. This represents the grid points for the voxels (so is resolution + 1 in each dim).
    partial_voxelized = None

    # Init active voxels to all voxels in the initial resolution.
    for x in range(initial_voxel_resolution):
        for y in range(initial_voxel_resolution):
            for z in range(initial_voxel_resolution):
                active_voxels.append([x, y, z])
    active_voxels = np.array(active_voxels)

    # Start main loop that ups resolution.
    current_voxel_resolution = initial_voxel_resolution
    while current_voxel_resolution <= final_voxel_resolution:
        # print(current_voxel_resolution)

        # Setup voxelizations at this dimension.
        partial_voxelized = np.zeros((current_voxel_resolution + 1,current_voxel_resolution + 1,current_voxel_resolution + 1), dtype=np.float32)


        # For all points sample SDF given the point cloud.
        for pt_ in centered_pc:
            # Convert points into grid voxels and set.
            x_ = int(round(((pt_[0] + bound)/(2 * bound)) * float(current_voxel_resolution)))
            y_ = int(round(((pt_[1] + bound)/(2 * bound)) * float(current_voxel_resolution)))
            z_ = int(round(((pt_[2] + bound)/(2 * bound)) * float(current_voxel_resolution)))
            partial_voxelized[x_,y_,z_] = 1.0


        # Determine filled and active voxels.
        new_active_voxels = []
        for x,y,z in active_voxels:
            if is_occupied(x, y, z, partial_voxelized):
                # Set all associated voxels on in full voxelization.
                voxels_per_voxel = final_voxel_resolution // current_voxel_resolution

                # Set all corresponding voxels in the full resolution to on.
                for x_ in range(voxels_per_voxel*x, voxels_per_voxel*x + voxels_per_voxel):
                    for y_ in range(voxels_per_voxel*y, voxels_per_voxel*y + voxels_per_voxel):
                        for z_ in range(voxels_per_voxel*z, voxels_per_voxel*z + voxels_per_voxel):
                            voxelized[x_, y_, z_] = 1.0
            elif is_active(x, y, z, partial_voxelized):
                # If final resolution, just set it as active.
                if current_voxel_resolution == final_voxel_resolution:
                    voxelized[x,y,z] = 1.0
                    continue
                
                # Up voxel position to match doubling of voxel resolution.
                x_base = 2*x
                y_base = 2*y
                z_base = 2*z

                # Add new voxels for higher resolution. Each voxel gets split into 8 new.
                new_active_voxels.append([x_base, y_base, z_base])
                new_active_voxels.append([x_base, y_base, z_base+1])
                new_active_voxels.append([x_base, y_base+1, z_base])
                new_active_voxels.append([x_base, y_base+1, z_base+1])
                new_active_voxels.append([x_base+1, y_base, z_base])
                new_active_voxels.append([x_base+1, y_base, z_base+1])
                new_active_voxels.append([x_base+1, y_base+1, z_base])
                new_active_voxels.append([x_base+1, y_base+1, z_base+1])                
        active_voxels = np.array(new_active_voxels)
        current_voxel_resolution = current_voxel_resolution * 2

    # print("Done with extraction.")

    # Padding to prevent holes if go up to edge.
    voxels = voxelized
    voxelized = np.pad(voxelized, ((1,1),(1,1),(1,1)), mode='constant')
    
    # Mesh w/ mcubes.
    voxelized = mcubes.smooth(voxelized)
    vertices, triangles = mcubes.marching_cubes(voxelized, 0)
    vertices = vertices * voxel_size

    # Center mesh.
    vertices[:,0] -= voxel_size * (((final_voxel_resolution) / 2) + 1)
    vertices[:,1] -= voxel_size * (((final_voxel_resolution) / 2) + 1)
    vertices[:,2] -= voxel_size * (((final_voxel_resolution) / 2) + 1)

    # vertices[:,0] -= centroid_diff[0]
    # vertices[:,1] -= centroid_diff[1]
    # vertices[:,2] -= centroid_diff[2]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh # convert_to_sparse_voxel_grid(voxels, threshold=0.5)

def generate_pcd_from_depth(depth, kuf, kvf):
    '''
    Given a depth image and camera intrinsics, convert to a point cloud
    and save as PCD file.
    '''
    # Convert depth to a point cloud and store as PCD.
    height = depth.shape[0]
    width = depth.shape[1]

    # Go through each nonzero and convert to a 3D point using camera matrix.
    mask = np.where(depth > 0)
    x = mask[1]
    y = mask[0]
    norm_x = (x.astype(np.float32)-(width*0.5)) / (width*0.5)
    norm_y = (y.astype(np.float32)-(height*0.5)) / (height*0.5)
    world_x = norm_x * depth[y,x] / kuf
    world_y = norm_y * depth[y,x] / kvf
    world_z = -depth[y,x]
    points = np.vstack((world_x, -world_y, world_z)).T # Negative on y because of image matrix layout.
    return points


def generate_point_clouds_from_mesh(object_mesh, object_name, num_poses_per_obj, 
                                    noise_sigma=0.001):
                          
                          
    '''
    Generate point clouds from object triangular mesh.
    '''
    
    import pyrender
    

    # Setup the scene.
    scene = pyrender.Scene()

    # Camera/Light
    cam = pyrender.PerspectiveCamera(yfov=0.820305, aspectRatio=(4.0/3.0))
    # Get some camera instrinsics to do depth to point cloud conversion.
    kuf = cam.get_projection_matrix()[0][0]
    kvf = cam.get_projection_matrix()[1][1]
    camera_pose = np.eye(4)
    scene.add(cam, pose=camera_pose)
    light_pose = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 10.],
        [0., 0., 0., 1.]
        ]
    scene.add(pyrender.DirectionalLight(), pose=light_pose)

    # Setup renderer.
    r = pyrender.OffscreenRenderer(viewport_width=1640, viewport_height=1480, point_size=1.0)

        
    m = pyrender.Mesh.from_trimesh(object_mesh)
    mesh_pose = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., -1.5],
        [0., 0., 0., 1.]
        ]
    mesh_node = pyrender.Node(mesh=m, matrix=mesh_pose)
    scene.add_node(mesh_node)

    # Do a bunch of orientations for each object.
    all_generated_point_clouds = []
    for rot_num in range(num_poses_per_obj):
        object_pose_name = object_name + "_" + str(rot_num)


        # Generate a random transformation.
        random_pose = trimesh.transformations.random_rotation_matrix()
        random_pose[2][3] = -1.5 # Move to spot we want.


        # Update the pose.
        scene.set_pose(mesh_node, pose=random_pose)

        # Render.
        color, depth = r.render(scene)
        
        import matplotlib.pyplot as plt
        # print(color)
        # plt.imshow(color)
        # plt.show()

        # Add noise to depth image.
        noise = np.random.normal(0.0, noise_sigma, depth.shape)
        nonzeros = np.nonzero(depth)
        depth[nonzeros] = depth[nonzeros] + noise[nonzeros]

        # Save PCD.
        points = generate_pcd_from_depth(depth, kuf, kvf)
        all_generated_point_clouds.append(points)

    # Remove object.
    scene.remove_node(mesh_node)
    # Cleanup
    r.delete()
    
    return all_generated_point_clouds