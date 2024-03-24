import numpy as np
# from sklearn.decomposition import PCA
import trimesh
from copy import deepcopy
import open3d

def down_sampling(pc, num_pts=1024, return_indices=False):
    # farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    # pc = pc[farthest_indices.squeeze()]  
    # return pc

    """
    Input:
        pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
        num_pts: number of samples
    Return:
        centroids: sampled pointcloud index, [num_pts, D]
        pc: down_sampled point cloud, [num_pts, D]
    """

    if pc.ndim == 2:
        # insert batch_size axis
        pc = deepcopy(pc)[None, ...]

    B, N, D = pc.shape
    xyz = pc[:, :,:3]
    centroids = np.zeros((B, num_pts))
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32)

    for i in range(num_pts):
        centroids[:, i] = farthest
        centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
        centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
        dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1) # (B,)

    pc = pc[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]

    if return_indices:
        return pc.squeeze(), centroids.astype(np.int32)

    return pc.squeeze()

def down_sampling_torch(pc, num_pts=1024, return_indices=False):
    """
    Input:
        pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
        num_pts: number of samples
    Return:
        centroids: sampled point cloud index, [num_pts, D]
        pc: down-sampled point cloud, [num_pts, D]
    """
    import torch
    if pc.ndim == 2:
        # Insert batch_size axis
        pc = pc.unsqueeze(0)

    B, N, D = pc.shape
    xyz = pc[:, :, :3]
    centroids = torch.zeros((B, num_pts), dtype=torch.long, device=pc.device)
    distance = torch.ones((B, N), dtype=pc.dtype, device=pc.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=pc.device)

    for i in range(num_pts):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(0, B), farthest, :]  # (B, D)
        centroid = centroid.unsqueeze(1)  # (B, 1, D)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)  # (B,)

    pc = pc[torch.arange(0, B).view(-1, 1), centroids, :]

    if return_indices:
        return pc.squeeze(), centroids

    return pc.squeeze()

def pcd_ize(pc, color=None, vis=False):
    """ 
    Convert point cloud numpy array to an open3d object (usually for visualization purpose).
    """
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd

def spherify_point_cloud_open3d(point_cloud, radius=0.002, color=None, vis=False):
    """
    Use Open3D to visualize a point cloud where each point is represented by a sphere.
    """
    """
    Visualize a point cloud where each point is represented by a sphere.
    
    Parameters:
    - point_cloud: NumPy array of shape (N, 3), representing the point cloud.
    - radius: float, the radius of each sphere used to represent a point.
    """
    # Create an empty list to hold the sphere meshes
    sphere_meshes = []
    
    # Iterate over the points in the point cloud
    for point in point_cloud:
        # Create a mesh sphere for the current point
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(tuple(point))  # Move the sphere to the point's location
        if color is not None:
            sphere.paint_uniform_color(color) 
        sphere_meshes.append(sphere)
    
    # Combine all spheres into one mesh
    combined_mesh = open3d.geometry.TriangleMesh()
    for sphere in sphere_meshes:
        combined_mesh += sphere
    if vis:
        open3d.visualization.draw_geometries([combined_mesh])
    return combined_mesh

def is_homogeneous_matrix(matrix):
    # Check matrix shape
    if matrix.shape != (4, 4):
        return False

    # Check last row
    if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
        return False

    # Check rotational part (3x3 upper-left submatrix)
    rotational_matrix = matrix[:3, :3]
    if not np.allclose(np.dot(rotational_matrix, rotational_matrix.T), np.eye(3), atol=1.e-6) or \
            not np.isclose(np.linalg.det(rotational_matrix), 1.0, atol=1.e-6):
        
        print(np.linalg.inv(rotational_matrix), "\n")
        print(rotational_matrix.T)        
        print(np.linalg.det(rotational_matrix))
        
        return False

    return True

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_min_ang_vec(world_vec, cam_vecs):
    min_ang = float('inf')
    min_ang_idx = -1
    min_ang_vec = None
    for i in range(cam_vecs.shape[1]):
        angle = angle_between(world_vec, cam_vecs[:, i])
        larger_half_pi = False
        if angle > np.pi * 0.5:
            angle = np.pi - angle
            larger_half_pi = True
        if angle < min_ang:
            min_ang = angle
            min_ang_idx = i
            if larger_half_pi:
                min_ang_vec = -cam_vecs[:, i]
            else:
                min_ang_vec = cam_vecs[:, i]

    return min_ang_vec, min_ang_idx

def world_to_object_frame(points):

    """  
    Compute 4x4 homogeneous transformation matrix to transform world frame to object frame. 
    The object frame is obtained by fitting a bounding box to the object partial-view point cloud.
    The centroid of the bbox is the the origin of the object frame.
    x, y, z axes are the orientation of the bbox.
    We then compare these computed axes against the ground-truth axes ([1,0,0], [0,1,0], [0,0,1]) and align them properly.
    For example, if the computed x-axis is [0.3,0.0,0.95], which is most similar to [0,0,1], this axis would be set to be the new z-axis.
    
    **This function is used to define a new frame for the object point cloud. Crucially, it creates the training data and defines the pc for test time.

    (Input) points: object partial-view point cloud. Shape (num_pts, 3)
    """

    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(points)

    # Compute the oriented bounding box (OBB) of the point cloud
    obb = point_cloud.bounding_box_oriented

    homo_mat = obb.primitive.transform
    axes = obb.primitive.transform[:3,:3]   # x, y, z axes concat together

    # Find and align z axis
    z_axis = [0., 0., 1.]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, axes)
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align x axis.
    x_axis = [1., 0., 0.]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, axes) 
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align y axis
    y_axis = [0., 1., 0.]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, axes) 

    R_o_w = np.column_stack((align_x_axis, align_y_axis, align_z_axis))
    
    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, homo_mat[:3,3])
    
    homo_mat[:3,:3] = R_w_o
    homo_mat[:3,3] = d_w_o_o

    assert is_homogeneous_matrix(homo_mat)

    return homo_mat


def transform_point_cloud(point_cloud, transformation_matrix):
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points



def transform_point_clouds(point_clouds, matrices):

    # If there's only one point cloud but multiple matrices, repeat and reshape point cloud to match matrices shape.
    if len(point_clouds.shape) == 2 and len(matrices.shape) == 3:
        num_matrices = matrices.shape[0]
        point_clouds = np.tile(point_clouds, (num_matrices, 1, 1))
    
    # If there's both only one point cloud and one matrix, add an extra dimension to both.    
    elif len(point_clouds.shape) == 2:
        point_clouds = point_clouds[np.newaxis, ...]
        matrices = matrices[np.newaxis, ...]
    
    # Convert 3D points to homogeneous coordinates
    homogeneous_points = np.concatenate((point_clouds, np.ones_like(point_clouds[..., :1])), axis=-1)
   
    # Perform matrix multiplication for all point clouds at once using broadcasting
    transformed_points = np.matmul(homogeneous_points, matrices.swapaxes(1, 2))
    
    return transformed_points[:,:,:3]


def random_transformation_matrix(translation_range=None, rotation_range=None):
    from scipy.spatial.transform import Rotation
    
    # Generate random translation vector
    if translation_range is None:
        translation = np.array([0,0,0])
    else:
        translation = np.random.uniform(translation_range[0], translation_range[1], size=3)
    
    if rotation_range is None:
        rotation_matrix = np.eye(3)
    else:
        # Generate random rotation angles
        rotation_angles = np.random.uniform(rotation_range[0], rotation_range[1], size=3)
        
        # Create rotation object
        rotation = Rotation.from_euler('xyz', rotation_angles, degrees=False)
        rotation_matrix = rotation.as_matrix()
        
    
    # Create 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix