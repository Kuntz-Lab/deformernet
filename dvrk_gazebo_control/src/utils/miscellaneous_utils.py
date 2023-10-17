# import open3d
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle5 as pickle
import os


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
    import open3d
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd


def get_object_particle_state(gym, sim, vis=False):
    from isaacgym import gymtorch
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    particles = particle_state_tensor.numpy()[:, :3]  
    
    if vis:
        pcd_ize(particles, vis=True)
    
    return particles.astype('float32')


# def record_data_stress_prediction(gym, sim, env, \
#                                 undeformed_object_pc, undeformed_object_particle_state, undeformed_gripper_pc, current_desired_force, \
#                                 object_name, object_young_modulus, object_scale, \
#                                 cam_handle, cam_prop, robot_segmentationId=11, min_z=0.01, vis=False):
#     ### Get current object pc and particle position:
#     object_pc = get_partial_pointcloud_vectorized(gym, sim, env, cam_handle, cam_prop, robot_segmentationId, "deformable", None, min_z, vis, device="cpu")
#     object_particle_position = get_object_particle_state(gym, sim)

# def record_data_stress_prediction(data_recording_path, gym, sim, 
#                                 current_force, grasp_pose, fingers_joint_angles, 
#                                 object_name, young_modulus, object_scale):
                                    
#     ### Get current object particle state:
#     object_particle_state = get_object_particle_state(gym, sim)

#     (tet_indices, tet_stress) = gym.get_sim_tetrahedra(sim)

       
#     data = {"object_particle_state": object_particle_state, "force": current_force, 
#         "grasp_pose": grasp_pose, "fingers_joint_angles": fingers_joint_angles, 
#         "tet_stress": tet_stress, 
#         "object_name": object_name, "young_modulus": young_modulus, "object_scale": object_scale}    
    
#     with open(data_recording_path, 'wb') as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def record_data_stress_prediction(data_recording_path, gym, sim, 
                                current_force, grasp_pose, fingers_joint_angles, force_fingers_joint_angles, 
                                object_name, young_modulus, object_scale):
    
    """
    Record data to pickle files.
    fingers_joint_angles: gripper's joint angles RIGHT AFTER making contact with the object (not applying force yet). Shape (2,)
    force_fingers_joint_angles: gripper's joint angles when gripper is APPLYING FORCE to the object. Shape (2,)
    
    """
                                    
    ### Get current object particle state:
    object_particle_state = get_object_particle_state(gym, sim)

    (tet_indices, tet_stress) = gym.get_sim_tetrahedra(sim)    # tet_stress: shape (num_tetrahedra,)

    all_cauchy_stresses = []
    for cauchy_stress in tet_stress:
        cauchy_stress_matrix = np.array([[cauchy_stress.x.x, cauchy_stress.y.x, cauchy_stress.z.x],
                                        [cauchy_stress.x.y, cauchy_stress.y.y, cauchy_stress.z.y],
                                        [cauchy_stress.x.z, cauchy_stress.y.z, cauchy_stress.z.z]])
        all_cauchy_stresses.append(cauchy_stress_matrix) 
    all_cauchy_stresses = np.array(all_cauchy_stresses)    # shape (num_tetrahedra, 3, 3)

       
    data = {"object_particle_state": object_particle_state, "force": current_force, 
        "grasp_pose": grasp_pose, "fingers_joint_angles": fingers_joint_angles, 
        "force_fingers_joint_angles": force_fingers_joint_angles,
        "tet_stress": all_cauchy_stresses, 
        "object_name": object_name, "young_modulus": young_modulus, "object_scale": object_scale}    
    
    with open(data_recording_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def normalize_list(lst):
    minimum = min(lst)
    maximum = max(lst)
    value_range = maximum - minimum

    normalized_lst = [(value - minimum) / value_range for value in lst]

    return normalized_lst


def scalar_to_rgb(scalar_list, colormap='jet', min_val=None, max_val=None):
    if min_val is None:
        norm = plt.Normalize(vmin=np.min(scalar_list), vmax=np.max(scalar_list))
    else:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.cm.get_cmap(colormap)
    rgb = cmap(norm(scalar_list))
    return rgb


def print_color(text, color="red"):

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    if color == "red":
        print(RED + text + RESET)
    elif color == "green":
        print(GREEN + text + RESET)
    elif color == "yellow":
        print(YELLOW + text + RESET)
    elif color == "blue":
        print(BLUE + text + RESET)
    else:
        print(text)


def read_pickle_data(data_path):
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)      

def write_pickle_data(data, data_path):
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=3)    
        

def read_youngs_value_from_urdf(urdf_file):
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    youngs_elem = root.find('.//fem/youngs')
    if youngs_elem is not None and 'value' in youngs_elem.attrib:
        return str(float(youngs_elem.attrib['value']))

    return None

# def find_folder_directory(folder_name):
#     """
#     Recursively searches for the absolute path of a folder (given its name), 
#     in the current working directory and its parent directories.

#     Parameters:
#         folder_name (str): The name of the folder to search for.

#     Returns:
#         str or None: The absolute path to the folder if found, else None.
#     """
    
#     current_dir = os.getcwd()   # absolute path of the current working directory
#     while current_dir != "/" and os.path.basename(current_dir) != folder_name:
#         current_dir = os.path.dirname(current_dir)
#     if os.path.basename(current_dir) == folder_name:
#         return current_dir
#     else:
#         return None  # Folder not found


def get_extents_object(tet_file):
    """Return [min_x, min_y, min_z], [max_x, max_y, max_z] for a tet mesh"""
    mesh_lines = list(open(tet_file, "r"))
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    zs = []
    particles = []
    for ml in mesh_lines:
        sp = ml.split(" ")
        if sp[0] == 'v':
            particles.append([float(sp[j]) for j in range(1,4)])
                
    particles = np.array(particles)
    xs = particles[:,0]
    ys = particles[:,1]
    zs = particles[:,2]
    
    return [[min(xs), min(ys), min(zs)],\
            [max(xs), max(ys), max(zs)]]   