import numpy as np
import transformations


def get_random_tool_pose(area_idx: int, translation_vector: np.ndarray, 
                         return_format: str ="quaternion"):

    min_tilt, max_tilt = 0, np.pi/4
    if area_idx == 0:
        min_yaw, max_yaw = -np.pi/2 + np.pi/6, -np.pi/2 + np.pi/3
    elif area_idx == 1:
        min_yaw, max_yaw = -np.pi/12, np.pi/12
    elif area_idx == 2:
        min_yaw, max_yaw = np.pi/2 - np.pi/3, np.pi/2 - np.pi/6

    # min_tilt, max_tilt = 1*np.pi/3, 1*np.pi/3
    # min_yaw, max_yaw = 0, 0 

    min_rotation = [0, min_yaw, np.pi + min_tilt]
    max_rotation = [0, max_yaw, np.pi + max_tilt]

    T = np.eye(4)
    angles = [np.random.uniform(min_rotation[i], max_rotation[i]) for i in range(3)]
    # angles = [0, np.pi/8, np.pi + np.pi/8]  #  [x, -np.pi/3, 0]   # rool, yaw, pitch
    T[:3, :3] = transformations.euler_matrix(*angles)[:3, :3]
    T[:3, 3] = translation_vector
    if return_format == "matrix":
        return T    # shape (4, 4)
    elif return_format == "quaternion":
        quat = transformations.quaternion_from_matrix(T)
        return np.concatenate([translation_vector, quat])


def get_random_tool_pose_old(quadrant: int, translation: int, 
                         min_rotation: np.ndarray = [-np.pi/4, -np.pi/4, -np.pi/4], max_rotation: np.ndarray = [np.pi/4, np.pi/4, np.pi/4], 
                         return_format: str ="quaternion"):
    if quadrant == 1:
        translation_vector = np.array([translation, translation, translation])
    elif quadrant == 2:
        translation_vector = np.array([translation, -translation, translation])
    elif quadrant == 3:
        translation_vector = np.array([-translation, translation, translation])
    elif quadrant == 4:
        translation_vector = np.array([-translation, -translation, translation])

    T = np.eye(4)
    # angles = [np.random.uniform(min_rotation[i], max_rotation[i]) for i in range(3)]
    # angles = [0, np.pi/8, np.pi + np.pi/8]  #  [x, -np.pi/3, 0]   # rool, yaw, pitch
    angles = [0, 0, np.pi + np.pi/4] 
    T[:3, :3] = transformations.euler_matrix(*angles)[:3, :3]
    T[:3, 3] = translation_vector
    if return_format == "matrix":
        return T    # shape (4, 4)
    elif return_format == "quaternion":
        quat = transformations.quaternion_from_matrix(T)
        return np.concatenate([translation_vector, quat])  # shape (7,) [x, y, z, w, x, y, z]
    

def sample_cylindrical_point(r_min, r_max, theta_min, theta_max, z_min, z_max):
    r = np.random.uniform(r_min, r_max)
    theta = np.random.uniform(theta_min, theta_max)
    z = np.random.uniform(z_min, z_max)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    
    return np.array([x, y, z])


def random_boolean_mask(shape=(4,), max_ones=3):
    """Generate a random mask with specified shape, and at most `max_ones` set to 1 (0 elsewhere)."""
    mask = np.zeros(shape, dtype=int)
    # mask[:np.random.randint(0, max_ones + 1)] = 1
    mask[:2] = 1
    np.random.shuffle(mask)
    return mask   
    