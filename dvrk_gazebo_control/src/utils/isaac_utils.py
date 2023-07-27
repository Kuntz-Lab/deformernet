from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
from dvrk_gazebo_control.srv import *
import rospy

def isaac_format_pose_to_PoseStamped(body_states):
    ros_pose = PoseStamped()
    ros_pose.header.frame_id = 'world'
    ros_pose.pose.position.x = body_states["pose"]["p"]["x"]
    ros_pose.pose.position.y = body_states["pose"]["p"]["y"]
    ros_pose.pose.position.z = body_states["pose"]["p"]["z"]
    ros_pose.pose.orientation.x = body_states["pose"]["r"]["x"]
    ros_pose.pose.orientation.y = body_states["pose"]["r"]["y"]
    ros_pose.pose.orientation.z = body_states["pose"]["r"]["z"]
    ros_pose.pose.orientation.w = body_states["pose"]["r"]["w"]
    return ros_pose


def fix_object_frame(object_world):
    object_world_fixed = copy.deepcopy(object_world)
    object_size = [object_world.width, object_world.height, object_world.depth]
    quaternion =  [object_world_fixed.pose.orientation.x,object_world_fixed.pose.orientation.y,\
                                            object_world_fixed.pose.orientation.z,object_world_fixed.pose.orientation.w]  
    r = R.from_quat(quaternion)
    rot_mat = r.as_matrix()
    # print("**Before:", rot_mat)
    # x_axis = rot_mat[:3, 0]
    max_x_indices = []
    max_y_indices = []
    max_z_indices = []
    for i in range(3):
        column = [abs(value) for value in rot_mat[:, i]]
       
        if column.index(max(column)) == 0:
            max_x_indices.append(i)
        elif column.index(max(column)) == 1:
            max_y_indices.append(i)
        elif column.index(max(column)) == 2:
            max_z_indices.append(i)
    
    # print("indices: ", max_x_indices, max_y_indices, max_z_indices)
    if (not max_x_indices):
        z_values = [abs(z) for z in rot_mat[2, max_z_indices]]  
        max_z_idx = max_z_indices[z_values.index(max(z_values))]
        max_x_idx = max_z_indices[z_values.index(min(z_values))]
        max_y_idx = max_y_indices[0]
    elif (not max_y_indices):
        z_values = [abs(z) for z in rot_mat[2, max_z_indices]]  
        max_z_idx = max_z_indices[z_values.index(max(z_values))]
        max_y_idx = max_z_indices[z_values.index(min(z_values))]
        max_x_idx = max_x_indices[0]
    else:
        max_x_idx = max_x_indices[0]
        max_y_idx = max_y_indices[0]
        max_z_idx = max_z_indices[0]
        
    # print("indices:", max_x_idx, max_y_idx, max_z_idx)
    # x_values = [abs(x) for x in rot_mat[0, :]]
    # y_values = [abs(y) for y in rot_mat[1, :]]
    # z_values = [abs(z) for z in rot_mat[2, :]]
    # max_x_idx = x_values.index(max(x_values))
    # max_y_idx = y_values.index(max(y_values))
    # max_z_idx = z_values.index(max(z_values))

    fixed_x_axis = rot_mat[:, max_x_idx]
    fixed_y_axis = rot_mat[:, max_y_idx]
    fixed_z_axis = rot_mat[:, max_z_idx]
   
    fixed_rot_matrix = np.column_stack((fixed_x_axis, fixed_y_axis, fixed_z_axis))

    # if (round(np.linalg.det(fixed_rot_matrix)) != 1):    # input matrices are not special orthogonal(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_matrix.html)
    #     for i in range(3): 
    #         fixed_rot_matrix[i][0] = -fixed_rot_matrix[i][0]  # reverse x axis
    for i in range(3):
        if fixed_rot_matrix[i][i] <= 0:
            fixed_rot_matrix[:,i] = -fixed_rot_matrix[:,i]



    r = R.from_matrix(fixed_rot_matrix)
    fixed_quat = r.as_quat()
    # print("**After:", fixed_rot_matrix)
    object_world_fixed.pose.orientation.x, object_world_fixed.pose.orientation.y, \
            object_world_fixed.pose.orientation.z, object_world_fixed.pose.orientation.w = fixed_quat
 

    r = R.from_quat(fixed_quat)
    # print("matrix: ", r.as_matrix())

    
    object_world_fixed.width = object_size[max_x_idx]
    object_world_fixed.height = object_size[max_y_idx]
    object_world_fixed.depth = object_size[max_z_idx]

    return object_world_fixed

def get_pykdl_client(q_cur):
    '''
    get Jacobian matrix
    '''
    # rospy.loginfo('Waiting for service get_pykdl.')
    # rospy.wait_for_service('get_pykdl')
    # rospy.loginfo('Calling service get_pykdl.')
    try:
        pykdl_proxy = rospy.ServiceProxy('get_pykdl', PyKDL)
        pykdl_request = PyKDLRequest()
        pykdl_request.q_cur = q_cur
        pykdl_response = pykdl_proxy(pykdl_request) 
    
    except(rospy.ServiceException, e):
        rospy.loginfo('Service get_pykdl call failed: %s'%e)
    # rospy.loginfo('Service get_pykdl is executed.')    
    # print("np.reshape(pykdl_response.ee_pose_flattened, (4,4)", np.reshape(pykdl_response.ee_pose_flattened, (4,4)).shape)
    return np.reshape(pykdl_response.jacobian_flattened, tuple(pykdl_response.jacobian_shape)), np.reshape(pykdl_response.ee_pose_flattened, (4,4))

def isaac_format_pose_to_list(body_states):
    return [body_states["pose"]["p"]["x"], body_states["pose"]["p"]["y"], body_states["pose"]["p"]["z"],
        body_states["pose"]["r"]["x"], body_states["pose"]["r"]["y"], body_states["pose"]["r"]["z"], body_states["pose"]["r"]["w"]]

def pad_rot_mat(rot_mat):
    """convert 3x3 to 4x4 rotation matrix for transformations package"""   
    new_mat = np.eye(4)  
    new_mat[:3,:3] = rot_mat  
    return new_mat 