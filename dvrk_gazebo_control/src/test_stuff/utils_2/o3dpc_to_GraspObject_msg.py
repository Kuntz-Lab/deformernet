#!/usr/bin/env python3
import open3d
import numpy as np
import open3d_ros_helper as orh
from dvrk_gazebo_control.msg import GraspObject
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation as R
import rospy
import copy





def o3dpc_to_GraspObject_msg(pcd):
    ros_cloud = orh.o3dpc_to_rospc(pcd)
    pcd.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                            max_nn=8))
    # Get normal:
    normals = open3d.geometry.PointCloud()
    normals.points = pcd.normals
    ros_normals = orh.o3dpc_to_rospc(normals)
    
    # Get pose of the obb
    # Position
    obb = pcd.get_oriented_bounding_box()
    pose = Pose()
    center = obb.get_center()
    pose.position.x = center[0]
    pose.position.y = center[1]
    pose.position.z = center[2]

    # Orientation
    orientation = copy.deepcopy(obb.R)
    if (round(np.linalg.det(orientation)) != 1):    # input matrices are not special orthogonal(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_matrix.html)
        for i in range(3): 
            orientation[i][0] = -orientation[i][0]  # reverse x axis
    orientation_rotmat = R.from_matrix(orientation)
    orientation_quat =  orientation_rotmat.as_quat()
    pose.orientation.x =  orientation_quat[0]
    pose.orientation.y =  orientation_quat[1]
    pose.orientation.z =  orientation_quat[2]
    pose.orientation.w =  orientation_quat[3] 

    # Get width, height, depth
    points = np.asarray(obb.get_box_points())
    x_axis = (points[1]-points[0])
    y_axis = (points[2]-points[0])
    z_axis = (points[3]-points[0])
    width = np.linalg.norm(x_axis)  # Length of x axis (https://www.cs.utah.edu/gdc/projects/alpha1/help/man/html/shape_edit/primitives.html)
    height = np.linalg.norm(y_axis)
    depth = np.linalg.norm(z_axis)

    # Return GraspObject msg (for preshape generation)
    msg = GraspObject()
    msg.pose = pose
    msg.width = width
    msg.height = height
    msg.depth = depth
    msg.cloud = ros_cloud
    msg.normals = ros_normals
    return msg

