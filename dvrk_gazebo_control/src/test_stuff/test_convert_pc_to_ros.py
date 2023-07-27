#!/usr/bin/env python3
# import tf
# import tf.transformations
# import rospy
from gazebo_msgs.msg import ModelStates
import roslib.packages as rp
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import open3d_ros_helper as orh
import open3d
import numpy as np
import rospy

rospy.init_node('test_stuff')
obb = open3d.geometry.OrientedBoundingBox()
pcd = open3d.geometry.PointCloud()
np_points = np.random.rand(100, 3)*100
# print(np_points)

# From numpy to Open3D
pcd.points = open3d.utility.Vector3dVector(np_points)
rospc = orh.o3dpc_to_rospc(pcd)
print(rospc)

# print("bao")
# rospy.init_node('gen_grasp_preshape_server')
# listener = tf.TransformListener()

# homo_matrix_world_frame = listener.fromTranslationRotation(
#     (.0, .0, .4),
#     (0,0,0,1)
# )
# print(homo_matrix_world_frame)