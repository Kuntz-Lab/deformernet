#!/usr/bin/env python3
import numpy as np
import pickle
import open3d
from dvrk_gazebo_control.srv import *
import rospy
import sys
sys.path.append('/home/baothach/dvrk_grasp_pipeline_issac/src/dvrk_env/dvrk_gazebo_control/src')
from utils import open3d_ros_helper as orh
from sensor_msgs.msg import PointCloud2

def VFH_client(cloud):
    rospy.loginfo('Waiting for VFH_feature_vector.')
    rospy.wait_for_service('VFH_feature_vector')
    rospy.loginfo('Calling service VFH_feature_vector.')
    try:
        VFH_proxy = rospy.ServiceProxy('VFH_feature_vector', VFH)
        VFH_request = VFHRequest()
        VFH_request.cloud = cloud
        VFH_response = VFH_proxy(VFH_request) 
    except (rospy.ServiceException):
        rospy.loginfo('Service VFH_feature_vector call failed: %s')
    rospy.loginfo('Service VFH_feature_vector is executed %s.'%str(VFH_response.success))
    return VFH_response.extended_FPFH_vector_135


def point_cloud_publisher(ros_cloud):
    pub = rospy.Publisher('point_cloud_bao', PointCloud2, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    count = 0
    while count < 5:
        rospy.loginfo("Publising pointcloud")
        pub.publish(ros_cloud)
        rate.sleep()
        count += 1