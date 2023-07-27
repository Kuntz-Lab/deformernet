#!/usr/bin/env python3
import numpy as np
import pickle
import open3d
from dvrk_gazebo_control.srv import *
import rospy
import sys
sys.path.append('/home/baothach/dvrk_grasp_pipeline_issac/src/dvrk_env/dvrk_gazebo_control/src')
from utils import open3d_ros_helper as orh

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



if __name__ == "__main__":
    
    rospy.init_node('process_pca_data')
    with open("/home/baothach/shape_servo_data/point_clouds_for_pca.txt", 'rb') as f:
        point_clouds = pickle.load(f)

    feature_vectors = []
    for i, point_cloud in enumerate(point_clouds):
        
        # if i == 3:
        #     break
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(point_cloud))
        # open3d.visualization.draw_geometries([pcd])  

        # Get request data
        ros_cloud = orh.o3dpc_to_rospc(pcd)

        # extended FPFH vector size 135
        feature_vector_135 = list(VFH_client(ros_cloud))
        feature_vectors.append(feature_vector_135)
   
    with open("/home/baothach/shape_servo_data/extended_FPFH_vector_135.txt", 'wb') as f:
        pickle.dump(feature_vectors, f)           
    
    print("count: ", i+1)
    print("All Done!")
  