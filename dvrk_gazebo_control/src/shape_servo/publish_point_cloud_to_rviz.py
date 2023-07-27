#!/usr/bin/env python3
import sys
sys.path.append('/home/baothach/dvrk_grasp_pipeline_issac/src/dvrk_env/dvrk_gazebo_control/src')
from utils import open3d_ros_helper as orh
import open3d
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2





def point_cloud_publisher(ros_cloud):
    pub = rospy.Publisher('point_cloud_bao_2', PointCloud2, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    # pub.publish(ros_cloud)
    count = 0
    # while count < 5:
    while not rospy.is_shutdown():
        rospy.loginfo("Publising pointcloud")
        pub.publish(ros_cloud)
        rate.sleep()
        count += 1

if __name__ == '__main__':
    rospy.init_node('publish_pc_to_rviz')
    pcd = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/visualization/goal_point_cloud.pcd")
    # open3d.visualization.draw_geometries([pcd])
    ros_cloud = orh.o3dpc_to_rospc(pcd, frame_id="map")

    try:
        point_cloud_publisher(ros_cloud)
    except rospy.ROSInterruptException:
        pass