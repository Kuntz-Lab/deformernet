#include "ros/ros.h"
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <typeinfo>
#include <dvrk_gazebo_control/VFH.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>


bool VFH(dvrk_gazebo_control::VFH::Request  &req,
         dvrk_gazebo_control::VFH::Response &res)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());

  // Convert PointCloud2 (ROS) to PCL format
  pcl::fromROSMsg(req.cloud, *cloud);



  // Create the normal estimation class, and pass the input dataset to it
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (cloud);

  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);

  // Output datasets
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

  // Use all neighbors in a sphere of radius 3cm
  // ne.setRadiusSearch (0.03);
  ne.setKSearch(8);  // number of normal neighbors

  // Compute the features
  ne.compute (*normals);

  // cloud_normals->size () should have the same size as the input cloud->size ()*  
  
  
  // Create the VFH estimation class, and pass the input dataset+normals to it
  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud (cloud);
  vfh.setInputNormals (normals);
  // alternatively, if cloud is of type PointNormal, do vfh.setInputNormals (cloud);

  // Create an empty kdtree representation, and pass it to the FPFH estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_2 (new pcl::search::KdTree<pcl::PointXYZ> ());
  vfh.setSearchMethod (tree_2);

  // Output datasets
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

  // Compute the features
  vfh.compute (*vfhs);
  // std::cout << vfhs;

  // std::cout << "output points.size (): " << vfhs->points[0] << std::endl; 

  // float extended_FPFH_vector_135[135];
  for (int i = 0; i< 135; i++)
  {
    res.extended_FPFH_vector_135.push_back(vfhs->points[0].histogram[i]);
  }
  res.success = true;
  return true;

}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "VFH_server");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("VFH_feature_vector", VFH);
  ROS_INFO("Service VFH_feature_vector");
  ROS_INFO("Ready to get VFH");
  ros::spin();

  return 0;
}