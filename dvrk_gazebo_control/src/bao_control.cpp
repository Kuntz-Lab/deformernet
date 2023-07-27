// Author: Bao Thach, Anton Deguet
// Date: 2021-01-01

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <bao_control.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/JointState.h>




template<>
void PublisherSubscriber<std_msgs::Float64, sensor_msgs::JointState>::subscriberCallback(const sensor_msgs::JointState::ConstPtr& receivedMsg)
{
//    ROS_INFO("I received the following: [%f %f %f %f %f %f]", receivedMsg->position[0], receivedMsg->position[1], receivedMsg->position[2], receivedMsg->position[3], receivedMsg->position[4], receivedMsg->position[5]);
    ROS_INFO("Sending the info to 'echo_bao' topic");
    std_msgs::Float64 echo_msg;
    echo_msg.data = receivedMsg->position[2];
    publisherObject.publish(echo_msg);
    std::vector<std_msgs::Float64> msg;
    msg.resize(6);
//      msg[0].data=receivedMsg->position[0];
//      msg[1].data=receivedMsg->position[1];
//      msg[2].data=receivedMsg->position[1];
//      msg[3].data=receivedMsg->position[3];
//      msg[4].data=receivedMsg->position[4];
//      msg[5].data=receivedMsg->position[5];
      
      
      msg[0].data=1.5708;
      msg[1].data=0.0;
      msg[2].data=0.0;
      msg[3].data=0.0;
      msg[4].data=0.0;
      msg[5].data=0.0;
      
      psm1Pub[0].publish(msg[0]);
      psm1Pub[1].publish(msg[1]);
      psm1Pub[2].publish(msg[2]);
      psm1Pub[3].publish(msg[3]);
      psm1Pub[4].publish(msg[4]);
      psm1Pub[5].publish(msg[5]);



}


int main(int argc, char **argv)
{

  ros::init(argc, argv, "bao_control_node");

  
  PublisherSubscriber<std_msgs::Float64, sensor_msgs::JointState> parrot("echo_bao", "/dvrk/PSM1/state_joint_desired", 1000);
  

  ros::spin();


  return 0;
}



