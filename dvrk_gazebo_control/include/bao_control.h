#include <ros/ros.h>
#include <string>
#include <std_msgs/Float64.h>

template <typename PublishT, typename SubscribeT>

class PublisherSubscriber
{
public:
    PublisherSubscriber() {}
    PublisherSubscriber(std::string publishTopicName, std::string subscribeTopicName, int queueSize)
    {
        publisherObject = nH.advertise<PublishT>(publishTopicName, queueSize);
        subscriberObject = nH.subscribe<SubscribeT>(subscribeTopicName, queueSize, &PublisherSubscriber::subscriberCallback, this);
        
        
        psm1Pub.resize(6);
        psm1Pub[0]=nH.advertise<std_msgs::Float64>("/dvrk/psm1/yaw_joint/SetPositionTarget",10);
        psm1Pub[1]=nH.advertise<std_msgs::Float64>("/dvrk/psm1/pitch_back_joint/SetPositionTarget",10);
        psm1Pub[2]=nH.advertise<std_msgs::Float64>("/dvrk/psm1/main_insertion_joint/SetPositionTarget",10);
        psm1Pub[3]=nH.advertise<std_msgs::Float64>("/dvrk/psm1/tool_pitch_joint/SetPositionTarget",10);
        psm1Pub[4]=nH.advertise<std_msgs::Float64>("/dvrk/psm1/tool_roll_joint/SetPositionTarget",10);
        psm1Pub[5]=nH.advertise<std_msgs::Float64>("/dvrk/psm1/tool_yaw_joint/SetPositionTarget",10);
        
        
        
        cartPub.resize(19);
        char topic[100];
        for (int i=1;i<4;i++)
        {
          for (int j=0;j<5;j++)
          {
            sprintf(topic, "/dvrk/suj/suj_psm%d_J%d/SetPositionTarget", i,j);

            cartPub[5*(i-1)+j] = nH.advertise<std_msgs::Float64>(topic,1000);
          }
        }
        for (int j=0;j<4;j++)
        {
          sprintf(topic, "/dvrk/suj/suj_ecm_J%d/SetPositionTarget",j);

          cartPub[15+j] = nH.advertise<std_msgs::Float64>(topic,1000);
        }        
    }
    void subscriberCallback(const typename SubscribeT::ConstPtr& receivedMsg);
    
protected:
    ros::Subscriber subscriberObject;
    ros::Publisher publisherObject;
    std::vector<ros::Publisher> ecmPub, psm1Pub, psm2Pub, psm3Pub;
    std::vector<ros::Publisher> cartPub;
    ros::NodeHandle nH;
        
    
};


//class dvrk_gazebo_control{
//private:
//  std::vector<ros::Publisher> ecmPub, psm1Pub, psm2Pub, psm3Pub;
//  ros::Subscriber link_states, image_sub;
//  ros::Publisher plot_x, plot_y, plot_z;
//  std::vector<ros::Publisher> cartPub;
//public:
//  dvrk_gazebo_control(ros::NodeHandle n){


//    psm1Pub.resize(5);
//    psm1Pub[0]=n.advertise<std_msgs::Float64>("/dvrk/psm1/outer_yaw_joint/SetPositionTarget",10);
//    psm1Pub[1]=n.advertise<std_msgs::Float64>("/dvrk/psm1/outer_pitch_joint_1/SetPositionTarget",10);
//    psm1Pub[2]=n.advertise<std_msgs::Float64>("/dvrk/psm1/outer_insertion_joint/SetPositionTarget",10);
//    psm1Pub[3]=n.advertise<std_msgs::Float64>("/dvrk/psm1/outer_roll_joint/SetPosition",10);
//    psm1Pub[4]=n.advertise<std_msgs::Float64>("/dvrk/psm1/rev_joint/SetPositionTarget",10);
//    
//    ecmPub.resize(4);
//    ecmPub[0]=n.advertise<std_msgs::Float64>("/dvrk/ecm/yaw_joint/SetPositionTarget",10);
//    ecmPub[1]=n.advertise<std_msgs::Float64>("/dvrk/ecm/pitch_front_joint/SetPositionTarget",10);
//    ecmPub[2]=n.advertise<std_msgs::Float64>("/dvrk/ecm/main_insertion_joint/SetPositionTarget",10);
//    ecmPub[3]=n.advertise<std_msgs::Float64>("/dvrk/ecm/tool_joint/SetPosition",1000);
//  

//    psm2Pub.resize(5);
//    psm2Pub[0]=n.advertise<std_msgs::Float64>("/dvrk/psm2/outer_yaw_joint/SetPositionTarget",10);
//    psm2Pub[1]=n.advertise<std_msgs::Float64>("/dvrk/psm2/outer_pitch_joint_1/SetPositionTarget",10);
//    psm2Pub[2]=n.advertise<std_msgs::Float64>("/dvrk/psm2/outer_insertion_joint/SetPositionTarget",10);
//    psm2Pub[3]=n.advertise<std_msgs::Float64>("/dvrk/psm2/outer_roll_joint/SetPosition",10);
//    psm2Pub[4]=n.advertise<std_msgs::Float64>("/dvrk/psm2/rev_joint/SetPositionTarget",10);

//     
//    psm3Pub.resize(5);
//    psm3Pub[0]=n.advertise<std_msgs::Float64>("/dvrk/psm3/outer_yaw_joint/SetPositionTarget",1000);
//    psm3Pub[1]=n.advertise<std_msgs::Float64>("/dvrk/psm3/outer_pitch_joint_1/SetPositionTarget",1000);
//    psm3Pub[2]=n.advertise<std_msgs::Float64>("/dvrk/psm3/outer_insertion_joint/SetPositionTarget",1000);
//    psm3Pub[3]=n.advertise<std_msgs::Float64>("/dvrk/psm3/outer_roll_joint/SetPosition",1000);
//    psm3Pub[4]=n.advertise<std_msgs::Float64>("/dvrk/psm3/rev_joint/SetPositionTarget",1000);


//    cartPub.resize(19);
//    char topic[100];
//    for (int i=1;i<4;i++)
//    {
//      for (int j=0;j<5;j++)
//      {
//        //sprintf(topic, "/dvrk/suj/suj_psm1%d_J%d/SetPosition", i,j);
//        sprintf(topic, "/dvrk/suj/suj_psm%d_J%d/SetPositionTarget", i,j);

//        cartPub[5*(i-1)+j] = n.advertise<std_msgs::Float64>(topic,1000);
//      }
//    }
//    for (int j=0;j<4;j++)
//    {
//      sprintf(topic, "/dvrk/suj/suj_ecm_J%d/SetPositionTarget",j);

//      cartPub[15+j] = n.advertise<std_msgs::Float64>(topic,1000);
//    }

//  }

////  void showImage(const sensor_msgs::ImageConstPtr& img);
////  void getecmEndEffector(const gazebo_msgs::LinkStatesPtr &msg);
//  void PublishcartStates();
//  void PublishecmStates();
//  void Publishpsm1States();
//  void Publishpsm2States();
//  void Publishpsm3States();

//};
