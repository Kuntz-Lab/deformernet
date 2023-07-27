#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64

def talker():
    pub_gripper1 = rospy.Publisher('/dvrk/psm1/tool_gripper1_joint/SetPositionTarget', Float64, queue_size=10)
    pub_gripper2 = rospy.Publisher('/dvrk/psm1/tool_gripper2_joint/SetPositionTarget', Float64, queue_size=10)
    pub_gripper1_effort = rospy.Publisher('/dvrk/psm1/tool_gripper1_joint/SetEffort', Float64, queue_size=10)
    pub_gripper2_effort = rospy.Publisher('/dvrk/psm1/tool_gripper2_joint/SetEffort', Float64, queue_size=10)    
    rospy.init_node('gripper', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        joint_angle = 0.15
        effort = 5
        rospy.loginfo(joint_angle)
        pub_gripper1.publish(joint_angle)
        pub_gripper2.publish(joint_angle)
#        pub_gripper1_effort.publish(effort)
#        pub_gripper2_effort.publish(effort)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
