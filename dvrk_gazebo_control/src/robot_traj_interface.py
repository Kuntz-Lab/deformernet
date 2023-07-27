#!/usr/bin/env python
# Contains functions for receiving joint topics and sending joint angles
import rospy
from dvrk_gazebo_control.srv import *
from sensor_msgs.msg import JointState
# from trajectory_smoothing.srv import GetSmoothTraj
from std_msgs.msg import Float64
import sys
# from rospkg import RosPack
# rp=RosPack()
# rp.list()
#path=rp.get_path('optimization_pkg')+'/scripts'
#sys.path.insert(0,path)
#from trajectory_pub import trajectoryServer
import numpy as np
import time

class robotTrajInterface:
    def __init__(self,arm_prefix='/dvrk/psm1',hand_prefix='/allegro_hand_right',init_node=True,traj_topic='/grasp/plan'):
        if(init_node):
            rospy.init_node('robot_node')
        # subscribers:
        self.arm_joint_state=JointState()
        self.arm_joint_sub=None
        self.arm_joint_sub=rospy.Subscriber("/dvrk/joint/states",
                                        JointState,self.arm_joint_state_cb,self.arm_joint_sub)
        self.got_state=False
        self.got_hand_state=False
        self.hand_joint_state=JointState()
        # self.hand_joint_sub=None
        # self.hand_joint_sub=rospy.Subscriber(hand_prefix+'/joint_states',
        #                                 JointState,self.hand_joint_state_cb,self.hand_joint_sub)

        #self.traj_server=trajectoryServer(100,robot_name='lbr4',topic_name=traj_topic,init_node=False)


        self.pub_yaw = rospy.Publisher(arm_prefix+'/yaw_joint/SetPositionTarget',Float64,queue_size=1)
        self.pub_pitchback = rospy.Publisher(arm_prefix+'/pitch_back_joint/SetPositionTarget',Float64,queue_size=1)
        self.pub_maininsert = rospy.Publisher(arm_prefix+'/main_insertion_joint/SetPositionTarget',Float64,queue_size=1)
        self.pub_t_roll = rospy.Publisher(arm_prefix+'/tool_roll_joint/SetPositionTarget',Float64,queue_size=1)
        self.pub_t_pitch = rospy.Publisher(arm_prefix+'/tool_pitch_joint/SetPositionTarget',Float64,queue_size=1)
        self.pub_t_yaw = rospy.Publisher(arm_prefix+'/tool_yaw_joint/SetPositionTarget',Float64,queue_size=1)
        
        self.loop_rate=rospy.Rate(100)


    def arm_joint_state_cb(self,joint_state):
        self.arm_joint_state=joint_state
        self.got_state=True

    # def hand_joint_state_cb(self,joint_state):
    #     self.hand_joint_state=joint_state
    #     self.got_hand_state=True

    #def viz_traj(self,j_traj):
    #    self.traj_server.viz_joint_traj(j_traj)
        
    def send_jtraj(self,j_traj):
        # before sending the joint trajectory, smoothly transfer to the initial waypoint:
        for i in range(len(j_traj.points)):
            new_jc=JointState()
            new_jc.name=j_traj.joint_names
            new_jc.position=j_traj.points[i].positions
            new_jc.velocity=j_traj.points[i].velocities
            new_jc.effort=j_traj.points[i].accelerations
            
            for i in range(len(new_jc.name)):
            # for joint in new_jc:
                # joint = new_jc[i]
                if "yaw_joint" in new_jc.name[i]:
                    self.pub_yaw.publish(new_jc.position[i])
                elif "pitch_back_joint" in new_jc.name[i]:
                    self.pub_pitchback.publish(new_jc.position[i])
                elif "main_insertion_joint" in new_jc.name[i]:
                    self.pub_maininsert.publish(new_jc.position[i])
                elif "tool_roll_joint" in new_jc.name[i]:
                    self.pub_t_roll.publish(new_jc.position[i])
                elif "tool_pitch_joint" in new_jc.name[i]: 
                    self.pub_t_pitch.publish(new_jc.position[i])
                elif "tool_yaw_joint" in new_jc.name[i]: 
                    self.pub_t_yaw.publish(new_jc.position[i])
                                    
            self.loop_rate.sleep()

        des_js=j_traj.points[-1].positions
        reached=False
        
        start_time = time.time()
        # # Check if goal is reached:
        # while(reached==False):
        #     if time.time() - start_time >= 2:
        #         break
        #     self.loop_rate.sleep()
        #     err=np.linalg.norm(np.array(des_js)-np.array(self.arm_joint_state.position))
        #     if(err<0.01):
        #         reached=True
        # rospy.loginfo('***Arm reached: %s' %str(reached))
        # rospy.loginfo('***Arm reach error: %s' %str(err))

    # def get_smooth_traj(self,jtraj):
    #     max_acc=np.ones(7)*0.25
    #     max_vel=np.ones(7)*0.4
    #     # call service for smoothing:
    #     rospy.wait_for_service('/get_smooth_trajectory')
    #     traj_call=rospy.ServiceProxy('/get_smooth_trajectory',GetSmoothTraj)
    #     resp=traj_call(jtraj,max_acc,max_vel,0.1,0.01)
    #     #resp=traj_call(jtraj,max_acc,max_vel,0.2,0.01)
    #     #print resp.smooth_traj
    #     #smooth_traj=resp.smooth_traj
    #     return resp.success, resp.smooth_traj

if __name__ == '__main__':
    robot_traj_manager = robotTrajInterface(init_node=True)
    
    go_home=True 
    place_goal_pose=None
    
    rospy.loginfo('Waiting for service moveit_cartesian_pose_planner.')
    rospy.wait_for_service('moveit_cartesian_pose_planner')
    rospy.loginfo('Calling service moveit_cartesian_pose_planner.')
    try:
        planning_proxy = rospy.ServiceProxy('moveit_cartesian_pose_planner', PalmGoalPoseWorld)
        planning_request = PalmGoalPoseWorldRequest()
        if go_home:
            planning_request.go_home = True
        elif place_goal_pose is not None:
            planning_request.palm_goal_pose_world = place_goal_pose
        else:
            # planning_request.palm_goal_pose_world = self.mount_desired_world.pose
            planning_request.palm_goal_pose_world = []
        planning_response = planning_proxy(planning_request) 
    except rospy.ServiceException, e:
        rospy.loginfo('Service moveit_cartesian_pose_planner call failed: %s'%e)
    rospy.loginfo('Service moveit_cartesian_pose_planner is executed %s.'
            %str(planning_response.success))
    
    if not planning_response.success:
        rospy.loginfo('Does not have a plan to execute!')

    plan_traj = planning_response.plan_traj
    robot_traj_manager.send_jtraj(plan_traj)