#!/usr/bin/env python
# from trac_ik_python.trac_ik import IK

# ik_solver = IK("psm_rev_link", "psm_tool_yaw_link")

# seed_state = [0.0] * ik_solver.number_of_joints

# IK_solution = ik_solver.get_ik(seed_state, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0) # X, Y, Z, QX, QY, QZ, QW
# print(IK_solution)



# import roslib; roslib.load_manifest('grasp_pipeline')
import sys
import rospy
from dvrk_gazebo_control.srv import *
# from grasp_pipeline.srv import *
# from geometry_msgs.msg import Pose, Quaternion
# from sensor_msgs.msg import JointState

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import moveit_msgs.msg
import moveit_commander

import numpy as np
from trac_ik_python.trac_ik import IK

from moveit_msgs.msg import RobotState
from std_msgs.msg import Header
from std_msgs.msg import String



class CartesianPoseMoveitPlanner:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_goal_pose_planner_node')
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander('isaac_dvrk_arm')
        #Reference link: https://answers.ros.org/question/255792/change-planner-in-moveit-python/
        #self.group.set_planner_id('RRTstarkConfigDefault')
        self.group.set_planner_id('RRTConnectkConfigDefault')
        self.group.set_planning_time(10)
        self.group.set_num_planning_attempts(3)
        #self.group.set_planning_time(15)
        #self.group.set_num_planning_attempts(1)

        self.zero_joint_states = np.zeros(8) 
        self.left_home_pose = True
        if self.left_home_pose:
            # self.home_joint_states = np.array([0.0001086467455024831, 0.17398914694786072, -0.00015721925592515618, 
            #                                     -1.0467143058776855, 0.0006054198020137846, 
            #                                     -0.00030679398332722485, 3.3859387258416973e-06])
            self.home_joint_states = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0] )
                                                                            
        else:
            self.home_joint_states = np.array([-0.3356692769522132, -0.18960693104623297, 0.9355599880218506, 0.045651170304702046,\
                                     0.020980324428042426, 1.7311636090617426, -0.5123334395848442, -0.4493461734062968])
        self.ik_solver = IK("world", "psm_tool_yaw_link")
        self.seed_state = [0.0] * self.ik_solver.number_of_joints

        
    def go_home(self, current_joint_states):
        print 'go home'

        # Set start state       
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['psm_yaw_joint', 'psm_pitch_back_joint', 'psm_pitch_bottom_joint', 'psm_pitch_end_joint',\
                            'psm_main_insertion_joint', 'psm_tool_roll_joint', 'psm_tool_pitch_joint', 'psm_tool_yaw_joint','psm_tool_gripper1_joint',\
                            'psm_tool_gripper2_joint']
        joint_state.position = current_joint_states
        joint_state.velocity = [0]
        joint_state.effort = [0]
        
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        self.group.set_start_state(moveit_robot_state)  

        # Get plan
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(self.home_joint_states)       
        plan_home = self.group.plan()


        return plan_home

    def go_joint_goal(self, current_joint_states, joint_goal):
        print 'go to joint goal'
                    
        # Set start state       
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['psm_yaw_joint', 'psm_pitch_back_joint', 'psm_pitch_bottom_joint', 'psm_pitch_end_joint',\
                            'psm_main_insertion_joint', 'psm_tool_roll_joint', 'psm_tool_pitch_joint', 'psm_tool_yaw_joint','psm_tool_gripper1_joint',\
                            'psm_tool_gripper2_joint']
        joint_state.position = current_joint_states
        joint_state.velocity = [0]
        joint_state.effort = [0]
        
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        self.group.set_start_state(moveit_robot_state)  
           
        # Get plan
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(joint_goal)       
        plan_joint_goal = self.group.plan()
        return plan_joint_goal

    def go_zero(self):
        print 'go zero'
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(self.zero_joint_states)
        plan_home = self.group.plan()
        return plan_home

    def go_goal(self, pose):
        print 'go goal'
        self.group.clear_pose_targets()
        self.group.set_pose_target(pose)
        plan_goal = self.group.plan()
        return plan_goal

    def go_goal_trac_ik(self, pose, current_joint_states):
        print 'go goal'
        self.group.clear_pose_targets()
        
        # Get IK solution
        ik_js = self.ik_solver.get_ik(self.seed_state, pose.position.x, pose.position.y, pose.position.z,
                                        pose.orientation.x, pose.orientation.y, pose.orientation.z, 
                                        pose.orientation.w)
        if ik_js is None:
            rospy.logerr('No IK solution for motion planning!')
            return None

        # Set start state       
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['psm_yaw_joint', 'psm_pitch_back_joint', 'psm_pitch_bottom_joint', 'psm_pitch_end_joint',\
                            'psm_main_insertion_joint', 'psm_tool_roll_joint', 'psm_tool_pitch_joint', 'psm_tool_yaw_joint','psm_tool_gripper1_joint',\
                            'psm_tool_gripper2_joint']
        joint_state.position = current_joint_states
        joint_state.velocity = [0]
        joint_state.effort = [0]
        
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        self.group.set_start_state(moveit_robot_state)       
        # print("is_known: ", self.scene.get_known_object_names()) # Check if collision objects found


        # Get plan
        self.group.set_joint_value_target(np.array(ik_js))         
        plan_goal = self.group.plan()
        return plan_goal



    def handle_pose_goal_planner(self, req):
        plan = None
        if req.go_home:
            plan = self.go_home_isaac(req.current_joint_states)
        elif req.go_zero:
            plan = self.go_zero()
        elif req.go_to_joint_goal:
            plan = self.go_joint_goal(req.current_joint_states, req.joint_goal)
        else:
            # plan = self.go_goal(req.palm_goal_pose_world)
            plan = self.go_goal_trac_ik(req.palm_goal_pose_world,req.current_joint_states)
        #print plan
        response = PalmGoalPoseWorldResponse()
        response.success = False
        if plan is None:
            return response
        if len(plan.joint_trajectory.points) > 0:
            response.success = True
            response.plan_traj = plan.joint_trajectory
        return response

    def create_moveit_planner_server(self):
        rospy.Service('moveit_cartesian_pose_planner', PalmGoalPoseWorld, self.handle_pose_goal_planner)
        rospy.loginfo('Service moveit_cartesian_pose_planner:')
        rospy.loginfo('Reference frame: %s' %self.group.get_planning_frame())
        rospy.loginfo('End-effector frame: %s' %self.group.get_end_effector_link())
        rospy.loginfo('Robot Groups: %s' %self.robot.get_group_names())
        rospy.loginfo('Ready to start to plan for given palm goal poses.')

    def handle_arm_movement(self, req):
        self.group.go(wait=True)
        response = MoveArmResponse()
        response.success = True
        return response

    def create_arm_movement_server(self):
        rospy.Service('arm_movement', MoveArm, self.handle_arm_movement)
        rospy.loginfo('Service moveit_cartesian_pose_planner:')
        rospy.loginfo('Ready to start to execute movement plan on robot arm.')

if __name__ == '__main__':
    
    planner = CartesianPoseMoveitPlanner()


    # # plan = planner.go_home()
    
    # print("------------",planner.robot.get_current_state())

    # plan = planner.go_home_isaac()

    
    # plan_list = []
    # for point in plan.joint_trajectory.points:
    #     plan_list.append(point.positions)
    # print("Here is the plan", plan_list)

    # print("Here is the plan", plan)


    planner.create_moveit_planner_server()
    rospy.spin()

