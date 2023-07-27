from isaacgym import gymapi
import rospy
import numpy as np
from dvrk_gazebo_control.srv import *
import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
from utils import isaac_utils
from copy import deepcopy
import transformations

class Robot:
    """Robot in isaacgym class - Bao"""

    def __init__(self, gym_handle, sim_handle, env_handle, robot_handle, n_arm_dof=8, robot_name='dvrk'):
        # Isaac Gym stuff
        self.gym_handle = gym_handle
        self.sim_handle = sim_handle
        self.env_handle = env_handle
        self.robot_handle = robot_handle   
        self.n_arm_dof = n_arm_dof 
        self.robot_name = robot_name

    def arm_moveit_planner_client(self, go_home=False, current_position=None, joint_goal=None, cartesian_goal=None):
        
        # rospy.loginfo('Waiting for service moveit_cartesian_pose_planner.')
        # rospy.wait_for_service('moveit_cartesian_pose_planner')
        # rospy.loginfo('Calling service moveit_cartesian_pose_planner.')
        try:
            planning_proxy = rospy.ServiceProxy('moveit_cartesian_pose_planner', PalmGoalPoseWorld)
            planning_request = PalmGoalPoseWorldRequest()       
            planning_request.current_joint_states = current_position
            if go_home:
                planning_request.go_home = True
            elif joint_goal is not None:
                planning_request.go_to_joint_goal = True
                planning_request.joint_goal = joint_goal    # 8 first joints        
            elif cartesian_goal is not None:
                planning_request.palm_goal_pose_world = cartesian_goal
            else:
                rospy.loginfo('Missing joint goal/ cartesian goal')
            self.planning_response = planning_proxy(planning_request) 
        except (rospy.ServiceException):
            rospy.loginfo('Service moveit_cartesian_pose_planner call failed')
        # rospy.loginfo('Service moveit_cartesian_pose_planner is executed %s.'
        #         %str(self.planning_response.success))

        return self.planning_response.plan_traj, self.planning_response.success

    def gen_grasp_preshape_client(self, object_point_cloud, non_random = False):

        rospy.loginfo('Waiting for service gen_grasp_preshape.')
        rospy.wait_for_service('gen_grasp_preshape')
        rospy.loginfo('Calling service gen_grasp_preshape.')
        try:
            preshape_proxy = rospy.ServiceProxy('gen_grasp_preshape', GraspPreshape)
            preshape_request = GraspPreshapeRequest()
            preshape_request.obj = object_point_cloud
            preshape_request.non_random = non_random
            self.preshape_response = preshape_proxy(preshape_request) 
        except (rospy.ServiceException):
            rospy.loginfo('Service gen_grasp_preshape call failed:')
        rospy.loginfo('Service gen_grasp_preshape is executed.')
        # return self.preshape_response

    def get_arm_joint_positions(self):
        return deepcopy(self.gym_handle.get_actor_dof_states(self.env_handle, self.robot_handle, gymapi.STATE_POS)['pos'][:self.n_arm_dof]) 

    def get_ee_joint_positions(self):
        return deepcopy(self.gym_handle.get_actor_dof_states(self.env_handle, self.robot_handle, gymapi.STATE_POS)['pos'][self.n_arm_dof:]) 

    def get_full_joint_positions(self):
        return deepcopy(self.gym_handle.get_actor_dof_states(self.env_handle, self.robot_handle, gymapi.STATE_POS)['pos']) 


    def get_ee_cartesian_position(self, euler_format=False):
        """
        7-dimension pos + rot (quaternion)
        6-dimension pos + euler angles (euler)
        """
        if self.robot_name == "dvrk":
            state = deepcopy(self.gym_handle.get_actor_rigid_body_states(self.env_handle, self.robot_handle, gymapi.STATE_POS)[-3])
        elif self.robot_name == "kuka":
            state = deepcopy(self.gym_handle.get_actor_rigid_body_states(self.env_handle, self.robot_handle, gymapi.STATE_POS)[-6])
        # elif self.robot_name == "baxter":
        #     state = deepcopy(self.gym_handle.get_actor_rigid_body_states(self.env_handle, self.robot_handle, gymapi.STATE_POS)[])
        state = np.array(isaac_utils.isaac_format_pose_to_list(state))
        if euler_format:
            return np.hstack((state[:3], transformations.euler_from_quaternion(state[3:]))) # size (6,)
            # return np.hstack((state[:3], transformations.euler_from_quaternion(state[3:],'rxyz')))
        
        return state


    # def get_preshape_target_pose(self, object_point_cloud, robot_z_offset, non_random = False):
    #     self.gen_grasp_preshape_client(object_point_cloud, non_random)
    #     target_pose = deepcopy(preshape_response.palm_goal_pose_world[0].pose)



