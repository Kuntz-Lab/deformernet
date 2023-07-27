import rospy

from sensor_msgs.msg import JointState
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import PoseStamped, Pose

import moveit_interface.util as moveit_util
from ll4ma_util import ros_util


def get_plan(target_pose, state, ee_link, frame='world', disable_collisions=[],
             max_vel_factor=0.3, max_acc_factor=0.1, cartesian_path=False,
             max_plan_attempts=1, planning_time=1.):
    """
    Helper function to request a plan from Isaac Gym state info.
    """
    pose_stmp = PoseStamped()
    pose_stmp.header.frame_id = frame
    pose_stmp.pose.position.x = target_pose[0]
    pose_stmp.pose.position.y = target_pose[1]
    pose_stmp.pose.position.z = target_pose[2]
    pose_stmp.pose.orientation.x = target_pose[3]
    pose_stmp.pose.orientation.y = target_pose[4]
    pose_stmp.pose.orientation.z = target_pose[5]
    pose_stmp.pose.orientation.w = target_pose[6]

    # Publish pose for debugging in rviz
    target_pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=1)
    for _ in range(100):
        target_pose_pub.publish(pose_stmp)
    
    joint_state = JointState()
    joint_state.position = state.joint_position.cpu().numpy().flatten().tolist()
    joint_state.velocity = state.joint_velocity.cpu().numpy().flatten().tolist()
    joint_state.effort = state.joint_torque.cpu().numpy().flatten().tolist()
    joint_state.name = state.joint_names

    objects = {k: v.to_dict() for k, v in state.objects.items()}

    for obj_name in state.objects.keys():
        if obj_name in disable_collisions:
            objects[obj_name]['operation'] = CollisionObject.REMOVE
        else:
            objects[obj_name]['operation'] = CollisionObject.ADD
        obj_state = state.object_states[obj_name].clone().cpu().numpy()
        obj_pose = Pose()
        obj_pose.position.x = obj_state[0]
        obj_pose.position.y = obj_state[1]
        obj_pose.position.z = obj_state[2]
        obj_pose.orientation.x = obj_state[3]
        obj_pose.orientation.y = obj_state[4]
        obj_pose.orientation.z = obj_state[5]
        obj_pose.orientation.w = obj_state[6]
        objects[obj_name]['pose'] = obj_pose
        objects[obj_name]['color'] = state.object_colors[obj_name]
    
    resp, success = moveit_util.get_plan(pose_stmp, joint_state, ee_link, objects,
                                         max_vel_factor, max_acc_factor, cartesian_path,
                                         max_plan_attempts, planning_time)        
    return resp, success
