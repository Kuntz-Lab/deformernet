from geometry_msgs.msg import PoseStamped
def isaac_format_pose_to_PoseStamped(body_states):
    ros_pose = PoseStamped()
    ros_pose.header.frame_id = 'world'
    ros_pose.pose.position.x = body_states["pose"]["p"]["x"]
    ros_pose.pose.position.y = body_states["pose"]["p"]["y"]
    ros_pose.pose.position.z = body_states["pose"]["p"]["z"]
    ros_pose.pose.orientation.x = body_states["pose"]["r"]["x"]
    ros_pose.pose.orientation.y = body_states["pose"]["r"]["y"]
    ros_pose.pose.orientation.z = body_states["pose"]["r"]["z"]
    ros_pose.pose.orientation.w = body_states["pose"]["r"]["w"]
    return ros_pose




