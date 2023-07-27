

# import roslib; roslib.load_manifest('grasp_pipeline')
# import rospy
# from dvrk_gazebo_control.srv import *
# from geometry_msgs.msg import Pose, Quaternion
# from sensor_msgs.msg import JointState, CameraInfo
# import tf
import numpy as np
import h5py
import h5sparse
import os


class RecordGraspData_sparse():
    def __init__(self):

        # self.data_recording_path = rospy.get_param('~data_recording_path', '/home/baothach/dvrk_grasp_data/')
        # self.use_hd = rospy.get_param('~use_hd', True)
        self.grasp_file_name = "/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch2/shape_servo_data.h5"
        self.initialize_data_file()
        

    def initialize_data_file(self):
        #a: Read/write if exists, create otherwise (default)
        grasp_file = h5sparse.File(self.grasp_file_name, 'a')

        cur_grasp_id_key = 'cur_grasp_id'
        if cur_grasp_id_key not in grasp_file:
            grasp_file.create_dataset(cur_grasp_id_key, data=-1)


        grasp_file.close()
    

    def handle_record_grasp_data(self, grasp_pose, pc, pc_goal, positions):
        #'r+': Read/write, file must exist
        grasp_file = h5sparse.File(self.grasp_file_name, 'r+')
        grasp_file['cur_grasp_id'][()] +=1      # Update grasp id

        # group = grasp_file.create_group("grasp " + str(grasp_file['cur_grasp_id'][()]))     # create new group for new grasp id
        # grasp_file.create_dataset('manipulation pose ' + str(grasp_file['cur_grasp_id'][()]), data = grasp_pose)
        grasp_file.create_dataset('point cloud init ' + str(grasp_file['cur_grasp_id'][()]), data = pc)
        grasp_file.create_dataset('point cloud goal ' + str(grasp_file['cur_grasp_id'][()]), data = pc_goal)
        grasp_file.create_dataset('position ' + str(grasp_file['cur_grasp_id'][()]), data = positions)
        

        grasp_file.close()


class RecordGraspData():
    def __init__(self):

        # self.data_recording_path = rospy.get_param('~data_recording_path', '/home/baothach/dvrk_grasp_data/')
        # self.use_hd = rospy.get_param('~use_hd', True)
        self.grasp_file_name = "/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch1/shape_servo_data.h5"
        self.initialize_data_file()
        

    def initialize_data_file(self):
        #a: Read/write if exists, create otherwise (default)
        grasp_file = h5py.File(self.grasp_file_name, 'a')

        cur_grasp_id_key = 'cur_grasp_id'
        if cur_grasp_id_key not in grasp_file:
            grasp_file.create_dataset(cur_grasp_id_key, data=-1)


        grasp_file.close()
    

    def handle_record_grasp_data(self, grasp_pose, point_clouds, positions):
        #'r+': Read/write, file must exist
        grasp_file = h5py.File(self.grasp_file_name, 'r+')
        grasp_file['cur_grasp_id'][()] +=1      # Update grasp id

        # group = grasp_file.create_group("grasp " + str(grasp_file['cur_grasp_id'][()]))     # create new group for new grasp id
        # grasp_file.create_dataset('manipulation pose ' + str(grasp_file['cur_grasp_id'][()]), data = grasp_pose)
        grasp_file.create_dataset('point cloud ' + str(grasp_file['cur_grasp_id'][()]), data = point_clouds)
        grasp_file.create_dataset('position ' + str(grasp_file['cur_grasp_id'][()]), data = positions)
        

        grasp_file.close()

class RecordGraspData_2():
    def __init__(self, data_recording_path="/home/baothach/shape_servo_data/keypoints/batch7/point_cloud"):

        # self.data_recording_path = rospy.get_param('~data_recording_path', '/home/baothach/dvrk_grasp_data/')
        # self.use_hd = rospy.get_param('~use_hd', True)
        # self.grasp_file_name = "/home/baothach/shape_servo_data/multi_grasps/batch_2"
        self.data_recording_path = data_recording_path
        self.initialize_data_file()
        

    def initialize_data_file(self):
        #a: Read/write if exists, create otherwise (default)
        grasp_file_name = os.path.join(self.data_recording_path, "source", "ordered_point_clouds.h5")
        grasp_file = h5py.File(grasp_file_name, 'a')
        cur_grasp_id_key = 'cur_id'
        if cur_grasp_id_key not in grasp_file:
            grasp_file.create_dataset(cur_grasp_id_key, data=-1)

        grasp_file.close()

        grasp_file_name = os.path.join(self.data_recording_path, "target", "ordered_point_clouds.h5")
        grasp_file = h5py.File(grasp_file_name, 'a')
        cur_grasp_id_key = 'cur_id'
        if cur_grasp_id_key not in grasp_file:
            grasp_file.create_dataset(cur_grasp_id_key, data=-1)

        grasp_file.close()    

    def handle_record_grasp_data(self, point_cloud, split = 'source'):
        #'r+': Read/write, file must exist
        
        
        grasp_file_name = os.path.join(self.data_recording_path, split, "ordered_point_clouds.h5")        
        grasp_file = h5py.File(grasp_file_name, 'r+')
        grasp_file['cur_id'][()] +=1      # Update grasp id

        
        grasp_file.create_dataset('point cloud ' + str(grasp_file['cur_id'][()]) + " " + split, data = point_cloud)
        
        

        grasp_file.close()


 

