#!/usr/bin/env python
import numpy as np
from math import *
import pickle
import random
import copy
# import open3d
import tf
import tf.transformations
import rospy
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, \
        PointStamped, Vector3Stamped
from dvrk_gazebo_control.srv import *
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs import point_cloud2
import transformations
import random
from scipy.spatial.transform import Rotation as R
from scipy import interpolate

# global half_width, half_height, half_depth

# with open("src/dvrk_env/dvrk_gazebo_control/src/stuff/point_cloud_box.txt", 'rb') as f:
#     points = pickle.load(f)


# obb = open3d.geometry.OrientedBoundingBox()
# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(np.array(points))
# obb = pcd.get_oriented_bounding_box()

# points = np.asarray(obb.get_box_points())
# lines = [
#     [0, 1],
#     [0, 2],
#     [0, 3],
#     [1, 6],
#     [1, 7],
#     [2, 5], 
#     [2, 7],
#     [3, 5],
#     [3, 6],
#     [4, 5],
#     [4, 6],
#     [4, 7],
# ]
# colors = [[1, 0, 0] for i in range(len(lines))]
# line_set = open3d.geometry.LineSet(
#     points=open3d.utility.Vector3dVector(points),
#     lines=open3d.utility.Vector2iVector(lines),
# )
# line_set.colors = open3d.utility.Vector3dVector(colors)
# open3d.visualization.draw_geometries([pcd, line_set]) 

# # print(obb.R)

'''
code to generate preshape
'''

def sqr_dist(x, y):
    '''
    Get the squared distance between two vectors
    '''
    xx = x-y
    xx = np.multiply(xx,xx)
    xx = np.sum(xx)
    return xx

def find_nearest_neighbor(sample_point, cloud):
    '''
    Find the nearest neigbor between sample_point and the points in cloud.
    Uses the squared Euclidean distance.
    '''
    min_dist = 10000
    min_pt = None
    for idx, pt in enumerate(cloud):
        pt_dist = sqr_dist(np.array(pt), sample_point)
        if pt_dist < min_dist:
            min_pt = np.array(pt[:])
            min_dist = pt_dist
            min_idx = idx
    return min_pt, min_idx, min_dist


class BoundingBoxFace():
    '''
    Simple class to store properties of the face of 3D bounding box
    '''
    def __init__(self, center, orientation_a, orientation_b, height, width,
                 is_top=False):
        self.center = center
        self.orientation_a = orientation_a
        self.orientation_b = orientation_b
        self.height = height
        self.width = width
        self.is_top = is_top


class GenGraspPreshape():
    def __init__(self):
        rospy.init_node('gen_grasp_preshape_server')
        # Read parameters from server
        self.use_bb_orientation = rospy.get_param('~use_bb_orientation',True)
        #self.hand_sample_dist = rospy.get_param('~hand_sample_dist', 0.03)
        # Due to gravity, the hand will be lower than the goal pose for top grasps in simulation. 
        # the top grasp need a further distance from the object.
        self.hand_type = rospy.get_param('~end_effector', 'allegro')
        self.hand_sample_dist_top = rospy.get_param('~hand_sample_dist_top', -0.01)
        self.hand_sample_dist_side = rospy.get_param('~hand_sample_dist_side', 0.03)
        # NOTE: Set to 0.0 to turn off sampling
        self.hand_dist_var = rospy.get_param('~hand_sample_dist_var', 0)
        self.palm_position_var = rospy.get_param('~palm_position_sample_var', 0)
        self.palm_ort_var = rospy.get_param('~palm_ort_sample_var', 0)
        self.roll_angle_sample_var = rospy.get_param('~/hand_roll_angle_sample_var', 0.001)
        # self.setup_joint_angle_limits()
        self.listener = tf.TransformListener()
        # self.tf_br = tf.TransformBroadcaster()
        self.service_is_called = False
        self.object_pose = PoseStamped()
        self.palm_goal_pose_world = [] 
        # The minimum distance from the object top to the table so that the hand can reach the grasp 
        # preshape without colliding the table.
        self.min_object_top_dist_to_table = rospy.get_param('~min_object_top_dist_to_table', 0.03)
        self.samples_num_per_preshape = 50
        self.exp_palm_pose = None
        # self.grasp_pose_pub = rospy.Publisher('/publish_box_points', MarkerArray, queue_size=1)

    def create_preshape_server(self):
        '''
        Create the appropriate simulated or real service callback based on parameter
        setting
        '''
        self.preshape_service = rospy.Service('gen_grasp_preshape', GraspPreshape,
                                              self.handle_gen_grasp_preshape)
        rospy.loginfo('Service gen_grasp_preshape:')
        rospy.loginfo('Ready to generate the grasp preshape.')
        #rospy.spin()
    
    def handle_gen_grasp_preshape(self, req):
        # r = R.from_quat([req.obj.pose.orientation.x, req.obj.pose.orientation.y,\
        #                                 req.obj.pose.orientation.z, req.obj.pose.orientation.w])

        print("bounding box pose haha 2: ", req.obj.pose)
        return self.gen_grasp_preshape(req)
    
    def gen_grasp_preshape(self, req):
        '''
        Grasp preshape service callback for generating a preshape from real perceptual
        data.
        '''
        response = GraspPreshapeResponse()

        self.non_random = req.non_random
        self.others = req.others

        self.object_pose.pose = req.obj.pose
        self.service_is_called = True

        bb_center_points, bb_faces = self.get_bb_faces(req.obj) 
        self.palm_goal_pose_world = []
        for i in range(len(bb_faces)):
            rospy.loginfo('############ New face ###############')
            if bb_faces[i].is_top:
                rospy.loginfo('This is the top grasp')
            # Get desired palm pose given the point from the bounding box
                palm_pose_world = self.find_palm_pose(bb_center_points[i], req.obj, bb_faces[i])
                
                self.palm_goal_pose_world.append(palm_pose_world)
                response.palm_goal_pose_world.append(palm_pose_world)
                response.is_top_grasp.append(bb_faces[i].is_top)

        response.object_pose = self.object_pose

        return response


    def get_bb_faces(self, grasp_object):
        '''
        Computes and return the centers of 3 bounding box sides: top face, two side faces 
        closer to the camera.
        '''
        homo_matrix_world_frame = self.listener.fromTranslationRotation(
            (.0, .0, .0),
            (grasp_object.pose.orientation.x, grasp_object.pose.orientation.y,
             grasp_object.pose.orientation.z, grasp_object.pose.orientation.w)
        )
        x_axis_world_frame = homo_matrix_world_frame[:3, 0]
        y_axis_world_frame = homo_matrix_world_frame[:3, 1]
        z_axis_world_frame = homo_matrix_world_frame[:3, 2]
        #x_axis_world_frame = np.array([1., 0., 0.])
        #y_axis_world_frame = np.array([0., 1., 0.])
        #z_axis_world_frame = np.array([0., 0., 1.])
        bb_center_world_frame = np.array([grasp_object.pose.position.x,
                                           grasp_object.pose.position.y,
                                           grasp_object.pose.position.z])
        # Append all faces of the bounding box except the back face.
        half_width = 0.5 * grasp_object.width#height
        half_height = 0.5 * grasp_object.height#depth
        half_depth = 0.5 * grasp_object.depth#width

        faces_world_frame = [BoundingBoxFace(bb_center_world_frame +
                                              half_width * x_axis_world_frame,
                                              y_axis_world_frame,
                                              z_axis_world_frame,
                                              grasp_object.height,
                                              grasp_object.depth),
                              BoundingBoxFace(bb_center_world_frame -
                                              half_width * x_axis_world_frame,
                                              y_axis_world_frame,
                                              z_axis_world_frame,
                                              grasp_object.height,
                                              grasp_object.depth),
                              BoundingBoxFace(bb_center_world_frame +
                                              half_height * y_axis_world_frame,
                                              x_axis_world_frame,
                                              z_axis_world_frame,
                                              grasp_object.width,
                                              grasp_object.depth),
                              BoundingBoxFace(bb_center_world_frame -
                                              half_height * y_axis_world_frame,
                                              x_axis_world_frame,
                                              z_axis_world_frame,
                                              grasp_object.width,
                                              grasp_object.depth),
                              BoundingBoxFace(bb_center_world_frame +
                                              half_depth * z_axis_world_frame,
                                              x_axis_world_frame,
                                              y_axis_world_frame,
                                              grasp_object.width,
                                              grasp_object.height),
                              BoundingBoxFace(bb_center_world_frame -
                                              half_depth * z_axis_world_frame,
                                              x_axis_world_frame,
                                              y_axis_world_frame,
                                              grasp_object.width,
                                              grasp_object.height)]

        faces_world_frame = sorted(faces_world_frame, key=lambda x: x.center[2])
        # Assign the top face
        faces_world_frame[-1].is_top = True
        
        
        ##### Need fix ######
        faces_world_frame = [faces_world_frame[-1]]
        ##### Need fix ######        
        
        # min_z_world = faces_world_frame[0].center[2]
        # max_z_world = faces_world_frame[-1].center[2]
        # # Delete the bottom face
        # del faces_world_frame[0]

        # # Sort along x axis and delete the face furthest from camera.
        # faces_world_frame = sorted(faces_world_frame, key=lambda x: x.center[0])
        # faces_world_frame = faces_world_frame[:-1]
        # # Sort along y axis and delete the face furthest from the robot. 
        # faces_world_frame = sorted(faces_world_frame, key=lambda x: x.center[1])
        # faces_world_frame = faces_world_frame[1:]
        face_centers_world_frame = []
        center_stamped_world = PointStamped()
        # center_stamped_world.header.frame_id = grasp_object.header.frame_id
        for i, face in enumerate(faces_world_frame):
            center_stamped_world.point.x = face.center[0]
            center_stamped_world.point.y = face.center[1]
            center_stamped_world.point.z = face.center[2]
            face_centers_world_frame.append(copy.deepcopy(center_stamped_world))

       

        # # If the object is too short, only select top grasps.
        # obj_height = max_z_world - min_z_world
        # rospy.loginfo('##########################')
        # rospy.loginfo('Obj_height: %s' %obj_height)
        # if obj_height < self.min_object_top_dist_to_table:
        #     rospy.loginfo('Object is short, only use top grasps!')
        #     return [face_centers_world_frame[0]], [faces_world_frame[0]]

        return face_centers_world_frame, faces_world_frame    
    
    def get_delta_x_delta_y(self, obj):
        '''
        Find delta x and delta y for the heuristic grasp (instead of just choosing the center of the top face which makes grasp less diverse)
        '''
        
        if self.non_random:
            # w_prime = 0#obj.width/2 * 0.8   # tube connect final
            # h_prime = -obj.height/2 * 0.4  #0.7             
            # w_prime = -obj.width/2 * 0.4
            # h_prime = -obj.height/2 * 0.8    
            # w_prime = np.random.uniform(low = -obj.width/2*0.85 , high = obj.width/2*0.85) # bimanual box and cylinder
            # h_prime = np.random.uniform(low = -obj.height/2*0.85 , high = -obj.height/2*0.1)   
            # w_prime = obj.width/2*0.7#np.random.uniform(low = -obj.width/2*0.7 , high = obj.width/2*0.7) # bimanual hemis
            # h_prime = -obj.height/2*0.7 #np.random.uniform(low = -obj.height/2*0.7 , high = -obj.height/2*0.1)
            # w_prime = 0 # tube connect
            # h_prime = np.random.uniform(low = 0 , high = obj.height/2*0.8)
            # w_prime = 0 # tube connect attached
            # h_prime = -np.random.uniform(low = 0 , high = obj.height/2*0.8)
            w_prime = 0   # tissue wrap
            h_prime = -obj.height/2 * 0.8
            # w_prime = np.random.uniform(low = -obj.width/2*0.3 , high = obj.width/2*0.3) # evaluate bimanual cylinder
            # h_prime = np.random.uniform(low = -obj.height/2*0.85 , high = -obj.height/2*0.7)   

            # w_prime = np.random.uniform(low = -obj.width/2*0.7 , high = obj.width/2*0.7) # evaluate bimanual hemis
            # h_prime = np.random.uniform(low = -obj.height/2*0.7 , high = -obj.height/2*0.6)   

            # w_prime = np.random.uniform(low = -obj.width/2*0.85 , high = obj.width/2*0.85) # evaluate bimanual box
            # h_prime = np.random.uniform(low = -obj.height/2*0.85 , high = -obj.height/2*0.7) 

        else:

            # w_prime = np.random.uniform(low = -obj.width/2*0.85 , high = obj.width/2*0.85) # stress prediction data collection
            # h_prime = np.random.uniform(low = 0 , high = obj.height/2*0.85)   

            # w_prime = 0#obj.height/2 * 0.8 * 0.8 #obj.width/2 * 0.4     # comparative study
            # h_prime = obj.height/2 * 0.8   #0.7  

            # w_prime = 0   # dynamics cylinder
            # h_prime = np.random.uniform(low = 0 , high = obj.height/2*0.8) 

            w_prime = obj.width/2 * 0.2
            h_prime = 0#obj.height/2 * 0.6 

            # w_prime = np.random.uniform(low = -obj.width/2*0.7 , high = obj.width/2*0.7) # evaluate bimanual hemis
            # h_prime = np.random.uniform(low = obj.height/2*0.6 , high = obj.height/2*0.7)  

            # w_prime = np.random.uniform(low = -obj.width/2*0.85 , high = obj.width/2*0.85) # evaluate bimanual box
            # h_prime = np.random.uniform(low = obj.height/2*0.7 , high = obj.height/2*0.85)   

            # w_prime = np.random.uniform(low = -obj.width/2*0.3 , high = obj.width/2*0.3) # evaluate bimanual cylinder
            # h_prime = np.random.uniform(low = obj.height/2*0.7 , high = obj.height/2*0.85)             
            
            # w_prime = 0#obj.width/2 * 0.8   # tube connect final 
            # h_prime = obj.height/2 * 0.4   #0.4  
            # w_prime = 0#-obj.width/2 * 0.3  # evaluate multi box rotation
            # # h_prime = obj.height/2 * 0
            # if self.others[0] <= 0.25:
            #     h_prime = obj.height/2 * 0.0
            # else:
            #     x = [0.15, 0.3, 0.45]
            #     y = np.array([0.25, 0.4, 0.55])#*2/4
            #     f = interpolate.interp1d(x, y) 
            #     xnew = self.others
            #     h_prime = obj.height/2 * f(xnew) 

            # w_prime = 0   # tissue wrap
            # h_prime = obj.height/2 * 0.9  

            # w_prime = 0#-obj.width/2 * 0.6   # evaluate multi cylinder rotation
            # h_prime = np.random.uniform(low = obj.height/2 * 0.2 , high = obj.height/2 * 0.4) 

            # w_prime = 0#-obj.width/2 * 0.6   # evaluate multi hemis rotation old
            # h_prime = np.random.uniform(low = obj.height/2 * 0 , high = obj.height/2 * 0.4) 

            # w_prime = obj.width/2 * 0.6     # Move plan algo
            # h_prime = obj.height/2 * 0.7              
            # w_prime = obj.width/2 * 0.0     # single box
            # h_prime = -obj.height/2 * 0.85   
            
            # w_prime = np.random.uniform(low = -obj.width/2*0.8 , high = obj.width/2*0.8)  # evaluate multi boxes rotation
            # h_prime = np.random.uniform(low = obj.height/2*0.6 , high = obj.height/2*0.8) 


            # w_prime = np.random.uniform(low = -obj.width/2*0.8 , high = obj.width/2*0.8) 
            # h_prime = np.random.uniform(low = -obj.height/2*0.8 , high = 0)  
            # w_prime = np.random.uniform(low = -obj.width/2*0.85 , high = obj.width/2*0.85) # with rotation
            # h_prime = np.random.uniform(low = 0 , high = obj.height/2*0.85)       
            # w_prime = np.random.uniform(low = -obj.width/2*0.8 , high = obj.width/2*0.8)  # evaluate multi hemis rotation
            # h_prime = np.random.uniform(low = obj.height/2*0.3 , high = obj.height/2*0.7)    
            # w_prime = np.random.uniform(low = -obj.width/2*0.85 , high = obj.width/2*0.85) # bimanual box and cylinder
            # h_prime = np.random.uniform(low = obj.height/2*0.1 , high = obj.height/2*0.85) 
            # w_prime = np.random.uniform(low = -obj.width/2*0.7 , high = obj.width/2*0.7) # bimanual hemis
            # h_prime = obj.height/2*0.7 #np.random.uniform(low = obj.height/2*0.1 , high = obj.height/2*0.7)
            # h_prime = obj.height/2 * 0.75              
            # w_prime = -obj.width/2 * 0.75     # single box test rot
            # w_prime = 0 # tube connect
            # h_prime = -np.random.uniform(low = 0 , high = obj.height/2*0.8)  
            # w_prime = 0 # tube connect attached
            # h_prime = np.random.uniform(low = 0 , high = obj.height/2*0.8)
            # w_prime = 0   # tissue wrap
            # h_prime = obj.height/2 * 0.8
            # w_prime = 0 #np.random.uniform(low = -obj.width/2*0.8 , high = obj.width/2*0.8)  # multi cylinders
            # h_prime = np.random.uniform(low = 0 , high = obj.height/2*0.8) 



        x_axis, y_axis, z_axis = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])

        quaternion =  [obj.pose.orientation.x, obj.pose.orientation.y,\
                                                obj.pose.orientation.z, obj.pose.orientation.w]  
        rot_mat = tf.transformations.quaternion_matrix(quaternion)[:3,:3]   
        if rot_mat[0,0] < 0:
            x_axis = -x_axis
        if rot_mat[1,1] < 0:
            y_axis = -y_axis
        if rot_mat[2,2] < 0:
            z_axis = -z_axis

        print("bounding box pose haha: ", rot_mat)
        # print("x, y axis:", rot_mat[:,0], rot_mat[:,1])
        x_axis_projected = w_prime * np.array([rot_mat[:,0].dot(x_axis), rot_mat[:,0].dot(y_axis)])
        y_axis_projected = h_prime * np.array([rot_mat[:,1].dot(x_axis), rot_mat[:,1].dot(y_axis)])
        print("projected: ", x_axis_projected, y_axis_projected)
        shift_vector = x_axis_projected + y_axis_projected
        # return 0, 0
        return shift_vector[0], shift_vector[1]   #delta x and delta y


    
    def find_palm_pose(self, object_point_stamped, obj, bb_face):
        '''
        Determine the desired palm pose given a point close to the object and the
        object point cloud. Finds the point on the point cloud closest to thc ve desired
        point and generates a palm pose given the point's surface normal.
        '''
        object_point = np.array([object_point_stamped.point.x, object_point_stamped.point.y, object_point_stamped.point.z])
        if not bb_face.is_top:
            # Find nearest neighbor
            points_xyz = point_cloud2.read_points(obj.cloud,field_names=['x','y','z'])

            # delta_x, delta_y = self.get_delta_x_delta_y(obj)  # instead of just the center, shift by x and y amount
            # print("delta x, y:", delta_x, delta_y)
            # object_point[:2] += np.array([delta_x, delta_y])


            closest_pt, min_idx, min_dist = find_nearest_neighbor(object_point, points_xyz)
            rospy.loginfo('Found closest point ' + str(closest_pt) + ' to obj_pt = ' +
                          str(object_point) + ' at dist = ' + str(min_dist))

            # Get associated normal
            point_normals = point_cloud2.read_points(obj.normals, field_names=['x',
                                                                           'y',
                                                                           'z'])
            for i, normal in enumerate(point_normals):
                if i == min_idx:
                    obj_normal = np.array(normal[:]) # object normal is the normal vector of the closest point
                    break
            rospy.loginfo('Associated normal n = ' + str(obj_normal))
            hand_dist = self.hand_sample_dist_side
            vec_center_to_face = object_point - np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
            if np.dot(obj_normal, vec_center_to_face) < 0.:
                obj_normal = -obj_normal
        else:
            closest_pt = object_point
            rospy.loginfo('Found closest point ' + str(closest_pt))
            obj_normal = closest_pt - np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
            obj_normal /= np.linalg.norm(obj_normal)
            hand_dist = self.hand_sample_dist_top
            rospy.loginfo('Associated normal n (top face) = ' + str(obj_normal))

        # Add Gaussian noise to the palm position
        if self.hand_dist_var > 0.:
            hand_dist += np.random.normal(0., self.hand_dist_var)
        palm_position = closest_pt + hand_dist*obj_normal
        if self.palm_position_var > 0.:
            palm_position += np.random.normal(0., self.palm_position_var, 3)

        palm_pose = PoseStamped()
        # palm_pose.header.frame_id = obj.cloud.header.frame_id
        palm_pose.header.stamp = rospy.Time.now()
        delta_x, delta_y = self.get_delta_x_delta_y(obj)  # instead of just the center, shift by x and y amount
        print("delta x, y:", delta_x, delta_y)
        palm_pose.pose.position.x = palm_position[0] + delta_x
        palm_pose.pose.position.y = palm_position[1] + delta_y
        palm_pose.pose.position.z = palm_position[2]
        rospy.loginfo('Chosen palm position is ' + str(palm_pose.pose.position))


       
        q = self.sample_palm_orientation(obj_normal)
        palm_pose.pose.orientation.x = q[0]
        palm_pose.pose.orientation.y = q[1]
        palm_pose.pose.orientation.z = q[2]
        palm_pose.pose.orientation.w = q[3]
        return palm_pose

 

    def sample_palm_orientation(self, obj_normal):
        '''
        Sample the hand wrist roll. Currently it is uniform.
        For the palm frame, x: palm normal, y: thumb, z: middle finger.
        This def is to change the yaw angle (we need roll instead ...)
        '''

        
        # roll = np.random.uniform(-np.pi, np.pi)
    
        # roll = random.uniform(*random.choice([(-np.pi, -0.05), (0.03, np.pi)]))


        
        roll = -0.02
        # roll = np.pi*(1/3)  # range from -pi/2 to pi/2
        roll_prime = np.pi/3
        x_rand = np.array([0.0, np.sin(roll_prime), np.cos(roll_prime)])

        # Project orientation vector into the tangent space of the normal
        # NOTE: assumes normal is a unit vector
        y_axis = obj_normal
        x_onto_y = x_rand.dot(y_axis)*y_axis
        x_axis = x_rand - x_onto_y
        
        # Normalize to unit length
        x_axis /= np.linalg.norm(x_axis)
        
        # Find the third orthonormal component and build the rotation matrix
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        rot_matrix = np.matrix([x_axis, y_axis, z_axis]).T
        # rospy.loginfo('Real rotation matrix: ' + str(rot_matrix))
        # rospy.loginfo('x, y, z axis = ' + str(x_axis) + "  |||  " + str(y_axis) + "  |||  " + str(z_axis))
        # rospy.loginfo('Determinant: ' + str(np.linalg.det(rot_matrix)))
        
        # Compute quaternion from rpy
        trans_matrix = np.matrix(np.zeros((4,4)))
        trans_matrix[:3,:3] = rot_matrix
        trans_matrix[3,3] = 1.
        quaternion = transformations.quaternion_from_matrix(trans_matrix)
        euler = list(transformations.euler_from_quaternion(quaternion))
        euler[0] = roll
        quaternion = transformations.quaternion_from_euler(euler[0],euler[1],euler[2])
        rospy.loginfo('Grasp orientation in quaternion= ' +  str(quaternion))
        rospy.loginfo('Grasp orientation in rotation matrix = ' +  str(transformations.quaternion_matrix(quaternion)))
        rospy.loginfo('Grasp orientation in euler angles = ' +  str(transformations.euler_from_quaternion(quaternion)))
        return quaternion


   
if __name__ == '__main__':

    gen_grasp_preshape = GenGraspPreshape()
    gen_grasp_preshape.create_preshape_server()
    rospy.spin()
    # gen_grasp_preshape.update_detection_grasp_server()
    # rate = rospy.Rate(100)
    # while not rospy.is_shutdown():
    #     gen_grasp_preshape.broadcast_palm_and_obj()
    #     rate.sleep()