import numpy as np
import random
import time
import math
import open3d
from copy import deepcopy
import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

def pcd_ize(pc, color=None, vis=False):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc   

class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

    def get_simulation_state(self, object_state, robot_state):
        self.object_simulation_state = object_state
        self.robot_simulation_state = robot_state

    def get_ee_pos(self, ee_pos):   # end-effector x, y, z
        self.ee_pos = ee_pos

class RRTSearchTree:
    def __init__(self, init):
        self.root = TreeNode(init)
        self.nodes = [self.root]
        self.edges = []

    def find_nearest(self, s_query):
        min_d = 1000000
        nn = self.root
        for n_i in self.nodes:
            d = np.linalg.norm(s_query - n_i.state)
            if d < min_d:
                nn = n_i
                min_d = d
        return (nn, min_d)

    def add_node(self, node, parent):
        self.nodes.append(node)
        self.edges.append((parent.state, node.state))
        node.parent = parent
        parent.add_child(node)

    def get_states_and_edges(self):
        states = np.array([n.state for n in self.nodes])
        return (states, self.edges)

    def get_back_path(self, n):
        path = []
        while n.parent is not None:
            path.append(n.state)
            n = n.parent
        path.reverse()
        return path

class RobotRRT:
    def __init__(self, num_samples, num_dimensions=2, step_length = 1, lims = None,
                 collision_func=None, goal_pc = None, success_threshold=None):   
        # Isaac Gym stuff
        # self.gym_handle = gym_handle
        # self.sim_handle = sim_handle
        # self.env_handle = env_handle
        # self.robot_handle = robot_handle

        self.run_FEM_is_complete = True

        # RRT stuff
        self.K = num_samples
        self.n = num_dimensions # 8 dof for da vinci arm
        self.sample_count = 0
        self.epsilon = step_length
        self.success_threshold = success_threshold


        self.in_collision = collision_func
        if collision_func is None:
            self.in_collision = self.fake_in_collision

        # Setup range limits
        self.limits = lims
        if self.limits is None:
            self.limits = []
            for n in range(num_dimensions):
                self.limits.append([0,100])
            self.limits = np.array(self.limits)

        self.ranges = self.limits[:,1] - self.limits[:,0]
        self.found_path = False
        self.invalid_state_occur = False
        self.reach_max_samples = False

        if goal_pc is not None:
            self.pcd_goal = pcd_ize(down_sampling(goal_pc, num_pts=1024))   
            self.pcd_goal.paint_uniform_color([1,0,0])

        self.traj_index = 0   
        self.reset_to_init = True    

    def save_init_states(self, init_obj_state, init_robot_state, init_joint_states, init_ee_pos):
        self.init_robot_state = init_robot_state
        self.init_obj_state = init_obj_state
               
        # Build tree and search
        init = np.array(init_joint_states)
        # print("init.shape:", init)
        self.T = RRTSearchTree(init)  

        self.T.nodes[0].get_simulation_state(init_obj_state, init_robot_state)
        self.T.nodes[0].get_ee_pos(init_ee_pos)  

    def is_goal(self, current_pc, goal_oriented = False, threshold = 0.5):
        if goal_oriented:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(current_pc)            
            if self.compute_chamfer_dist(pcd) <= threshold:
                return True
            else:
                return False           
        else:
            assert self.run_FEM_is_complete == True
            obj_point_cloud = node.object_simulation_state.numpy()[:, :3]  
            failed_points = np.array([p for p in obj_point_cloud if self.constrain_plane[0]*p[0] + self.constrain_plane[1]*p[1] + self.constrain_plane[2]*p[2] > -self.constrain_plane[3]])                   
            if len(failed_points) == 0:
                return True
            else:
                return False 

    def is_valid(self, node, max_range=[0.05, 0.05, 0.05]):
        if node.parent == None:
            return True
        if abs(node.ee_pos["x"] - node.parent.ee_pos["x"]) <  max_range[0] and \
            abs(node.ee_pos["y"] - node.parent.ee_pos["y"]) <  max_range[1] and \
            abs(node.ee_pos["z"] - node.parent.ee_pos["z"]) <  max_range[2]:
            return True

        else:
            return False

    def get_final_path(self):
        if self.found_path:
            path = self.T.get_back_path(self.T.nodes[-1])
            return path
        else:
            return None
            # self.T.add_node(TreeNode(self.goal), self.T.nodes[-1])        
            # self.found_path = True
            
        
        # else:
        #     return None  # have not found a path yet because not reached goal

    def update_node_object_state(self, object_state, robot_state):
        """
        store object simulation state to the node that was just added to the tree
        """
        self.T.nodes[-1].get_simulation_state(object_state, robot_state)    

    # def get_pos_target(state):
    #     pos_target = state
    #     return np.array(pos_target + [0.35,-0.35], dtype=np.float32)   
    
    def build_rrt(self):
        '''
        Build the rrt from init to goal
        Returns new node needed for FEM running

        '''

        # Fill me in!
        
        # for i in range(self.K):
        if self.sample_count <= self.K:
            self.sample_count += 1
            if self.sample_count > self.K:
                self.reach_max_samples = True

            rnd_node = self.sample()
            new_node, parent_node = self.extend(rnd_node)
            print("random: ", rnd_node)
            
            if not self.in_collision(new_node.state):
                # self.T.add_node(new_node, parent_node)    # add node to tree
                return (new_node, parent_node)
            else:
                return None

        # else:
        #     self.reach_max_samples = True

    def sample(self):
        '''
        Sample a new configuration and return
        '''
        # Return goal with connect_prob probability
        # Fill me in!
        rnd_config = []
        
        for i in range(self.n):
            rnd = random.uniform(self.limits[i][0], self.limits[i][1])
            rnd_config.append(rnd)
            
        return np.array(rnd_config)
        
        

    def extend(self, q):
        '''
        Perform rrt extend operation.
        q - new configuration to extend towards
        '''
        

        nearest_node, d = self.T.find_nearest(q) 
        # print("nearest_node.state", nearest_node.state)
           
        delta = q - nearest_node.state

        delta_new = (self.epsilon/d)*np.array(delta)
        
        new_node = TreeNode(nearest_node.state + delta_new)
        return new_node, nearest_node

    def compute_chamfer_dist(self, pcd, vis=False):
        if vis:
            open3d.visualization.draw_geometries([pcd, self.pcd_goal])
        return np.linalg.norm(np.asarray(self.pcd_goal.compute_point_cloud_distance(pcd))) 

    def fake_in_collision(self, q):
        '''
        We never collide with this function!
        '''
        return False                
    
