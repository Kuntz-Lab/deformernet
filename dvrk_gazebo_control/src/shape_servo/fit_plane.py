import open3d
import pyransac3d as pyrsc
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


def vecalign(a, b):
    '''
    Returns the rotation matrix that can rotate the 3 dimensional b vector to
    be aligned with the a vector.

    @param a (3-dim array-like): destination vector
    @param b (3-dim array-like): vector to rotate align with a in direction

    the vectors a and b do not need to be normalized.  They can be column
    vectors or row vectors
    '''
    with np.errstate(divide='raise', under='raise', over='raise', invalid='raise'):
        a = np.asarray(a)
        b = np.asarray(b)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        cos_theta = a.dot(b) # since a and be are unit vecs, a.dot(b) = cos(theta)

        # if dot is close to -1, vectors are nearly equal
        if np.isclose(cos_theta, 1):
            # TODO: solve better than just assuming they are exactly identical
            return np.eye(3)

        # if dot is close to -1, vectors are nearly opposites
        if np.isclose(cos_theta, -1):
            # TODO: solve better than just assuming they are exactly opposite
            return -np.eye(3)

        axis = np.cross(b, a)
        sin_theta = np.linalg.norm(axis)
        axis = axis / sin_theta
        c = cos_theta
        s = sin_theta
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]

        # angle-axis formula to create a rotation matrix
        return np.array([
            [t*x*x + c  , t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c  , t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c  ],
            ])


def get_goal_plane(constrain_plane, initial_pc, check = False, delta = 0.01, current_pc=[]):
    # print("constrain_plane: ", constrain_plane)
    
    if check:
        # success = all([constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3] \
        #             for p in current_pc])
        failed_points = np.array([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])                   
        if len(failed_points) == 0:
            return 'success'
        else:
            # failed_points = np.array([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])
            # print("failed_points:", failed_points)
            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(failed_points)
            pcd2.paint_uniform_color([1,0,0])
            # open3d.visualization.draw_geometries([pcd2])             
            constrain_plane = constrain_plane.copy()
            constrain_plane[3] += delta


    constrain_plane = constrain_plane.copy()
    constrain_plane[3] += 0.03

    # if len(current_pc) != 0:
    #     initial_pc = current_pc 
    failed_points = np.array([p for p in initial_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])
    passed_points = np.array([p for p in initial_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] <= -constrain_plane[3]])

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(failed_points)
    pcd2.paint_uniform_color([1,0,0])
    # open3d.visualization.draw_geometries([pcd2])

    # Find rotation point:
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(initial_pc))
    center = pcd.get_center().reshape(3)
    dist = (np.dot(center, constrain_plane[:3]) + constrain_plane[3])/np.linalg.norm(constrain_plane[:3])
    unit_normal = np.array(constrain_plane[:3])/np.linalg.norm(constrain_plane[:3])
    rot_pt = center - dist*unit_normal

    pcd.points = open3d.utility.Vector3dVector(passed_points)
    pcd.paint_uniform_color([0,0,0])


    # Fit a plane to the failed_points
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(failed_points, thresh=0.001, maxIteration=1000)
    if best_eq[2] > 0:
        best_eq = -np.array(best_eq)


    r = vecalign(np.array(best_eq)[:3], constrain_plane[:3])
    pcd2.rotate(R = r.T, center = rot_pt.reshape((3,1)))

    goal_pcd = pcd + pcd2
    # open3d.visualization.draw_geometries([goal_pcd])
    return np.asarray(goal_pcd.points)


def main():

    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.paint_uniform_color([0,0,1])
    pcd_original = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd")
    pcd = copy.deepcopy(pcd_original)
    # open3d.visualization.draw_geometries([pcd])
    pcd_original.paint_uniform_color([0,0,0])

    # constrain_plane = np.array([3, 1, 0, 0.2])
    # constrain_plane = np.array([1, 2, 0, 0.85])  
    # constrain_plane = np.array([0, 1, 0, 0.45]) #0.36 - 0.45
    # constrain_plane = np.array([1, 1, 0, 0.33])  # 0.33 - 0.42
    constrain_plane = np.array([0.12162162, 1, 0, 0.40989865])     #0.34 - 0.45
    constrain_plane = np.array([0.27766052,  1.,          0.,          0.39016108+2.5/100]) 



    # plane = open3d.geometry.TriangleMesh.create_box(width=0.2, height=0.001, depth=0.1)
    # plane.paint_uniform_color([0,0,1])
    # open3d.visualization.draw_geometries([pcd, plane.translate((0-0.1,-0.4,0))])
    points  = np.array([p for p in np.asarray(pcd.points) if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])

    # Find rotation point:
    max_z = max([p[2] for p in points])
    center = pcd.get_center().reshape(3)
    dist = (np.dot(center, constrain_plane[:3]) + constrain_plane[3])/np.linalg.norm(constrain_plane[:3])
    unit_normal = np.array(constrain_plane[:3])/np.linalg.norm(constrain_plane[:3])
    rot_pt = center - dist*unit_normal

    pcd.points = open3d.utility.Vector3dVector(np.array([p for p in np.asarray(pcd.points) if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] <= -constrain_plane[3]]))
    pcd.paint_uniform_color([0,0,0])

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(points)
    pcd2.paint_uniform_color([1,0,0])



    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(points, thresh=0.001, maxIteration=1000)
    if best_eq[2] > 0:
        best_eq = -np.array(best_eq)
    # best_eq = np.array([0.02465008, 0.0274112,  0.99932027])



    r = vecalign(np.array(best_eq)[:3], constrain_plane[:3])
    pcd2.rotate(R = r.T, center = rot_pt.reshape((3,1)))
    # open3d.visualization.draw_geometries([pcd, pcd2])
    # open3d.visualization.draw_geometries([pcd + pcd2, sphere])
    # open3d.io.write_point_cloud("/home/baothach/shape_servo_data/new_task/test_fit_plane/goal_1.pcd", pcd+pcd2)




    # plane1 = pyrsc.Plane()
    # best_eq, best_inliers = plane1.fit(np.array(pcd2.points), thresh=0.001, maxIteration=1000)
    # # best_eq = constrain_plane
    # num_pts = 50000
    # x_range = 0.1
    # z_range = 0.1
    # plane = []
    # for i in range(num_pts):
    #     x = np.random.uniform(-x_range, x_range)
    #     z = np.random.uniform(0, z_range)
    #     y = -(best_eq[0]*x + best_eq[2]*z + best_eq[3])/best_eq[1]
    #     if -0.5 < y < 0:
    #         plane.append([x, y, z])

    # plane_vis = open3d.geometry.PointCloud()
    # plane_vis.points = open3d.utility.Vector3dVector(np.array(plane))
    # plane_vis.paint_uniform_color([0,1,0])

    best_eq = constrain_plane
    num_pts = 5000
    x_range = 0.2
    z_range = 0.2
    plane = []
    for i in range(num_pts):
        x = np.random.uniform(-x_range, x_range)
        z = np.random.uniform(0, z_range)
        y = -(best_eq[0]*x + best_eq[2]*z + best_eq[3])/best_eq[1]
        if -1 < y < 1:
            plane.append([x, y, z])        

    plane_vis2 = open3d.geometry.PointCloud()
    plane_vis2.points = open3d.utility.Vector3dVector(np.array(plane))
    plane_vis2.paint_uniform_color([0,0,1])

    # open3d.visualization.draw_geometries([pcd_original, pcd2, plane_vis2])


    open3d.visualization.draw_geometries([pcd2, pcd_original, plane_vis2, sphere])     

main()



    # print("best eq:", best_eq)
    # print("best inliers:", best_inliers)
    # for i in range(1):
    #     r = R.align_vectors(constrain_plane[:3].reshape((1,3)), np.array(best_eq)[:3].reshape((1,3))/np.sqrt(2))
    #     print(r[0].as_matrix())
    #     r = r[0].as_matrix()
    #     print("from scipy:", constrain_plane[:3].reshape((1,3)) @ r)
    #     print("ground truth: ", np.array(best_eq)[:3].reshape((1,3)))
    #     # print("two vector:", constrain_plane[:3].reshape((1,3)), np.array(best_eq)[:3].reshape((1,3)))
    #     print("===========================")

