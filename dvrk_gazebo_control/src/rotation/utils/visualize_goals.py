import open3d
import pickle5 as pickle
import os

prim_name, stiffness = "hemis", "5k"
object_category = f"{prim_name}_{stiffness}"
distribution_keyword = "inside"
main_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_{object_category}Pa/evaluate"
goal_recording_path = os.path.join(main_path, "goal_data", object_category, distribution_keyword)

# for obj_name in [15, 71, 62, 24, 50, 56, 58, 65, 41, 21]:
for obj_name in range(100):
    print("obj_name:", obj_name)
    with open(os.path.join(goal_recording_path, f"{prim_name}_{obj_name}.pickle"), 'rb') as handle:
        goal_datas = pickle.load(handle) 
    goal_pc_numpy = goal_datas["partial pcs"][1]   # first goal pc

    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy)  
    pcd_goal.paint_uniform_color([1,0,0]) 
    open3d.visualization.draw_geometries([pcd_goal])