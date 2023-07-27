import os
import pickle

# object_mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting/multi_box"
# base_mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting/multi_box"
# object_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/retraction_cutting/multi_box"

object_mesh_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/Custom_mesh_2"
base_mesh_path = object_mesh_path
object_urdf_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/Custom_urdf_2"

os.makedirs(object_urdf_path,exist_ok=True)

density = 100
poissons = 0.3
scale = 0.5
attach_dist = 0.001

base_thickness = 0.005



with open(os.path.join(object_mesh_path, "primitive_dict.pickle"), 'rb') as handle:
    data = pickle.load(handle)

for i in range(10):
# for i in [1]:

    object_name = f"box_{i}"
    base_name = f"base_{i}"
    height = data[object_name]["height"]
    width = data[object_name]["width"]
    thickness = data[object_name]["thickness"]
    youngs = 2000   #round(data[object_name]["youngs"])    

   
    cur_urdf_path = object_urdf_path + '/' + object_name + '.urdf'
    f = open(cur_urdf_path, 'w')
    
    urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>    
    
    <robot name="{object_name}">
        <link name="{object_name}">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                <density value="{density}" />
                <youngs value="{youngs}"/>
                <poissons value="{poissons}"/>
                <damping value="0.0" />
                <attachDistance value="{attach_dist}"/>
                <tetmesh filename="{os.path.join(object_mesh_path, object_name+".tet")}"/>
                <scale value="{scale}"/>
            </fem>
        </link>

        <link name="fix_frame">
            <visual>
                <origin xyz="0.0 0.0 {-(thickness+base_thickness)*scale/2:.3f}"/>              
                <geometry>
                    <mesh filename="{os.path.join(base_mesh_path, base_name+".obj")}" scale="{scale} {scale} {scale}"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz="0.0 0.0 {-(thickness+base_thickness)*scale/2:.3f}"/>           
                <geometry>
                    <mesh filename="{os.path.join(base_mesh_path, base_name+".obj")}" scale="{scale} {scale} {scale}"/>
                </geometry>
            </collision>
            <inertial>
                <mass value="500000"/>
                <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
            </inertial>
        </link>
        
        <joint name = "attach" type = "fixed">
            <origin xyz = "0.0 0.0 0.0" rpy = "0 0 0"/>
            <parent link ="{object_name}"/>
            <child link = "fix_frame"/>
        </joint>  




    </robot>
    """
    
    f.write(urdf_str)
    f.close()

