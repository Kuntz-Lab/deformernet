import os
import pickle

stiffness = "1k"
urdf_path = f"/home/baothach/sim_data/Custom/Custom_urdf/physical_CAO/single/multi_box_{stiffness}Pa"
os.makedirs(urdf_path,exist_ok=True)

mesh_path = f"/home/baothach/sim_data/Custom/Custom_mesh/physical_CAO/multi_box_{stiffness}Pa"
mesh_relative_path = "/".join(mesh_path.split("/")[-3:])

shape_name = "box"

density = 100
# youngs = 1e3
poissons = 0.3
scale = 1.0
attach_dist = 0.01
plate_thickness = 0.002


with open(os.path.join(mesh_path, "primitive_dict_box.pickle"), 'rb') as handle:
    data = pickle.load(handle)

for i in range(1):    # 100

    object_name = shape_name + "_" + str(i)
    height = data[object_name]["height"] / 2
    width = data[object_name]["width"] / 2
    thickness = data[object_name]["thickness"]
    youngs = round(data[object_name]["youngs"])

    cur_urdf_path = urdf_path + '/' + shape_name + "_" + str(i) + '.urdf'
    
    f = open(cur_urdf_path, 'w')
    urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>    
    
    <robot name="{object_name}">
        <link name="{object_name}">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 {plate_thickness}" />
                <density value="{density}" />
                <youngs value="{youngs}"/>
                <poissons value="{poissons}"/>
                <damping value="0.0" /> 
                <attachDistance value="{attach_dist}"/>
                <tetmesh filename="../../../../{mesh_relative_path}/{object_name+".tet"}"/>
                <scale value="{scale}"/>
            </fem>
        </link>
        
        <link name="fix_frame">
            <visual>
                <origin xyz="0.0 {0.0} {-thickness*scale/2 + plate_thickness/2:.3f}"/>              
                <geometry>
                    <box size="{height*scale} {width*scale} {plate_thickness}"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz="0.0 {0.0} {-thickness*scale/2 + plate_thickness/2:.3f}"/>              
                <geometry>
                    <box size="{height*scale} {width*scale} {plate_thickness}"/>
                </geometry>
            </collision>
            <inertial>
                <mass value="500000"/>
                <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
            </inertial>
        </link>
        
        <joint name = "attach" type = "fixed">
            <origin xyz = "0.0 0.0 0.0" rpy = "0 0 0"/>
            <parent link = "{object_name}"/>
            <child link = "fix_frame"/>
        </joint>  
        
    </robot>
    """

    f.write(urdf_str)
    f.close()

