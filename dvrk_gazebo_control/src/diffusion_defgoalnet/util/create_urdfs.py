import os
import pickle

main_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_tool"
urdf_path = os.path.join(main_path, "urdf")
os.makedirs(urdf_path,exist_ok=True)
mesh_path = os.path.join(main_path, "mesh")

density = 100
youngs = "1e3"
poissons = 0.3
scale = 1
attach_dist = 0.01

num_object_per_category = 100   #50

categoies = ["cylinder"] #["ellipsoid", "cylinder"]

for category in categoies:
    for object_idx in range(num_object_per_category):


        object_name = f"{category}_{object_idx}"
        base_name = f"{category}_{object_idx}_base" # 

        with open(os.path.join(mesh_path, f"{object_name}_info.pickle"), 'rb') as handle:
            data = pickle.load(handle)
            height = data["height"]
            radius = data["radius"]

        cur_urdf_path = urdf_path + '/' + object_name + '.urdf'
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
                    <tetmesh filename="../mesh/{object_name+".tet"}"/>
                    <scale value="{scale}"/>
                </fem>
            </link>


        <link name="fix_frame">
            <visual>
                <origin xyz="10.0 {radius} {0}"/>              
                <geometry>
                    <box size="0.05 {0.005} {0.005}"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz="0.0 {radius} {0}"/>              
                <geometry>
                    <box size="0.05 {0.005} {0.005}"/>
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

            # <link name="fix_frame">
            #     <visual>
            #         <origin xyz="0.0 0.0 {-0.0025:.3f}"/>               
            #         <geometry>
            #             <box size="0.15 0.15 0.0025"/>
            #         </geometry>
            #     </visual>
            #     <collision>
            #         <origin xyz="0.0 0.0 {0.0}"/>             
            #         <geometry>
            #             <box size="0.15 0.15 0.0025"/>
            #         </geometry>
            #     </collision>
            #     <inertial>
            #         <mass value="500000"/>
            #         <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
            #     </inertial>
            # </link>            

            # <joint name = "attach" type = "fixed">
            #     <origin xyz = "0.0 0.0 0.0" rpy = "0 0 0"/>
            #     <parent link ="{object_name}"/>
            #     <child link = "fix_frame"/>
            # </joint>  