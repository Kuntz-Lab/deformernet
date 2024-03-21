import os
import pickle

main_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_cutting"
urdf_path = os.path.join(main_path, "urdf")
# mesh_path = os.path.join(main_path, "mesh")
os.makedirs(urdf_path,exist_ok=True)

shape_name = "cylinder"

density = 100
youngs = "1e4"
poissons = 0.3
scale = 1
attach_dist = 0.01



for i in range(0,1):
    object_name = f"{shape_name}_{i}"
    base_name = f"{shape_name}_{i}_base" # 

    # with open(os.path.join(mesh_path, object_name + ".pickle"), 'rb') as handle:
    #     data = pickle.load(handle)

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
                <origin xyz="0.0 0.0 {0.0:.3f}"/>              
                <geometry>
                    <mesh filename="../mesh/{base_name+".obj"}" scale="{scale} {scale} {scale}"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz="0.0 0.0 {0.0}"/>           
                <geometry>
                    <mesh filename="../mesh/{base_name+".obj"}" scale="{scale} {scale} {scale}"/>
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

