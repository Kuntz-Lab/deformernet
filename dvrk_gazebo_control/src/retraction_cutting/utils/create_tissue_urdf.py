import os
import pickle

object_mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting"
object_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/retraction_cutting"
base_mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting"
os.makedirs(object_urdf_path,exist_ok=True)


base_name = "base_2"
shape_name = "box_2"

density = 100
youngs = 1e3
poissons = 0.3
scale = 0.5
attach_dist = 0.001

# height = 0.45
# width = 0.4
thickness = 0.04    #0.04
base_thickness = 0.005


# for i in range(1):
for i in [1]:


    object_name = shape_name #+ "_" + str(i)

    cur_urdf_path = object_urdf_path + '/' + object_name + '.urdf'
    f = open(cur_urdf_path, 'w')
    if True:
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

