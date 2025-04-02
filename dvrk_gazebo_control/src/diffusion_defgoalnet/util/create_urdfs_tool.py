import os
import pickle

main_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_tool"
urdf_path = os.path.join(main_path, "urdf_tool")
os.makedirs(urdf_path,exist_ok=True)


scale = 1
num_object_per_category = 50



for object_idx in range(num_object_per_category):


    object_name = f"tool_{object_idx}"

    cur_urdf_path = urdf_path + '/' + object_name + '.urdf'
    f = open(cur_urdf_path, 'w')
    
    urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>            
<robot name="kidney">
<link name="kidney">
    <visual>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
        <mesh filename="../mesh_tool/{object_name}.obj" scale="{scale:.3f} {scale:.3f} {scale:.3f}"/>
    </geometry>
    </visual>
    <collision>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
        <mesh filename="../mesh_tool/{object_name}.obj" scale="{scale:.3f} {scale:.3f} {scale:.3f}"/>
    </geometry>
    </collision>
    <inertial>
    <mass value="5000"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
</link>
</robot>
    """

    f.write(urdf_str)
    f.close()

