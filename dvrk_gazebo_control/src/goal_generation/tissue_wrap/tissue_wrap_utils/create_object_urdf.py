import os
import numpy as np


density = 100
youngs = "3e3"
poissons = 0.3
attach_dist = 0.0
scale = 1




for object_category in ["cylinder", "tissue"]:
    for i in range(200):

        object_urdf_path = f"/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/objects_dataset_eval/{object_category}/urdf"
        os.makedirs(object_urdf_path, exist_ok=True)

        object_name = f"{object_category}_{i}"

        
        cur_urdf_path = object_urdf_path + f"/{object_name}.urdf"
        f = open(cur_urdf_path, 'w')

        if object_category == "tissue":
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
            <tetmesh filename="{"../mesh/" + object_name + ".tet"}"/>
            <scale value="{scale}"/>
        </fem>   
    </link>

</robot>
"""

        elif object_category == "cylinder":
            urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>            
<robot name="{object_name}">
<link name="{object_name}">
    <visual>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
        <mesh filename="{"../mesh/" + object_name + ".obj"}" scale="{scale:.3f} {scale:.3f} {scale:.3f}"/>
    </geometry>
    </visual>
    <collision>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
        <mesh filename="{"../mesh/" + object_name + ".obj"}" scale="{scale:.3f} {scale:.3f} {scale:.3f}"/>
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


