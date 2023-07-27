import os
import pickle


# save_urdf_path = "/home/baothach/shape_servo_data/rotation_extension/visualization/display_objects/urdf"
# object_meshes_path = "/home/baothach/shape_servo_data/rotation_extension/visualization/display_objects/mesh"
save_urdf_path = "/home/baothach/shape_servo_data/dynamics/urdf"
object_meshes_path = "/home/baothach/shape_servo_data/dynamics/mesh"

os.makedirs(save_urdf_path,exist_ok=True)

shape_name = "hemis"  #"box", "cylinder", "hemis"

density = 100
# youngs = 1e3
poissons = 0.3
scale = 0.5
attach_dist = 0.01




for i in range(2):

    object_name = shape_name + "_" + str(i)
    with open(os.path.join(object_meshes_path, object_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)

    youngs = 1000#round(data["youngs"])


    cur_urdf_path = save_urdf_path + '/' + shape_name + "_" + str(i) + '.urdf'
    
    f = open(cur_urdf_path, 'w')
    if True:
        urdf_str = """<?xml version="1.0" encoding="utf-8"?>        
    <robot name=\"""" + object_name + """\">
        <link name=\"""" + object_name + """\">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                <density value=\"""" + str(density) + """\" />
                <youngs value=\"""" + str(youngs) + """\"/>
                <poissons value=\"""" + str(poissons) + """\" />
                <damping value="0.0" />
                <attachDistance value=\"""" + str(attach_dist) + """\" />
                <tetmesh filename=\"""" + object_meshes_path + """/""" + str(object_name) + """.tet\" />
                <scale value=\"""" + str(scale) + """\"/>
            </fem>
        </link>

    </robot>
    """
        f.write(urdf_str)
        f.close()

