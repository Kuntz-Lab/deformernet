import os
import pickle


save_urdf_path = "/home/baothach/shape_servo_data/TAMP/object_data/urdf"
object_meshes_path = "/home/baothach/shape_servo_data/TAMP/object_data/mesh"

os.makedirs(save_urdf_path,exist_ok=True)


# shape_name = "cylinder"

density = 100
youngs = "3e3"
poissons = 0.3
scale = 0.5
attach_dist = 0.


for i in range(0,1):


    object_name = "cylinder"

    # with open(os.path.join(object_meshes_path, object_name + ".pickle"), 'rb') as handle:
    #     data = pickle.load(handle)

    # height = data["height"]
    # width = data["width"]
    # thickness = data["thickness"]
    # youngs = round(data["youngs"])

    cur_urdf_path = f"{save_urdf_path}/{object_name}_soft.urdf"
    
    
    f = open(cur_urdf_path, 'w')
    if True:
        urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>      
    <robot name=\"""" + object_name + """\">
        <link name=\"""" + object_name + """\">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                <density value=\"""" + str(density) + """\" />
                <youngs value=\"""" + youngs + """\"/>
                <poissons value=\"""" + str(poissons) + """\" />
                <damping value="0.0" />
                <attachDistance value=\"""" + str(attach_dist) + """\" />
                <tetmesh filename=\"../mesh/""" + str(object_name) + """.tet\" />
                <scale value=\"""" + str(scale) + """\"/>
            </fem>
        </link>
    </robot>
    """
        f.write(urdf_str)
        f.close()

