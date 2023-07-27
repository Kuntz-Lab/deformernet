import os

save_urdf_path = "/home/baothach/sim_data/BigBird/BigBird_urdf_new"
object_meshes_path = "/home/baothach/sim_data/BigBird/BigBird_mesh"

density = 100
youngs = 1e3
poissons = 0.3
scale = 0.5

# object_name = "cup_noodles_chicken"
for object_name in os.listdir(object_meshes_path):
    cur_urdf_path = save_urdf_path + '/' + object_name + '.urdf'
    f = open(cur_urdf_path, 'w')
    if True:
        urdf_str = """
    <robot name=\"""" + object_name + """\">
        <link name=\"""" + object_name + """_link">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                <density value=\"""" + str(density) + """\" />
                <youngs value=\"""" + str(youngs) + """\"/>
                <poissons value=\"""" + str(poissons) + """\" />
                <damping value="0.0" />
                <attachDistance value="0.0" />
                <tetmesh filename=\"/home/baothach/sim_data/BigBird/BigBird_mesh/""" + str(object_name) + """/meshes/poisson.tet\" />
                <scale value=\"""" + str(scale) + """\"/>
            </fem>
        </link>
    </robot>
    """
        f.write(urdf_str)
        f.close()

