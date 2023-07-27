if __name__ == "__main__":

    # Create Gym object and set_up environment 

    # Load robot asset

    #  Load object asset from Set random pose for object 

    # for i in range(num_envs):

    dc_client = GraspDataCollectionClient()
    dataset_name = 'BigBird'
    base_dir = '/home/mohanraj/sim_data/'
    dataset_dir = base_dir + dataset_name \
                    + '/' + dataset_name + '_mesh'
    object_mesh_dirs = os.listdir(dataset_dir)

    object_mesh_dirs = object_mesh_dirs[::-1]

    # Bigbird objects that don't work for gazebo+dart.
    bigbird_objects_exclude = {'coca_cola_glass_bottle', 'softsoap_clear',
                               'softsoap_green', 'softsoap_purple', 'windex',
                               'palmolive_orange', 'palmolive_green', 'cinnamon_toast_crunch',
                               #'progresso_new_england_clam_chowder', 
                               'listerine_green', 'hersheys_bar', 'mahatma_rice'}
    for i, object_name in enumerate(object_mesh_dirs):
        rospy.loginfo('Object: %s' %object_name)

        dc_client.get_last_object_id_name()
        # Resume from the last object.
        if dc_client.last_object_name == 'empty' or \
                object_name == dc_client.last_object_name:
            skip_object = False
        if skip_object:
            continue

        if dataset_name == 'BigBird' and object_name in bigbird_objects_exclude:
            continue

        if dataset_name == 'BigBird':
            obj_mesh_path = dataset_dir + '/' + object_name + \
                            '/textured_meshes/optimized_tsdf_textured_mesh.stl'
        elif dataset_name == 'GraspDatabase':
            obj_mesh_path = dataset_dir + '/' + object_name + \
                            '/' + object_name + '_proc.stl' 


    while not all done:
        if state = "start":
            set_actor_state(object, random_pose)      
            set_pos_target(robot, home_pos)     
            if check_reach_desired_position(i, pos_targets) == True:
                state = "get_preshape"

        if state = "get_preshape":
            points = get_point_cloud()
            generate_preshape(points)       

        if state ==   