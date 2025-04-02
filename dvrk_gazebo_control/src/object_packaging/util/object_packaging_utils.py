import numpy as np
from copy import deepcopy

def compute_balls_new_positions(desired_delta_pos_1, desired_delta_pos_2,
                          ball_frame_count, delta_pos_per_frame=0.001):
    max_steps_1 = round(np.linalg.norm(desired_delta_pos_1) / delta_pos_per_frame)
    max_steps_2 = round(np.linalg.norm(desired_delta_pos_2) / delta_pos_per_frame)
    
    # new_delta_pos_1, new_delta_pos_2 = None, None
    new_delta_pos_1, new_delta_pos_2 = desired_delta_pos_1, desired_delta_pos_2
    done = True
    
    if ball_frame_count < max_steps_1:
        new_delta_pos_1 = desired_delta_pos_1 / np.linalg.norm(desired_delta_pos_1) * delta_pos_per_frame * ball_frame_count
        done = False    
    if ball_frame_count < max_steps_2:
        new_delta_pos_2 = desired_delta_pos_2 / np.linalg.norm(desired_delta_pos_2) * delta_pos_per_frame * ball_frame_count
        done = False
    
    return new_delta_pos_1, new_delta_pos_2, done

def get_balls_current_positions(ball_state):
    num_objects = len(ball_state['pose']['p']['x'])
    ball_pos = np.zeros((num_objects, 3))
    for i in range(num_objects):
        ball_pos[i] = np.array([ball_state['pose']['p']['x'][i],
                                ball_state['pose']['p']['y'][i],
                                ball_state['pose']['p']['z'][i]])  
    print("ball_state: ", ball_state)
    print("\n========\n")
    print("ball_pos: ", ball_pos)
    print("\n========\n")                              
    return ball_pos

def move_two_balls(gym, env, object_handle, ball_state, new_delta_pos_1, new_delta_pos_2):    
    from isaacgym import gymapi
    copied_ball_state = deepcopy(ball_state)
    if new_delta_pos_1 is not None:       
        copied_ball_state['pose']['p']['x'][0] += new_delta_pos_1[0]
        copied_ball_state['pose']['p']['y'][0] += new_delta_pos_1[1]
        copied_ball_state['pose']['p']['z'][0] += new_delta_pos_1[2]

    if new_delta_pos_2 is not None:
        copied_ball_state['pose']['p']['x'][1] += -new_delta_pos_2[0]
        copied_ball_state['pose']['p']['y'][1] += -new_delta_pos_2[1]
        copied_ball_state['pose']['p']['z'][1] += new_delta_pos_2[2]

    gym.set_actor_rigid_body_states(env, object_handle, copied_ball_state, gymapi.STATE_ALL)





        