import numpy as np
#!/usr/bin/env python3

def compute_goal(init_pose, goal_pose, num_cp, fractions):
    assert len(fractions) == num_cp
    # assert sum(fractions) == 1

    checkpoints = []
    for i in range(num_cp):
        checkpoints.append(init_pose + fractions[i]*(goal_pose-init_pose))
    checkpoints.append(goal_pose)
    return checkpoints

# init_pose = np.array([0,0,0,0,0,0])
# goal_pose = np.array([0.04,0.06,0.08,np.pi/3,np.pi/3,np.pi/3])
# cps = compute_goal(init_pose, goal_pose, num_cp=2, fractions=[2/3, 3/4])

# for cp in cps:
#     print(cp)
#     print("==============")  
