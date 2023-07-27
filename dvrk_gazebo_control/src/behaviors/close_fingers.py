import math
import rospy
import numpy as np
from ll4ma_isaacgym.behaviors import Behavior


class CloseFingers(Behavior):
    """
    Simple behavior to close the fingers to grasp something.

    TODO would be good to add force monitoring or better state checking to
    ensure it's completed the close action, right now it just repeats the
    action for a set number of timesteps
    """

    def __init__(self, behavior_config, robot, env_config, sim, *args):
        super().__init__(behavior_config, robot, env_config, sim, *args)

        # TODO these are temporary until I get force monitoring working
        self._step = 0
        self.close_for_steps = self.robot.end_effector.config.close_for_steps

    def get_trajectory(self, action):
        action.set_ee_discrete('close')
        trajectory = [action] * self.close_for_steps
        return trajectory

    def get_action(self, state):
        if self.is_complete():
            return None

        if self.is_not_started():
            rospy.loginfo(f"Running behavior: {self.name.upper()}")
            self.set_in_progress()

        action = state.prev_action
        action.set_ee_discrete('close')

        # TODO temporary until force monitoring works
        self._step += 1
        if self._step > self.close_for_steps:
            self.set_success()

        return action
