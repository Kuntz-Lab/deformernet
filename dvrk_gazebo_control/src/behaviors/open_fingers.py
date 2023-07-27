import math
import rospy

from ll4ma_isaacgym.behaviors import Behavior


class OpenFingers(Behavior):
    """
    Simple behavior to open the fingers all the way.

    TODO should add better checks that the fingers actually opened,
    right now there's essentially no failure case
    """

    def __init__(self, behavior_config, robot, env_config, *args):
        super().__init__(behavior_config, robot, env_config, *args)
        
        self._step = 0
        self.open_for_steps = self.robot.end_effector.config.open_for_steps

    def get_trajectory(self, action):
        action.set_ee_discrete('open')
        trajectory = [action] * self.open_for_steps
        return trajectory

    def get_action(self, state):
        if self.is_complete():
            return None

        if self.is_not_started():
            rospy.loginfo(f"Running behavior: {self.name.upper()}")
            self.set_in_progress()

        action = state.prev_action
        action.set_ee_discrete('open')

        # TODO temporary until better checks on result
        self._step += 1
        if self._step > self.open_for_steps:
            self.set_success()

        return action
