from isaacgym import gymtorch, gymapi

import sys
import torch
import rospy
from tqdm import tqdm

from ll4ma_isaacgym.behaviors import Behavior
from ll4ma_isaacgym.core import Simulator
from ll4ma_util import ui_util, file_util


class Session:
    """
    Manages simulator execution, including managing behaviors, data collection,
    stepping the simulator, etc.
    """

    def __init__(self, session_config):
        self.config = session_config
        self.simulator = Simulator(self.config)
        self._step_idx = 0
        self._extra_step_idxs = [0] * self.config.n_envs
        self._reset_behaviors()

    def run(self):
        """
        Main entry point for running session.
        """
        if self.config.run_forever:
            while not rospy.is_shutdown():
                self.step()
        elif not self.config.data_root:
            for _ in range(self.config.n_demos):
                if rospy.is_shutdown():
                    break
                self._run()
                self.reset()
        else:
            n_demos = len(file_util.list_dir(self.config.data_root, '.pickle'))
            pbar = tqdm(total=self.config.n_demos - n_demos, file=sys.stdout)
            while not rospy.is_shutdown() and n_demos < self.config.n_demos:
                self._run()
                if not rospy.is_shutdown():
                    self.simulator.save_data(pbar)
                    self.reset()
                    n_demos = len(file_util.list_dir(self.config.data_root, '.pickle'))
            if not rospy.is_shutdown():
                ui_util.print_happy("\n\nData collection complete\n")
                print(f"  {self.config.data_root} has {n_demos} demos\n")

    def _run(self):
        if self.config.n_steps > 0:
            step = 0
            while not rospy.is_shutdown() and step < self.config.n_steps:
                self.step()
                step += 1
        else:
            while not rospy.is_shutdown() and not self.is_complete():
                self.step()

    def step(self, pre_physics=True, post_physics=True, increment_step_idx=True):
        """
        Perform one step in the task including applying actions, stepping physics,
        and computing observations.
        """
        # Apply actions
        if pre_physics:
            self._pre_physics_step()

        self.simulator.step(post_physics)

        for env_idx in range(self.config.n_envs):
            # We don't need to further log any data for envs that are finished
            if self.behaviors[env_idx].is_complete():
                if self.config.task.extra_steps > 0:
                    # Account for extra steps that are being recorded after task completion
                    self._extra_step_idxs[env_idx] += 1
                    extra_complete = self._extra_step_idxs[env_idx] >= self.config.task.extra_steps
                    self.simulator.should_log[env_idx] = not extra_complete
                else:
                    self.simulator.should_log[env_idx] = False

        if increment_step_idx:
            self._step_idx += 1 # Used mainly to add step buffer at start

    def _pre_physics_step(self):
        """
        Perform operations prior to a simulator step. Primarily computes actions from behaviors.
        """
        for env_idx in range(self.config.n_envs):
            if self.behaviors[env_idx].is_complete():
                continue
            env_state = self.simulator.get_env_state(env_idx)
            action = self.behaviors[env_idx].get_action(env_state)
            if action is not None:
                self.simulator.apply_action(action, env_idx)

    def is_complete(self):
        """
        Determines when this task is complete.
        """
        behaviors_complete = all([b.is_complete() for b in self.behaviors])
        extra_steps_complete = all([s >= self.config.task.extra_steps for s in self._extra_step_idxs])
        return behaviors_complete and extra_steps_complete
        
    def reset(self):
        """
        Resets the environments, behaviors, and data caches.
        """
        self.simulator.reset()
        self._reset_behaviors()
        self._step_idx = 0
        self._extra_step_idxs = [0] * self.config.n_envs

    def _reset_behaviors(self):
        """
        Resets the behaviors for each env.
        """
        self.behaviors = [
            Behavior(self.config.task.behavior, self.simulator.robot, self.config.env,
                     self.simulator, self.config.device, self.config.open_loop)
            for _ in range(self.config.n_envs)
        ]
