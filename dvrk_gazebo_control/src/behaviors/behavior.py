import sys
import rospy
from enum import Enum
from copy import deepcopy
from collections import OrderedDict

# from ll4ma_util import func_util

# from ll4ma_isaacgym.core.config import BEHAVIOR_MODULES  # for dynamic class creation


class BehaviorStatus(Enum):
    NOT_STARTED = 1
    IN_PROGRESS = 2
    SUCCESS = 3
    FAILURE = 4


class Behavior:
    """
    Base class for behaviors.

    This class does a lot of heavy lifting for hierarchical behaviors in terms
    of managing transitions between sub-behaviors and retrieving actions/status
    from the sub-behaviors.
    """

    def __init__(self):
        self.name = None



        self._prev_act = None
        self._wait_after_behavior_idx = 0

        self.set_not_started()

    def get_action(self, state):
        """
        Returns the next action to apply in the simulator given the current
        state of the simulator.

        Args:
            state (EnvironmentState): Current state of the simulator
        Returns:
            action (Action): Next action to apply in the simulator
        """
        pass
        # if self._open_loop and self.is_not_started():
        #     state = deepcopy(state)
        #     # for behavior in self.behaviors.values():
        #     behavior.set_policy(state)
        #     behavior.override_state(state) # Need to set start points as plan endpoints


        # if self.is_complete():
        #     return None

        # # Initialize next behavior if none is active
        # if self._current_behavior is None:
        #     self._current_behavior = list(self.behaviors.values())[self._current_behavior_idx]

        # act = self._current_behavior.get_action(state)
        # self._prev_act = deepcopy(act)
        
        # return act

    def set_policy(self, state):
        """
        Base implementation is a no-op, some behaviors maybe require not initialization.
        """
        pass

    def override_state(self, state):
        """
        Base implementation is a no-op, some behaviors will override the state values for
        generating open loop behavior sequences (e.g. sequence of motion plans will set
        the start joint position of the second plan to be the end position of the first plan).
        """
        pass

    def is_complete(self):
        return self._status in [BehaviorStatus.SUCCESS, BehaviorStatus.FAILURE]

    def is_complete_success(self):
        return self._status == BehaviorStatus.SUCCESS

    def is_complete_failure(self):
        return self._status == BehaviorStatus.FAILURE

    def is_in_progress(self):
        return self._status == BehaviorStatus.IN_PROGRESS

    def is_not_started(self):
        return self._status == BehaviorStatus.NOT_STARTED

    def set_success(self):
        self._set_status(BehaviorStatus.SUCCESS)

    def set_failure(self):
        self._set_status(BehaviorStatus.FAILURE)

    def set_in_progress(self):
        self._set_status(BehaviorStatus.IN_PROGRESS)

    def set_not_started(self):
        self._set_status(BehaviorStatus.NOT_STARTED)
    


    def _set_status(self, status=None):
        """
        Maintaining internal _status of type BehaviorStatus to track state machine, and then
        public status that appends behavior name and informs parent behaviors what's going on.

        TODO this gets kind of confusing, can probably make this much easier to follow
        """
        if status is not None:
            self._status = status




