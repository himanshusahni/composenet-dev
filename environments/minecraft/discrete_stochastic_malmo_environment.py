# -------------------------------------------------------------------------------------------------
# DiscreteStochasticMalmoEnvironment does basic checks on discrete actions, but does not
# verify movement, or more complex discrete actions. This can lead to stochasticity
# in the environment experienced by agents.
# It is currently the only environment that is compatible with discrete building,
# jumping, etc.
# Some strange effects such as the rewards or observations being received or acted
# upon asynchronously are possible if Minecraft is overclocked beyond synchronicity.
#
# Copyright (C) Microsoft Corporation.  All rights reserved.
# -------------------------------------------------------------------------------------------------

from malmo_environment import MalmoEnvironment


class DiscreteStochasticMalmoEnvironment(MalmoEnvironment):

    def __init__(self, parameters):
        '''Initialize the malmo environment.'''
        super(DiscreteStochasticMalmoEnvironment, self).__init__(parameters)

    def _prepare_mission(self, params):
        '''
        Overrides _prepare_mission to run additional preparation on the mission.
        '''
        super(DiscreteStochasticMalmoEnvironment,
              self)._prepare_mission(params)

    def is_valid_world_state(self, world_state):
        '''
        Valid states for discrete actions: contain observation, reward.
        '''
        valid = super(DiscreteStochasticMalmoEnvironment,
                      self).is_valid_world_state(world_state)
        if not valid:
            return False

        # check if this is the first step
        if not self._get_previous_action():
            return True

        # rewards cannot be empty, unless it is the first step (checked above)
        if len(world_state.rewards) < 1:
            # self.valid_counter += 1
            # print "rewards are empty {}".format(self.valid_counter)
            return False
        has_nonzero_reward = False
        for r in world_state.rewards:
            if r.getValue() != 0:
                has_nonzero_reward = True
                break
        if not has_nonzero_reward:
            # self.valid_counter += 1
            # print "rewards are not non zero {}".format(self.valid_counter)
            return False

        return True

