# -------------------------------------------------------------------------------------------------
# Copyright (C) Microsoft Corporation.  All rights reserved.
# -------------------------------------------------------------------------------------------------

'''Defines the environment interface.'''


class Environment(object):
    '''Defines the environment interface. Inheriting environment classes are expected to
    implement all inherited functions.'''

    def __init__(self, params):
        pass

    def record(self, filename, params):
        '''Set up recording.'''
        raise NotImplementedError

    def start_mission(self, params):
        '''Initialize and start a single mision (mission in Malmo == episode in RL) according
        to the received parameters. When this method returns, agents that interact with this
        environment assume that a mission is ready for interaction. Raise an exception when
        something goes wrong.
        '''
        raise NotImplementedError

    def get_action_set(self, params):
        '''Return the action set for this environment.'''
        raise NotImplementedError

    def act(self, action):
        '''Send a single action to be executed in the environment. Raise exceptions when something goes wrong.'''
        raise NotImplementedError

    def is_mission_running(self):
        '''Return True if the mission is running, False otherwise.'''
        raise NotImplementedError

    def get_world_state(self, params):
        '''Request the next world state.'''
        raise NotImplementedError

