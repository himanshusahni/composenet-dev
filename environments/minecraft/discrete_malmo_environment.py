# -------------------------------------------------------------------------------------------------
# DiscreteMalmoEnvironment conducts additional checks for predictable navigational
# tasks with discrete actions. It does these checks for basic movement (on a single
# plane, so no jumping). It assumes that only air is walkable, and that there are no
# head-high obstacles. Given these assumptions, it guarantees that discrete movement
# commands are acted on before the next world state is observed.
#
# Copyright (C) Microsoft Corporation.  All rights reserved.
# -------------------------------------------------------------------------------------------------

import json
import math
import re

from discrete_stochastic_malmo_environment import DiscreteStochasticMalmoEnvironment


# we can be generous with accuracy, 0.1 is only a tenth of a block
def almost_equal(value_1, value_2, accuracy = 0.1):
    return abs(value_1 - value_2) < accuracy



class DiscreteMalmoEnvironment(DiscreteStochasticMalmoEnvironment):

    def __init__(self, parameters):
        '''Initialize the malmo environment.'''
        super(DiscreteMalmoEnvironment, self).__init__(parameters)

    def _prepare_mission(self, params):
        '''
        Overrides _prepare_mission to run additional preparation on the mission.
        '''
        super(DiscreteMalmoEnvironment, self)._prepare_mission(params)

         # request 3x2x3 grid observation at and below the player's feet
        self.my_mission.observeGrid(-1, -1, -1, 1, 0, 1, "Grid")
        self._set_previous_position(0, 0, 0)
        self._set_previous_yaw(-10.0)

    def act(self, action):
        '''Override act, send action and store it.'''
        # set the previous position / yaw before taking an action
        self._update_state()
        super(DiscreteMalmoEnvironment, self).act(action)

    def _update_state(self):
        if self.world_state is None:
            return

        if len(self.world_state.observations) > 0:
            obs = json.loads(self.world_state.observations[-1].text)

            self._set_previous_position(obs['XPos'], obs['YPos'],
                                        obs['ZPos'])
            self._set_previous_yaw(obs['Yaw'])

    def is_valid_world_state(self, world_state):
        '''
        Valid states for discrete actions: contain observation, reward, and
        show enough movement unless the path along the selected direction of
        movement is blocked. Assumes discrete movement actions.
        '''
        valid = super(DiscreteMalmoEnvironment, self).\
            is_valid_world_state(world_state)
        if not valid:
            return False

        # print errors, just for information
        for error in world_state.errors:
            print "Error:", error.text

        # check if this is the first step
        if not self._get_previous_action():
            return True

        obs = json.loads(world_state.observations[-1].text)

        # check path: assumes grid observation with name "Grid",
        # 3x3 around agent's feet
        if 'Grid' not in obs:
            # self.valid_counter += 1
            # print "grid is not in obs {}".format(self.valid_counter)
            return False

        if 'use' in self._get_previous_action():
            if not self._is_target_in_player_bounds(obs['Grid']):
                return True # cannot actually place the block
            if self._is_path_blocked(obs['Grid']):
                # either there was already a block, or we've placed it
                return True
            return False

        if 'attack' in self._get_previous_action():
            if not self._is_target_in_player_bounds(obs['Grid']):
                return True # cannot destroy the block
            if self._is_path_blocked(obs['Grid']):
                # self.valid_counter += 1
                # print "path is blocked {}".format(self.valid_counter)
                return False
            return True

        if 'turn' in self._get_previous_action():
            if almost_equal(self._get_previous_yaw(), obs['Yaw']):
                # if Yaw has not updated, world state is not ready
                # self.valid_counter += 1
                # print "yaw has not updated {}".format(self.valid_counter)
                return False

        if 'move' in self._get_previous_action():
            direction = int(self._get_previous_action().split()[1])
            if self._is_path_blocked(obs['Grid'], direction):
                # got non-zero reward + don't have to wait for movement
                return True
            else:
                # check distance moved
                distance_2D = math.sqrt(
                    (self.get_previous_position()['x'] - obs['XPos'])**2 +
                    (self.get_previous_position()['z'] - obs['ZPos'])**2)
                if distance_2D < 0.7:
                    # self.valid_counter += 1
                    # print "distance 2d is less than 0.7 {}".format(self.valid_counter)
                    return False

        if 'tp ' in self._get_previous_action():
            x, y, z = map(float, re.findall(r'[+-]?[0-9.]+', self._get_previous_action()))
            distance_2D = math.sqrt(
                (self.get_previous_position()['x'] - x)**2 +
                (self.get_previous_position()['y'] - y)**2 +
                (self.get_previous_position()['z'] - z)**2)
            if distance_2D < 0.1:
                return False

        return True

    def _is_target_in_player_bounds(self, grid):

        # we can build on these materials
        building_ground = ['lapis_block', 'sandstone']

        if grid[7] not in building_ground and \
                almost_equal(self._get_previous_yaw(), 0.0): # facing south
            return False
        if grid[1] not in building_ground and \
                almost_equal(self._get_previous_yaw(), 180.0) or \
                almost_equal(self._get_previous_yaw(), -180.0): # facing north
            return False
        if grid[3] not in building_ground and \
                almost_equal(self._get_previous_yaw(), 90.0): # facing west
            return False
        if grid[5] not in building_ground and \
                almost_equal(self._get_previous_yaw(), 270.0) or \
                almost_equal(self._get_previous_yaw(), -90.0): # facing east
            return False

        return True

    def _is_path_blocked(self, grid, direction = 1):
        '''Detect whether the path is free. This assumes a 3x2x3 observation
           centered at the agent's feet, i.e., for a player that faces north:
           [ 0][ 1][ 2]
           [ 3][ 4][ 5]
           [ 6][ 7][ 8] # ground
           ------------
           [ 9][10][11] # player's feet
           [12][13][14]
           [15][16][17]
           Note: assumes that only air is walkable, and that there are no
                 head-high obstacles.
        '''

        # move forward or place a block in front of the agent
        if ('move' in self._get_previous_action() and direction == 1) \
                or 'use' in self._get_previous_action() \
                or 'attack' in self._get_previous_action():
            if almost_equal(self._get_previous_yaw(), 0.0) and grid[16] != 'air':
                return True
            if (almost_equal(self._get_previous_yaw(), 180.0) or
                almost_equal(self._get_previous_yaw(), -180.0)) and grid[10] != 'air':
                return True
            if almost_equal(self._get_previous_yaw(), 90.0) and grid[12] != 'air':
                return True
            if (almost_equal(self._get_previous_yaw(), 270.0) or
                almost_equal(self._get_previous_yaw(), -90.0)) and grid[14] != 'air':
                return True

        # move back
        if ('move' in self._get_previous_action() and direction == -1):
            if almost_equal(self._get_previous_yaw(), 0.0) and grid[10] != 'air':
                return True
            if (almost_equal(self._get_previous_yaw(), 180.0) or
                almost_equal(self._get_previous_yaw(), -180.0)) and grid[16] != 'air':
                return True
            if almost_equal(self._get_previous_yaw(), 90.0) and grid[14] != 'air':
                return True
            if (almost_equal(self._get_previous_yaw(), 270.0) or
                almost_equal(self._get_previous_yaw(), -90.0)) and grid[12] != 'air':
                return True

        if 'north' in self._get_previous_action() and grid[10] != 'air':
            return True
        if 'west' in self._get_previous_action() and grid[12] != 'air':
            return True
        if 'east' in self._get_previous_action() and grid[14] != 'air':
            return True
        if 'south' in self._get_previous_action() and grid[16] != 'air':
            return True
        return False

    def _set_previous_yaw(self, yaw):
        self._yaw = yaw

    def _get_previous_yaw(self):
        return self._yaw

    def _set_previous_position(self, x, y, z):
        '''Private method to update agent position.'''
        self._prev_position = {'x': x, 'y': y, 'z': z}

    def get_previous_position(self):
        '''Helper function: get previous position.'''
        if not hasattr(self, "_prev_position"):
            return None
        return self._prev_position

