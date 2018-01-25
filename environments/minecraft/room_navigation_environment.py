# -------------------------------------------------------------------------------------------------
# A custom extension to DiscreteMalmoEnvironmentWithVideo that adds lava blocks to a 'room maze'.
#
# Copyright (C) Microsoft Corporation.  All rights reserved.
# -------------------------------------------------------------------------------------------------

import random
import itertools
from copy import deepcopy
import math
import MalmoPython
import os
import json

from malmo_environment import MalmoEnvironment

def pop_sample(L, n):
    '''
    samples n elements from list L and removes them
    '''
    samples = [L.pop(random.randrange(len(L))) for _ in xrange(n)]
    return samples, L


class RoomNavigationEnvironment(MalmoEnvironment):
    '''
    class for running general discrete, grid room navigation tasks
    Assumes grid size is specified in mission xml somehow.
    For now the hack is it is in the mission description as the first number
    before the hyphen.
    '''
    def __init__(self, params):
        '''
        sets the grid width and height
        '''
        super(RoomNavigationEnvironment, self).__init__(params)
        self.grid_width = 8
        self.grid_length = 8
        self.tolerance = 0.1

    def _load_mission_xml(self, params):
        """
        draw any objects if given in params and set agent spawn location
        """
        all_positions = [(x,y) for x in range(self.grid_width) \
            for y in range(self.grid_length)]

        self.my_mission = None
        mission_def = params.environment.mission_xml
        assert type(mission_def) in [unicode, str], \
            "Mission XML must be a JSON array or str or unicode but is {}".\
            format(type(mission_def))
        mission_def = str(mission_def)
        with open(mission_def, 'r') as f:
            print "Loading mission from %s" % params.environment.mission_xml
            mission_xml = f.readlines()

        # termination criteria
        if hasattr(params.environment, "blocks"):
            for block in params.environment.blocks:
                mission_xml.insert(
                    52,
                    '        <Block type="{}" />\n'.format(str(block)))

        # what is the reward structure
        if hasattr(params.environment, "to_reward"):
            to_reward = params.environment.to_reward
            self.collect_reward = False
            # add block rewards first
            for tr in to_reward:
                if tr in params.environment.blocks:
                    mission_xml.insert(
                        49,
                        '        <Block reward="1.0" type="{}" behaviour="onceOnly"/>\n'.format(tr))
            # now add object rewards
            for tr in to_reward:
                if tr in params.environment.objects:
                    mission_xml.insert(
                        46,
                        '      <RewardForCollectingItem>\n'+\
                        '        <Item type="{}" reward="0.5"/>\n'.format(tr)+\
                        '      </RewardForCollectingItem>\n')
            # now step rewards if needed
            for tr in to_reward:
                if tr == 'step':
                    mission_xml[45] = mission_xml[45].replace("-0.01", "0.02")
            # # finally if rewarding attack objects
                # if tr == 'glowstone':
                    # self.collect_reward = True

        #################### GOAL #####################
        if hasattr(params.environment, "blocks"):
            for block in params.environment.blocks:
                goal_loc, all_positions = pop_sample(all_positions, 1)
                goal_loc = goal_loc[0]
                mission_xml.insert(
                    31,
                    '\t<DrawCuboid type="{}" x1="{}" x2="{}" y1="4" y2="5" z1="{}" z2="{}"/>\n'\
                    .format(block, goal_loc[0], goal_loc[0], goal_loc[1], goal_loc[1]))

        # ################### LAVA ######################
        # if hasattr(params.environment, "lava"):
            # n_lava = params.environment.lava
            # obj_locs, all_positions = pop_sample(all_positions, n_lava)
            # for n in range(n_lava):
                # mission_xml.insert(
                    # 23,
                    # '      <MovingTargetDecorator>\n'+
                    # '        <ArenaBounds>\n'+
                    # '          <min x="0" y="3" z="0"/>\n'+
                    # '          <max x="8" y="3" z="8"/>\n'+
                    # '        </ArenaBounds>\n'+
                    # '        <StartPos x="{}" y="3" z="{}"/>\n'.format(*obj_locs[n])+
                    # '        <Seed>random</Seed>\n'+
                    # '        <UpdateSpeed>16</UpdateSpeed>\n'+
                    # '        <PermeableBlocks type="cobblestone"/>\n'+
                    # '        <BlockType type="lava"/>\n'+
                    # '      </MovingTargetDecorator>\n')


            # # draw n_lava random positions from map
            # obj_locs, all_positions = pop_sample(all_positions, n_lava)
            # for i in range(n_lava):
                # mission_xml.insert(
                    # 31,
                    # '\t<DrawBlock type="lava" x="{}" y="3" z="{}"/>\n'\
                    # .format(*obj_locs[i]))

        # ################## GLOWSTONE #######################
        # if hasattr(params.environment, "glowstone"):
            # obj_locs, all_positions = pop_sample(all_positions, 1)
            # mission_xml.insert(
                # 23,
                # '      <MovingTargetDecorator>\n'+
                # '        <ArenaBounds>\n'+
                # '          <min x="0" y="4" z="0"/>\n'+
                # '          <max x="8" y="4" z="8"/>\n'+
                # '        </ArenaBounds>\n'+
                # '        <StartPos x="{}" y="4" z="{}"/>\n'.format(*obj_locs[0])+
                # '        <Seed>random</Seed>\n'+
                # '        <UpdateSpeed>8</UpdateSpeed>\n'+
                # '        <PermeableBlocks type="air"/>\n'+
                # '        <BlockType type="glowstone"/>\n'+
                # '      </MovingTargetDecorator>\n')


        mission_xml = ''.join(mission_xml)
        # print mission_xml
        self.my_mission = MalmoPython.MissionSpec(mission_xml, True)

        # set agent spawn location (random)
        agent_locs, all_positions = pop_sample(all_positions, 1)
        agent_yaw = random.sample([0.,90.,180.,270.], 1)[0]
        # agent_loc = (0,0)
        # agent_yaw = 0
        self.my_mission.startAtWithPitchAndYaw(
            agent_locs[0][0]+0.5,
            4.,
            agent_locs[0][1]+0.5,
            30.0,
            agent_yaw)

        ##################### APPLES #######################
        # draw objects
        if hasattr(params.environment, "objects"):
            for i, obj in enumerate(params.environment.objects):
                num_objects = int(params.environment.num_objects[i])
                object_locs, all_positions = pop_sample(all_positions, num_objects)
                for object_loc in object_locs:
                    self.my_mission.drawItem(
                        object_loc[0],4,object_loc[1],
                        str(obj))

    def _prepare_mission(self, params):
        '''
        Overrides _prepare_mission to run additional preparation on the mission.
        '''
        super(RoomNavigationEnvironment, self)._prepare_mission(params)

        # request 3x2x3 grid observation at and below the player's feet
        self.my_mission.observeGrid(-1, -1, -1, 1, 0, 1, "Grid")
        self._set_previous_position(0, 0, 0)
        self._set_previous_yaw(-10.0)

        if hasattr(params.environment, 'video_frame_width') and \
           hasattr(params.environment, 'video_frame_height'):
            print '>> requesting video'
            self.my_mission.requestVideo(params.environment.video_frame_width,
                                         params.environment.video_frame_height)
        else:
            raise AttributeError('Missing required parameters: '
                                 + 'video_frame_height, video_frame_width')

    def act(self, action):
        '''Take an action from the action_set and set the expected outcome so we can wait for it.'''
        assert(action in self.action_set)
        self._update_state()
        self.prev_x = self.get_previous_position()['x']
        self.prev_y = self.get_previous_position()['y']
        self.prev_z = self.get_previous_position()['z']
        self.prev_yaw = self.get_previous_yaw()
        self._set_previous_action(action)
        self.successful_attack = False
        # only attack pigs so as to not break blocks!
        if action != 'attack':
            self.agent_host.sendCommand(action)
        else:
            msg = self.world_state.observations[-1].text
            obs = json.loads(msg)
            if obs[u'LineOfSight']['type'] == 'glowstone':
                self.agent_host.sendCommand("attack")
                self.successful_attack = True
            else:
                # do some useless action (to get step cost)
                self.agent_host.sendCommand("jump 0")

    def is_valid_world_state(self, world_state):
        '''Check that a valid observation has been received together with a video frame.'''
        # wait for a valid observation
        if len(world_state.observations) == 0:
            return False
        if all(e.text=='{}' for e in world_state.observations):
            self.valid_counter += 1
            self.world_state_status = \
                "{}: obs.text is all empty!".format(self.valid_counter)
            # print self.world_state_status
            return False

        # require video frame
        if(world_state.number_of_video_frames_since_last_state == 0 or \
                len(world_state.video_frames) == 0):
            self.valid_counter += 1
            self.world_state_status = \
                "{}: number of video frames since last state is zero or ".format(self.valid_counter) + \
                "video frames is zero"
            # print self.world_state_status
            return False

        # check if this is the first step
        if not self._get_previous_action():
            return self.initialStateValid(world_state)

        # rewards cannot be empty, unless it is the first step (checked above)
        if len(world_state.rewards) < 1:
            self.valid_counter += 1
            self.world_state_status = \
                "{}: rewards are empty!".format(self.valid_counter)
            # print self.world_state_status
            return False
        # and they cannot be all zeros!
        has_nonzero_reward = False
        for r in world_state.rewards:
            if r.getValue() != 0:
                has_nonzero_reward = True
                break
        if not has_nonzero_reward:
            self.valid_counter += 1
            self.world_state_status = \
                "{}: rewards are not non zero!".format(self.valid_counter)
            # print self.world_state_status
            return False

        # check if other things about the state hold up
        return self.nextStateValid(world_state)

    def initialStateValid(self, world_state):
        '''
        Before a command has been sent we wait for an observation of the world
        and a frame.
        '''
        # wait for mission to start
        if not world_state.is_mission_running:
            return False
        # agent position should be returned in some observation
        e = world_state.observations[-1]
        obs = json.loads(e.text)
        if not (u'XPos' in obs and \
                u'YPos' in obs and \
                u'ZPos' in obs and \
                u'Yaw' in obs):
            return False
        # wait for one more frame just to be sure
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.number_of_video_frames_since_last_state == num_frames_seen:
            self.valid_counter += 1
            self.world_state_status = \
                "{}: initial state, no new frames have arrived since last ".format(self.valid_counter) + \
                "state!"
            # print self.world_state_status
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                raise ValueError('Mission Ended before a frame arrived after ' + \
                    'a valid observation')
        return True

    def nextStateValid(self, world_state):
        '''
        After each command has been sent we wait for the observation to
        change past a tolerance.
        '''
        # agent position should be returned in some observation
        e = world_state.observations[-1]
        obs = json.loads(e.text)
        if not (u'XPos' in obs and \
                u'YPos' in obs and \
                u'ZPos' in obs and \
                u'Yaw' in obs):
            return False
        self.curr_x = obs[u'XPos']
        self.curr_y = obs[u'YPos']
        self.curr_z = obs[u'ZPos']
        self.curr_yaw = obs[u'Yaw']

        # if movement is required by an action
        if 'move' in self._get_previous_action():
            # need the grid to determine if movement was possible
            if 'Grid' not in obs:
                self.valid_counter += 1
                self.world_state_status = \
                    "{}: grid is not in obs".format(self.valid_counter)
                # print self.world_state_status
                return False
            # wait till the agent has moved a little atleast
            if self.valid_counter < 100:
                if not self._is_path_blocked(obs['Grid']):
                    # if this issue has been raised more than 100 times, likely
                    # the agent is just stuck, so we can move on
                    if math.fabs(self.curr_x - self.prev_x) < self.tolerance and \
                            math.fabs(self.curr_y - self.prev_y) < self.tolerance and \
                            math.fabs(self.curr_z - self.prev_z) < self.tolerance:
                        self.valid_counter += 1
                        self.world_state_status = \
                            "{}: next state, agent did not move! ".format(self.valid_counter) + \
                            "{} {} {} {} {} {} {}".format(
                            self.curr_x,
                            self.prev_x,
                            self.curr_y,
                            self.prev_y,
                            self.curr_z,
                            self.prev_z,
                            self._get_previous_action())
                        # print self.world_state_status
                        # print world_state.observations[-1]
                        return False
            else:
                print "exceeded movement requirements. {}".format(self.world_state_status)
        elif 'turn' in self._get_previous_action():
            if math.fabs(self.curr_yaw - self.prev_yaw) < 90*self.tolerance:
                self.valid_counter += 1
                self.world_state_status = \
                    "{}: next state, agent did not yaw change!".format(self.valid_counter) + \
                    "{} {} {}".format(
                    self.curr_yaw,
                    self.prev_yaw,
                    self._get_previous_action())
                # print self.world_state_status
                return False

        # wait for the render position to have changed
        world_state = self.agent_host.peekWorldState()
        frame = world_state.video_frames[-1]
        curr_x_from_render   = frame.xPos
        curr_y_from_render   = frame.yPos
        curr_z_from_render   = frame.zPos
        curr_yaw_from_render = frame.yaw
        if 'move' in self._get_previous_action():
            if self.valid_counter < 100:
                if not self._is_path_blocked(obs['Grid']):
                    # if this issue has been raised more than 100 times, likely
                    # the agent is just stuck, so we can move on
                    if math.fabs(curr_x_from_render - self.prev_x) < self.tolerance and \
                            math.fabs(curr_y_from_render - self.prev_y) < self.tolerance and \
                            math.fabs(curr_z_from_render - self.prev_z) < self.tolerance:
                        self.valid_counter += 1
                        self.world_state_status = \
                            "{}: next state, agent did not render the move! ".format(self.valid_counter) + \
                            "{} {} {} {} {} {} {}".format(
                            curr_x_from_render,
                            self.prev_x,
                            curr_y_from_render,
                            self.prev_y,
                            curr_z_from_render,
                            self.prev_z,
                            self._get_previous_action())
                        # print self.world_state_status
                        return False
        elif 'turn' in self._get_previous_action():
            if math.fabs(curr_yaw_from_render - self.prev_yaw) < self.tolerance:
                self.valid_counter += 1
                self.world_state_status = \
                    "{}: next state, agent did not render the yaw change!".format(self.valid_counter) + \
                    "{} {} {}".format(
                    curr_yaw_from_render,
                    self.prev_yaw,
                    self._get_previous_action())
                # print self.world_state_status
                return False
        return True

    def _is_path_blocked(self, grid):
        '''
        this is determined from underlying grid observation around agent's feet
        only determines if direction agent is currently facing is blocked!
        '''
        if math.fabs(self.prev_yaw - 0.0) < self.tolerance*90 and \
                grid[16] != 'air':
            return True
        if (math.fabs(self.prev_yaw - 180.0) < self.tolerance*90 or \
                math.fabs(self.prev_yaw + 180.0) < self.tolerance*90) and \
                grid[10] != 'air':
            return True
        if math.fabs(self.prev_yaw - 90.0) < self.tolerance*90 and \
                grid[12] != 'air':
            return True
        if (math.fabs(self.prev_yaw - 270.0) < self.tolerance*90 or \
                math.fabs(self.prev_yaw + 90.0) < self.tolerance*90) and \
                grid[14] != 'air':
            return True

    def get_num_actions(self):
        '''
        return number of actions
        '''
        return len(self.action_set)

    def _update_state(self):
        if self.world_state is None:
            return

        if len(self.world_state.observations) > 0:
            obs = json.loads(self.world_state.observations[-1].text)

            self._set_previous_position(obs['XPos'], obs['YPos'],
                                        obs['ZPos'])
            self._set_previous_yaw(obs['Yaw'])

    def _set_previous_position(self, x, y, z):
        '''Private method to update agent position.'''
        self._prev_position = {'x': x, 'y': y, 'z': z}
    def _set_previous_yaw(self, yaw):
        self._yaw = yaw
    def get_previous_yaw(self):
        return self._yaw
    def get_previous_position(self):
        '''Helper function: get previous position.'''
        if not hasattr(self, "_prev_position"):
            return None
        return self._prev_position
