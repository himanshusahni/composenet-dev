'''
Wrapper that converts any map style in this directory
to a standard RL interface.
'''
from collections import deque
from PIL import Image
import numpy as np
import os
import sys
from inspect import getsourcefile
import json
from sets import Set

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
sys.path.append(os.path.join(current_path, 'minecraft'))

from room_navigation_environment import RoomNavigationEnvironment
from parameters import Parameters

class MinecraftEnv(object):

    # number of frames in a state
    NUM_CHANNELS = 3

    def __init__(self, param_file):
        self.params = Parameters(param_file)
        self.params.environment.mission_xml = os.path.join(
            current_path,
            'minecraft',
            self.params.environment.mission_xml)
        self.env = RoomNavigationEnvironment(self.params)
        self.max_retries = 25
        self.max_steps = 50

    def reset(self, max_steps=25):
        '''
        starts a new mission and returns the first state
        '''
        self.world_state_status = ''
        # clear out rotating state buffer
        self.state = deque(
            [np.zeros((self.params.environment.video_frame_width,
                self.params.environment.video_frame_width, 1)) \
                for _ in range(self.NUM_CHANNELS)],
            maxlen=self.NUM_CHANNELS)
        self.steps = 0

        # start a new mission
        retry = 0
        while True:
            self.env.start_mission(self.params)
            try:
                world_state = self.env.get_world_state()
                if world_state.is_mission_running:
                    break
                else:
                    if retry < self.max_retries:
                        print "crashed because mission ended before it started. retry {}".format(retry)
                        # try again
                        retry += 1
            except ValueError as ex:
                if retry < self.max_retries:
                    print "crashed starting new mission. {}. retry {}".format(
                        str(ex),
                        retry)
                    # try again
                    retry += 1
                else:
                    # some other exception or maxed out retries
                    raise ex

        return self.__get_state(world_state)

    def __get_state(self, world_state):
        '''
        constructs a state from malmo world_state object
        '''
        # grab most recent frame from minecraft
        pixels = world_state.video_frames[-1].pixels
        # preprocess to grayscale
        image = Image.frombytes(
            'RGB',
            (self.params.environment.video_frame_width,
                self.params.environment.video_frame_width),
            str(pixels))
        image = np.array(image.convert('L'))
        image = np.expand_dims(image, axis=-1)
        # add to rotating state buffer
        self.state.append(image)
        return np.concatenate(self.state, axis=-1)

    def __get_reward(self, world_state):
        base_reward = sum(r.getValue() for r in world_state.rewards)
        if self.env.successful_attack and self.env.collect_reward:
            base_reward += 1.0
        return base_reward

    def __get_terminal(self, world_state):
        # hack because sometimes minecraft does not quit on time
        # so we manually encode two conditions to exit -
        # greater than 0.5 absolute reward (goal or lava)
        # and > max_steps steps
        return abs(self.__get_reward(world_state)) > 0.9 or \
            self.steps >= self.max_steps or \
            (not world_state.is_mission_running)

    def step(self, action):
        '''
        execute the action and return RL tuple
        '''
        self.action = self.env.action_set[action]
        # action is index of action in environment
        self.env.act(self.env.action_set[action])
        self.steps += 1

        world_state = self.env.get_world_state()
        # if episode has terminated, there will be no more
        # frames sent from minecraft, so state is just replicated
        if self.__get_terminal(world_state):
            cur_state = np.concatenate(self.state, axis=-1)
        else:
            cur_state = self.__get_state(world_state)

        return (cur_state,
            self.__get_reward(world_state),
            self.__get_terminal(world_state))


    def get_state_size(self):
        return (self.params.environment.video_frame_width,
            self.params.environment.video_frame_height)

    def get_num_actions(self):
        return self.env.get_num_actions()

    def get_num_channels(self):
        return self.NUM_CHANNELS

# m = MinecraftEnv('minecraft/params_beacon_apples.cfg')
# m.env._load_mission_xml(m.params)
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import random
# import sys
# import time
# while True:
    # m.reset()
    # done = False
    # while not done:
        # nb = random.choice(["turn 1","turn -1", "move 1"])
        # m.env.agent_host.sendCommand(nb)
        # m.env.get_world_state()
        # print m.env.world_state.observations[-1]
        # print [re.getValue() for re in m.env.world_state.rewards]
        # r = sum(re.getValue() for re in m.env.world_state.rewards)
        # # if r > 0:
            # # break
        # time.sleep(1)
    # # if r > 0:
        # # break
