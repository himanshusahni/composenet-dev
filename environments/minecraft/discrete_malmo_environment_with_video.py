# -------------------------------------------------------------------------------------------------
# An extension to DiscreteMalmoEnvironment that additionally and waits for
# video observations.
#
# Copyright (C) Microsoft Corporation.  All rights reserved.
# -------------------------------------------------------------------------------------------------


from discrete_malmo_environment import DiscreteMalmoEnvironment
import MalmoPython
import json
import math

class DiscreteMalmoEnvironmentWithVideo(DiscreteMalmoEnvironment):

    def __init__(self, parameters):
        '''Initialize the malmo environment.'''
        super(DiscreteMalmoEnvironmentWithVideo, self).__init__(parameters)
        self.action_type = parameters.environment.action_type
        self.tolerance = 0.1


    def _prepare_mission(self, params):
        '''
        Overrides _prepare_mission to run additional preparation on the mission.
        '''
        super(DiscreteMalmoEnvironmentWithVideo, self)._prepare_mission(params)

        self._set_previous_action(None)

        # if "keep_all_frames" in params.environment:
            # self.agent_host.setVideoPolicy(
                # MalmoPython.VideoPolicy.KEEP_ALL_FRAMES)

        if hasattr(params.environment, 'video_frame_width') and \
           hasattr(params.environment, 'video_frame_height'):
            print '>> requesting video'
            self.my_mission.requestVideo(params.environment.video_frame_width,
                                         params.environment.video_frame_height)
        else:
            raise AttributeError('Missing required parameters: '
                                 + 'video_frame_height, video_frame_width')

    def is_valid_world_state(self, world_state):
        '''Check that a valid observation has been received together with a video frame.'''
        is_valid = super(DiscreteMalmoEnvironmentWithVideo, self).is_valid_world_state(world_state)

        if is_valid is False:
            return False

        # require video frame
        if(world_state.number_of_video_frames_since_last_state == 0 or len(world_state.video_frames) == 0):
               # self.valid_counter += 1
               # print "number of video frames since last state is zero or video frames is zero {}".format(self.valid_counter)
               return False

        # check if this is the first step
        if not self._get_previous_action():
            self.waitForInitialState()
            return True

        # rewards cannot be empty, unless it is the first step (checked above)
        if len(world_state.rewards) < 1:
            # self.valid_counter += 1
            # print "rewards are empty again! {}".format(self.valid_counter)
            return False
        has_nonzero_reward = False
        for r in world_state.rewards:
            if r.getValue() != 0:
                has_nonzero_reward = True
                break
        if not has_nonzero_reward:
            # self.valid_counter += 1
            # print "rewards are not non zero again! {}".format(self.valid_counter)
            return False
        #need to wait till the rendering has completed
        self.waitForNextState()
        return True

    def waitForInitialState(self):
        '''Before a command has been sent we wait for an observation of the world and a frame.'''
        # wait for a valid observation
        world_state = self.agent_host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            # self.valid_counter += 1
            # print "waiting for initial state bcs observations.text is empty! {}".format(self.valid_counter)
            world_state = self.agent_host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            # self.valid_counter += 1
            # print "waiting for initial state bcs no new frames have arrived since last state! {}".format(self.valid_counter)
            world_state = self.agent_host.peekWorldState()

        retries = 3
        for r in range(retries):
            world_state = self.agent_host.peekWorldState()
            if world_state.is_mission_running:
                assert len(world_state.video_frames) > 0, 'No video frames!?'
                i = len(world_state.observations) - 1
                while i >= 0:
                    try:
                        obs = json.loads( world_state.observations[i].text )
                        self.prev_x   = obs[u'XPos']
                        self.prev_y   = obs[u'YPos']
                        self.prev_z   = obs[u'ZPos']
                        self.prev_yaw = obs[u'Yaw']
                        return world_state
                    except KeyError:
                        i -= 1
        if r == retries:
            raise KeyError("Positions not in world state")


    def waitForNextState(self):
        '''After each command has been sent we wait for the observation to change as expected and a frame.'''
        # wait for the observation position to have changed
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                break
            if not all(e.text=='{}' for e in world_state.observations):
                i = len(world_state.observations) - 1
                while i >= 0:
                    try:
                        obs = json.loads( world_state.observations[-1].text )
                        self.curr_x   = obs[u'XPos']
                        self.curr_y   = obs[u'YPos']
                        self.curr_z   = obs[u'ZPos']
                        self.curr_yaw = obs[u'Yaw']
                        break
                    except KeyError:
                        i -= 1
                if i == -1:
                    raise KeyError("Positions not in world state")
                if self.require_move:
                    if math.fabs( self.curr_x - self.prev_x ) > self.tolerance or\
                       math.fabs( self.curr_y - self.prev_y ) > self.tolerance or\
                       math.fabs( self.curr_z - self.prev_z ) > self.tolerance:
                        break
                    # else:
                        # self.valid_counter += 1
                        # print "waiting for next state bcs movement is required but agent did not move! {}".format(self.valid_counter)
                elif self.require_yaw_change:
                    if math.fabs( self.curr_yaw - self.prev_yaw ) > self.tolerance:
                        break
                    # else:
                        # self.valid_counter += 1
                        # print "waiting for next state yaw change is required but agent did not move! {} {} {} {}".format(self.valid_counter, self.prev_yaw, self.curr_yaw, self.action)
                else:
                    break
        # wait for the render position to have changed
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                break
            # self.valid_counter += 1
            # print "waiting for next state bcs agent has not moved into tolerance! {}".format(self.valid_counter)
            frame = world_state.video_frames[-1]
            curr_x_from_render   = frame.xPos
            curr_y_from_render   = frame.yPos
            curr_z_from_render   = frame.zPos
            curr_yaw_from_render = frame.yaw
            if self.require_move:
                if math.fabs( curr_x_from_render - self.prev_x ) > self.tolerance or\
                   math.fabs( curr_y_from_render - self.prev_y ) > self.tolerance or\
                   math.fabs( curr_z_from_render - self.prev_z ) > self.tolerance:
                    break
            elif self.require_yaw_change:
                if math.fabs( curr_yaw_from_render - self.prev_yaw ) > self.tolerance:
                    break
            else:
                break

        num_frames_before_get = len(world_state.video_frames)
        world_state = self.agent_host.peekWorldState()

        # if world_state.is_mission_running:
            # assert len(world_state.video_frames) > 0, 'No video frames!?'
            # num_frames_after_get = len(world_state.video_frames)
            # assert num_frames_after_get >= num_frames_before_get, 'Fewer frames after getWorldState!?'
            # frame = world_state.video_frames[-1]
            # obs = json.loads( world_state.observations[-1].text )
            # self.curr_x   = obs[u'XPos']
            # self.curr_y   = obs[u'YPos']
            # self.curr_z   = obs[u'ZPos']
            # self.curr_yaw = obs[u'Yaw']
            # if math.fabs( self.curr_x   - self.expected_x   ) > self.tolerance or\
               # math.fabs( self.curr_y   - self.expected_y   ) > self.tolerance or\
               # math.fabs( self.curr_z   - self.expected_z   ) > self.tolerance or\
               # math.fabs( self.curr_yaw - self.expected_yaw ) > self.tolerance:
                   # raise ValueError('ERROR DETECTED in observations! Expected: '+str(self.expected_x)+', '+str(self.expected_y)+', '+str(self.expected_z)+', yaw '+str(self.expected_yaw)+'. RECEIVED OBSERVATION: '+str(self.curr_x)+', '+str(self.curr_y)+', '+str(self.curr_z)+', yaw '+str(self.curr_yaw))
            # curr_x_from_render   = frame.xPos
            # curr_y_from_render   = frame.yPos
            # curr_z_from_render   = frame.zPos
            # curr_yaw_from_render = frame.yaw
            # if math.fabs( curr_x_from_render   - self.expected_x   ) > self.tolerance or\
               # math.fabs( curr_y_from_render   - self.expected_y   ) > self.tolerance or \
               # math.fabs( curr_z_from_render   - self.expected_z   ) > self.tolerance or \
               # math.fabs( curr_yaw_from_render - self.expected_yaw ) > self.tolerance:
                # raise ValueError('ERROR DETECTED in rendering! Expected: '+str(self.expected_x)+', '+str(self.expected_y)+', '+str(self.expected_z)+', yaw '+str(self.expected_yaw)+'. RECEIVED RENDER: '+str(curr_x_from_render)+', '+str(curr_y_from_render)+', '+ str(curr_z_from_render)+', yaw ' + str(curr_yaw_from_render))
            # self.prev_x   = self.curr_x
            # self.prev_y   = self.curr_y
            # self.prev_z   = self.curr_z
            # self.prev_yaw = self.curr_yaw

        return world_state

