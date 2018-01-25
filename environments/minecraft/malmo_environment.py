# -------------------------------------------------------------------------------------------------
# This is the entry point to obtain an environment. This file basically implements a large switch
# so one can request for a supported environment. Ideally one could also use this file to implement
# functions that verify the consistency of environment information. I leave this for the future for
#  now.
#
# Copyright (C) Microsoft Corporation.  All rights reserved.
# -------------------------------------------------------------------------------------------------

import os
import sys
import random
import time
from inspect import getsourcefile

# import representation

# This is necessary because of the directory structure I imposed.
sys.path.insert(1, os.path.join(sys.path[0], 'lib'))

# Path to the schema files
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
os.environ['MALMO_XSD_PATH'] = os.path.join(current_path, 'include')

import MalmoPython

from environment import Environment

class MalmoEnvironment(Environment):
    '''Handles connection to the Malmo Platform for arbitrary missions.'''

    def __init__(self, params):
        '''Initialize the malmo environment.'''
        super(MalmoEnvironment, self).__init__(params)

        self.client_pool_list = params.environment.clients

        self.clients = MalmoPython.ClientPool()
        assert hasattr(params.environment, "clients"), "Missing required parameter: clients"
        for c in self.client_pool_list:
            # conversion to ascii string needed because json parses strings as unicode
            self.clients.add(MalmoPython.ClientInfo(str(c.ip), c.port))

        self.agent_host = MalmoPython.AgentHost()

        self.mission_record = MalmoPython.MissionRecordSpec()
        self.record_requested = False

        try:
            self.action_set = [str(i) for i in params.environment.action_set]
        except TypeError:
            self.action_set = ["movenorth 1", "movesouth 1", "moveeast 1", "movewest 1"]
            print "Warning: Using default action set: " + str(self.action_set)


    def _reset(self):
        self.agent_host = MalmoPython.AgentHost()
        self.clients = MalmoPython.ClientPool()
        import random
        # Don't rely on the random number generator used by agents and the environment.
        rng = random.Random()
        rng.shuffle(self.client_pool_list)
        for c in self.client_pool_list:
            # conversion to ascii string needed because json parses strings as unicode
            self.clients.add(MalmoPython.ClientInfo(str(c.ip), c.port))

    def _record(self):
        self.mission_record = MalmoPython.MissionRecordSpec(self.mission_record_filename)
        # what to record
        self.mission_record.recordCommands()
        self.mission_record.recordMP4(40, 400000)  # bitrate of video can be changed
        self.mission_record.recordRewards()
        self.mission_record.recordObservations()

    def record(self, filename, params):
        '''Sets up mission record'''
        self.record_requested = True
        self.mission_record_filename = filename
        self._record()

    def _load_mission_xml(self, params):
        '''Helper function that loads a mission definition as specified by params.'''
        self.my_mission = None
        mission_def = params.environment.mission_xml
        if type(mission_def) == list:
            mission_id = random.randint(0, len(mission_def) - 1)
            mission_def = str(mission_def[mission_id])
        else:
            assert type(mission_def) in [unicode, str], "Mission XML must be a JSON array or str or unicode but is %s" % type(mission_def)
            mission_def = str(mission_def)
            if ".xml" in mission_def:
                mission_def = str(mission_def)
            else:
                assert os.path.isdir(os.path.abspath(mission_def))
                mission_def = random.choice([os.path.join(os.path.abspath(mission_def), f)
                                            for f in os.listdir(os.path.abspath(mission_def))
                                                if f.endswith(".xml")])

        with open(mission_def, 'r') as f:
            print "Loading mission from %s" % params.environment.mission_xml
            mission_xml = f.read()
            self.my_mission = MalmoPython.MissionSpec(mission_xml, True)

        ### Ideally, the action set should be loaded here.

    def _save_mission_start(self, params):
        try:
            self.agent_host.startMission(self.my_mission, self.clients,
                                         self.mission_record,
                                         params.environment.role,
                                         str(params.environment.experiment_id))
        except RuntimeError as e:
            raise e

        print "Waiting for the mission to start",
        start_time = time.time()
        self.world_state = self.agent_host.peekWorldState()
        while not self.has_mission_begun():
            sys.stdout.write(".")
            time.sleep(0.1)
            self.world_state = self.agent_host.peekWorldState()
            for error in self.world_state.errors:
                print "Error:", error.text
            if time.time() - start_time > 60:
                raise RuntimeError('Could not start mission for 60s.')

    def start_mission(self, params):
        '''
        Summary: Start the next mission.
        '''
        max_retries = 100
        if hasattr(params.environment, 'max_retries'):
            max_retries = params.environment.max_retries
        retry = 0
        while True:
            try:
                time.sleep(1)
                self._load_mission_xml(params)
                # hook that can be overwritten by inheriting environments
                self._prepare_mission(params)
                # actually start the mission
                self._save_mission_start(params)
                print 'mission has started'
                break # only reached if no exception is thrown, i.e., mission started
            except RuntimeError as e:  # Catch Runtime Errors
                print e
                if retry >= max_retries:
                    print "\n\Could not connect after (%d) retries" % (retry)
                    time.sleep(30)
                    raise e
                retry += 1
                print "\n\nRetry (%d)" % (retry)
                time.sleep(5)
                self._reset() # new AgentHost() object created, re-shuffles the client pool

        # Mission started must be true (Return kept for backward compatibility)
        return True

    def _prepare_mission(self, params):
        '''
        Function called by start_mission, after the mission XML is loaded, but before
        the mission starts. Override to run additional preparation on the mission.
        '''
        self._set_previous_action(None)
        self.world_state = None

    def act(self, action):
        '''Send a single action to be executed in the environment. Raise exceptions when something goes wrong.'''
        assert(action in self.action_set)
        self._set_previous_action(action)
        self.agent_host.sendCommand(action)

    def has_mission_begun(self):
        '''Return True if the mission has started, False otherwise.'''
        if not self.world_state:
            return False
        return self.world_state.has_mission_begun

    def is_mission_running(self):
        '''Return True if the mission is running, False otherwise.'''
        if not self.world_state:
            return False
        return self.world_state.is_mission_running

    def get_world_state(self):
        '''Request the next world state.'''
        start_time = time.time()
        self.world_state = self.agent_host.peekWorldState()

        self.valid_counter = 0
        # wait until the world is ready
        while self.is_mission_running() and not self.is_valid_world_state(self.world_state):
            time.sleep(0.01)
            self.world_state = self.agent_host.peekWorldState()
            if (time.time() - start_time) > 5: # 5 second timeout
                print self._get_previous_action()
                if len(self.world_state.observations) > 0:
                    import json
                    obs = json.loads(self.world_state.observations[-1].text)
                    print obs
                raise ValueError('Did not get valid world state bcs: {}'.format(
                    self.world_state_status))

        # consume (get) the world state once it's valid or the mission ended
        # note that an incomplete world state can be returned after mission end
        # the caller needs to check whether the mission was still running when
        # the world state was retrieved
        self.world_state = self.agent_host.getWorldState()

        return self.world_state

    def is_valid_world_state(self, world_state):
        '''Check whether the provided world state is valid.
           @override to customize checks
        '''
        # print errors, just for information
        for error in world_state.errors:
            print "Error:", error.text

        # observation cannot be empty
        if len(world_state.observations) < 1:
            self.valid_counter += 1
            self.world_state_status = \
                "{}: there are no observations".format(self.valid_counter)
            # print self.world_state_status
            return False

        if hasattr(self, "repr_"):
            try:
                self.repr_.get_features(world_state)
            except Exception as ex:
                print "  >> Warning, could not get representation: {}".format(ex.message)
                return False
        return True

    def get_action_set(self, parameters):
        '''Get the action set for this environment.'''
        # to do: parameterize this based on the parameters / make sure this matches the mission definition. Action set determined in __init__.
        return self.action_set

    def _set_previous_action(self, action):
        self._action = action

    def _get_previous_action(self):
        if not hasattr(self, "_action"):
            return None
        return self._action

