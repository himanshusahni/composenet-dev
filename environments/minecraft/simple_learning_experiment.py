from experiment import Experiment

import environment
import agent

import os
import random
import csv
import time
import traceback

class SimpleLearningExperiment(Experiment):
    def __init__(self, params):

        self.mission_retry = 10

        self.params = params
        # Not calling superclass' __init__ (Seed is set on each evaluation repeat; See first line of run())
        if 'seed' in params:
            if type(params.seed) == list:
                self.random_seed_list = [int(x) for x in params.seed]
            else:
                self.random_seed_list = [int(params.seed)]
        if 'numpy_seed' in params:
            if type(params.numpy_seed) == list:
                self.numpy_random_seed_list = [int(x) for x in params.numpy_seed]
            else:
                self.numpy_random_seed_list = [int(params.numpy_seed)]

        if not hasattr(self, 'random_seed_list'):
            self.random_seed_list = [self._get_seed() for i in range(params.num_repeat_eval)]
            params.set_param('added_during_execution.chosen_seed', self.random_seed_list)

        if not hasattr(self, 'numpy_random_seed_list'):
            self.numpy_random_seed_list = [self._get_seed() for i in range(params.num_repeat_eval)]
            params.set_param('added_during_execution.chosen_numpy_seed', self.numpy_random_seed_list)

        train_env = params.train_environment
        eval_env = params.eval_environment

        params.environment = train_env

        self.training_missions = params.environment.mission_xml

        # set up the training environment (mission).
        self.train_env = environment.get_environment(params)

        params.environment = eval_env

        self.evalution_missions = params.environment.mission_xml

        # set up the evaluation environment (mission).
        self.eval_env = environment.get_environment(params)

        self.num_training_episodes = params.num_train_ep
        self.num_repeat_eval = params.num_repeat_eval

        assert len(self.random_seed_list) == self.num_repeat_eval, \
                   """Random python seeds will be reused between evaluation repetitions
                         since the length(s) of the random seed list(s) given is/are less than the
                         number of evaluation repeats. This is most likely not desired."""

        assert len(self.numpy_random_seed_list) == self.num_repeat_eval, \
                   """Random numpy seeds will be reused between evaluation repetitions
                         since the length(s) of the random seed list(s) given is/are less than the
                         number of evaluation repeats. This is most likely not desired."""


        self.num_eval_episodes = params.num_eval_episodes
        self.eval_schedule = params.eval_schedule
        self.prog_schedule = params.prog_schedule

        self.env = None

        self.params = params

        self.exp_start_time = time.time()
        self.num_episodes_complete = 0

    def _set_seed(self, eval_rep):
        import random as rand
        from numpy import random as nprand

        rand.seed(int(self.random_seed_list[eval_rep]))
        nprand.seed(int(self.numpy_random_seed_list[eval_rep]))

    def run(self):
        '''Run the specified experiment.'''
        # set up the agent/algorithm.
        self.train_agent = \
            agent.get_agent(self.params,
                            self.train_env.get_action_set(self.params))
        self.eval_agent = None
        self.agent = None

        # back up experiment config
        self.experiment_cfg_filename = str(os.path.join(
            os.path.expanduser(self.params.exp_dir),
            "experiment.cfg"))
        if not os.path.exists(os.path.expanduser(self.params.exp_dir)):
            os.mkdir(os.path.expanduser(self.params.exp_dir))
        with open(self.experiment_cfg_filename, "wb") as f:
            self.params.write_json(f)
        # run the experiment
        for i in xrange(1, self.num_repeat_eval + 1):
            self._set_seed(i - 1)
            self.env = self.train_env
            self.params.environment = self.params.train_environment
            self.train_agent = \
                agent.get_agent(self.params, self.train_env.get_action_set(self.params))
            self.agent = self.train_agent

            print("Experiment run repetition %d out of %d" %
                  (i, self.num_repeat_eval))

            for j in xrange(1, self.num_training_episodes + 1):
                print("Training mission %d out of %d" %
                      (j, self.num_training_episodes))

                self.curr_mission = "TRAIN_" + \
                    self.choose_training_mission()

                # catch mission errors here - re-run if needed
                success = False
                retry = 0
                while not success:
                    agent_environment_interaction_logger.\
                        log_mission_start(self.curr_mission)
                    ep_start_time = time.time()
                    try:
                        reward, num_steps = self.run_episode()
                    except Exception as e:
                        agent_environment_interaction_logger.\
                            cancel_mission(self.curr_mission)
                        if retry < self.mission_retry:
                            retry += 1
                            print traceback.print_exc()
                            print ">> Warning: failed to run mission. Retry %d" % retry
                            continue
                        else:
                            raise e
                    print("\n\n----------")
                    print("Total Reward: %.2f, Num. Steps: %d" %
                          (reward, num_steps))
                    ep_end_time = time.time()
                    print("Episode took: " +
                          hms_string(ep_end_time - ep_start_time))
                    self.num_episodes_complete += 1
                    print("Eval Rep: %d, Training Ep: %d, " %
                          (i, j) +
                          "Total Num Ep. Completed: " +
                          str(self.num_episodes_complete) + " episode(s).\n"
                          "Time Run: " + hms_string(ep_end_time - self.exp_start_time) +
                          " (%.3f" % (float(ep_end_time - self.exp_start_time)/self.num_episodes_complete) + " seconds/episode.)" )
                    print("----------\n\n")

                    agent_environment_interaction_logger.log_mission_end(
                        reward, num_steps)
                    success = True
                    # end of TRAINING while .. success loop

                if check_schedule(self.params.eval_schedule, j):
                    self.env = self.eval_env
                    self.params.environment = self.params.eval_environment
                    self.agent = self.agent.get_eval_agent()

                    for k in xrange(1, self.num_eval_episodes + 1):
                        print("Evaluation mission %d out of %d" %
                              (k, self.num_eval_episodes))

                        self.curr_mission = "EVAL_" + \
                            self.choose_evaluation_mission(k - 1)

                        # catch mission errors here - re-run if needed
                        success = False
                        retry = 0
                        while not success:

                            # self.env.record(str(os.path.join(
                            #    os.path.expanduser(self.params.exp_dir),
                            #   "eval_exp-%d-%d-%d.tgz" % (i, j, k))), self.params)

                            # print("Recorded.")

                            ep_start_time = time.time()
                            agent_environment_interaction_logger.\
                                log_mission_start(self.curr_mission)
                            try:
                                reward, num_steps = self.run_episode()
                            except Exception as e:
                                agent_environment_interaction_logger.\
                                    cancel_mission(self.curr_mission)
                                if retry < self.mission_retry:
                                    retry += 1
                                    print traceback.print_exc()
                                    print ">> Warning: failed to run mission. Retry %d" % retry
                                    continue
                                else:
                                    raise e
                            print("\n\n----------")
                            print("Total Reward: %.2f, Num. Steps: %d" %
                                  (reward, num_steps))
                            ep_end_time = time.time()
                            print("Episode took: " +
                                  hms_string(ep_end_time - ep_start_time))

                            self.num_episodes_complete += 1
                            print("Eval Rep: %d, Last Training Ep: %d, Eval. Ep: %d, "
                                  % (i, j, k) + "Total Num Ep. Completed: " +
                                  str(self.num_episodes_complete) + " episode(s).\n"
                                  "Time Run: " + hms_string(ep_end_time - self.exp_start_time) +
                                  " (%.3f" % (float(ep_end_time - self.exp_start_time)/self.num_episodes_complete) + " seconds/episode.)" )

                            print("----------\n\n")
                            agent_environment_interaction_logger.log_mission_end(
                                reward, num_steps)
                            success = True
                            # end of EVAL while .. success loop

                    log_file_name = str(os.path.join(
                             os.path.expanduser(self.params.exp_dir),
                             "logs-%d-%d.csv" % (i, j)))
                    agent_environment_interaction_logger.save(log_file_name)

                    agent_file_name = str(os.path.join(
                             os.path.expanduser(self.params.exp_dir),
                             "agent-%d-%d.pkl" % (i, j)))
                    try:
                        self.agent.save(agent_file_name)
                    except NotImplementedError:
                        print """Note: Agent has not implemented save(). Not
                                       saving the agent."""
                        agent_file_name = None

                    self.env = self.train_env
                    self.params.environment = self.params.train_environment
                    self.agent = self.train_agent

        # CALLBACKS #
        if 'notification_email' in self.params:
            from util.callbacks import emailer
            try:
                emailer.send_mail(
                    send_from=self.params.notification_email,
                    send_to=[self.params.notification_email],
                    subject="Update on progress.",
                    text="Experiment complete. " +
                        "Completed %d" %
                        self.num_episodes_complete +
                        " episode(s) of which %d" %
                        ((i - 1) * self.num_training_episodes + j) +
                        " are training episodes in " +
                        hms_string(time.time() -
                                    self.exp_start_time) +
                        ".",
                    server="emea.064d.cloudmail.microsoft.com"
                    )
            except:
                print "Did not send mail, error occured."

    def run_episode(self):
        '''Run this agent on a mission.
           @return the total reward accumulated on this mission
        '''
        # start the environment mission
        self.env.start_mission(self.params)

        # hook for agent initialization at the start of the mission
        self.agent.mission_start()

        self.total_reward = 0
        num_actions = 0
        action = None

        hist = []

        while True:
            # get the next world state. note: we get the next world state just
            # before the check for is_running, to ensure that action selection
            # reacts to a running state

            world_state = self.env.get_world_state()

            if world_state and action:
                # accumulate reward
                reward = sum(r.getValue() for r in world_state.rewards)
                self.total_reward += reward
                hist.append([action, reward]) # reward for the previous action

            if not world_state.is_mission_running:
                # the mission ended
                break

            # select and send the next action
            action = self.agent.select_action(world_state)
            self.env.act(action)
            num_actions += 1

        # end of while loop

        # callback for inheriting agents at the end of the mission
        self.agent.mission_end(world_state)
        for transition in hist:
            agent_environment_interaction_logger.log_interaction(transition[0],
                                                                 transition[1])
        return self.total_reward, num_actions

    def choose_training_mission(self):
        mission_def = self.training_missions
        if type(mission_def) == list:
            mission_id = random.randint(0, len(mission_def) - 1)
            mission_def = mission_def[mission_id]
            self.params.environment.mission_xml = unicode(mission_def)
            return mission_def
        else:
            assert type(mission_def) == unicode, \
                "Mission XML must be a JSON array or unicode"
            mission_def = str(mission_def)
            if ".xml" in mission_def:
                self.params.environment.mission_xml = unicode(mission_def)
                return mission_def
            else:
                assert os.path.isdir(os.path.abspath(mission_def)), \
                    "Folder does not exist: %s" % os.path.abspath(mission_def)
                mission_def_dir = mission_def
                mission_def = random.choice(
                    [f for f in os.listdir(os.path.abspath(mission_def))
                     if f.endswith(".xml")])
                self.params.environment.mission_xml = \
                    unicode(os.path.join(mission_def_dir, mission_def))
                return mission_def

    def choose_evaluation_mission(self, index):
        mission_def = self.evalution_missions
        if type(mission_def) == list:
            mission_def = mission_def[index % len(mission_def)]
            self.params.environment.mission_xml = unicode(mission_def)
            return mission_def
        else:
            assert type(mission_def) == unicode, \
                "Mission XML must be a JSON array or unicode"
            mission_def = str(mission_def)
            if ".xml" in mission_def:
                self.params.environment.mission_xml = unicode(mission_def)
                return mission_def
            else:
                assert os.path.isdir(os.path.abspath(mission_def))
                mission_def_dir = mission_def
                mission_def = \
                    sorted([f for f in os.listdir(os.path.abspath(mission_def))
                            if f.endswith(".xml")])
                mission_def = mission_def[index % len(mission_def)]
                self.params.environment.mission_xml = \
                    unicode(os.path.join(mission_def_dir, mission_def))
                return mission_def


def check_schedule(schedule, num):
    if schedule.type == "regular":
        return num % schedule.params == 0

    elif schedule.type == "logarithmic":
        return isPower(float(num) / float(schedule.params.multiplier),
                       schedule.params.base)

    elif schedule.type == "custom":
        return num in schedule.params

    else:
        assert "Unknown schedule: " + schedule.type

# HELPER FUNCTIONS #


def isPower(num, base):
    testnum = 1
    while testnum < num:
        testnum = testnum * base
    return testnum == num


def hms_string(sec_elapsed):
    d = sec_elapsed // 86400
    h = sec_elapsed // 3600 % 24
    m = sec_elapsed // 60 % 60
    s = sec_elapsed % 60
    return "%d days, %d hours, %d minutes, %d seconds" % (d, h, m, s)


# LOGGER #


class agent_environment_interaction_logger:
    """ Functor for logging agent environment interactions."""
    DATA = []

    @classmethod
    def log_mission_start(agent_environment_interaction_logger, mission_name):
        agent_environment_interaction_logger.DATA.append([mission_name])

    @classmethod
    def cancel_mission(agent_environment_interaction_logger, mission_name):
        if len(agent_environment_interaction_logger.DATA[-1]) == 0 or \
            agent_environment_interaction_logger.DATA[-1][0] == mission_name:
            agent_environment_interaction_logger.DATA.pop()

    @classmethod
    def log_interaction(agent_environment_interaction_logger, action, reward):
        agent_environment_interaction_logger.DATA[-1].append(action)
        agent_environment_interaction_logger.DATA[-1].append(reward)

    @classmethod
    def log_mission_end(agent_environment_interaction_logger,
                        total_reward, num_steps):
        agent_environment_interaction_logger.DATA[-1].insert(1, total_reward)
        agent_environment_interaction_logger.DATA[-1].insert(2, num_steps)

    @classmethod
    def save(agent_environment_interaction_logger, name):
        with open(name, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(agent_environment_interaction_logger.DATA)
