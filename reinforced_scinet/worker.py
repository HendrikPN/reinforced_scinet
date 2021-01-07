#   Copyright 2020 reinforced_scinet (https://github.com/hendrikpn/reinforced_scinet)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import torch
from torch.multiprocessing import Process, Value
import numpy as np
import gym
import sys, os
from datetime import datetime
import warnings

from config import Config
from memory import ShortTermMemory

class WorkerProcess(Process):
    def __init__(self, experience_q, prediction_q, observation_q, env_id, episode_log_q, agent_id):
        """
        Workers are the agents interacting with the environment.
        Workers run a copy of the environment with their own specifications. 
        It requires Predictor processes to make decisions.
        Gathered experiences are submitted to a Queue on which the shared models are trained.

        Args:
            experience_q (mp.Queue): Shared memory queue containing experiences across workers of the same type.
            prediction_q (mp.Queue): Shared memory queue containing predictions of this worker.
            observation_q (mp.Queue): Shared memory queue containing observation across workers of the same type.
            env_id (str): The id of the environment instance this worker is interacting with.
            episode_log_q (mp.Queue): Shared memory queue containing the experience of past episodes.
            agent_id (int): The id of the worker process.
        """
        super(WorkerProcess, self).__init__()
        self.experience_q = experience_q
        self.prediction_q = prediction_q
        self.observation_q = observation_q
        self.env_id = env_id
        self.episode_log_q = episode_log_q
        self.id = agent_id

        #:class:`memory.ShortTermMemory`: Short term memory where the history is saved and experiences are memorized.
        self.memory = ShortTermMemory(Config.GLOW)
        #int: Signal for process exit.
        self.exit_flag = Value('i', 0)
        #torch.Tensor of float: Array of actions in one-hot encoding.
        self.actions = torch.Tensor(np.eye(Config.NUM_ACTIONS))
        #:class:`gym.Env`: The environment the agent interacts with.
        self.env = gym.make(Config.ENV_NAME, **Config.ENV_PARAMS[self.env_id])
        #bool: Boolean value that signals that an episode is finished.
        self.done = False
        #int: Current size of batches.
        self.batch_size = 0
        #torch.Tensor: Tensor of observation batch.
        self.o_batch = torch.Tensor([0.])
        #torch.Tensor: Tensor of action batch.
        self.a_batch = torch.Tensor([0.])
        #torch.Tensor: Tensor of target batch.
        self.t_batch = torch.Tensor([0.])
    
    def predict(self, observation): 
        """ 
        Predict h-values with deep PS network of Predictor process.

        (i) Put observation into queue for predictor.
        (ii) Get prediction from predictor.
         
        Args: 
            observation (torch.Tensor): Tensor which represents the current state of the environment. 
 
        Returns: 
            h_values (torch.Tensor): List of weights which define h-values in the PS framework. 
        """
        # (i) Queueing
        self.observation_q.put((torch.LongTensor([self.id]), observation))
        # (ii) Prediction
        h_values = self.prediction_q.get()
        return h_values

    def select_action(self, h_values, observation, time_step):
        """
        Selects an action with a softmax function according to predicted h-values.
        
        In 'selection' mode we may select some actions at random among actions which
        have a likelihood to be sampled that is at least random.
        Also, additional training data will be generated for actions which 
        should not be chosen, i.e. which have a chance to be sampled that is less 
        than random.
        
        Args: 
            h_values (torch.Tensor): List of weights which define the probability distribution  
                                     over the action space. 
            observation (torch.Tensor): Tensor which represents the current state of the environment. 
            time_step (int): The current time step.

        Returns:
            action (int): Action index.
            is_saved (bool): Whether or not to save this experience.
        """
        rescale = torch.max(h_values)
        h_values = torch.exp(Config.SOFTMAX*(h_values-rescale))
        h_values = h_values/torch.sum(h_values)
        is_saved = True

        # selection data
        if Config.TRAIN_MODE == 'selection':
            is_saved = False
            # record only after some actions
            if Config.POLICY_STEPS > 0:
                steps = np.random.choice(Config.POLICY_STEPS)
            else:
                steps = 0
            if time_step in range(steps, steps+Config.NUM_ACTIONS_PREDICT):
                #add naught experience
                self._add_naught_experience(observation, h_values)
                
                #choose non-random action at random
                h_values = (h_values >= 1./Config.NUM_ACTIONS).float()
                is_saved = True
        
        action = torch.multinomial(h_values, 1).item()

        return action, is_saved

    def generate_episode(self):
        """
        Plays an episode of the respective game version.
        (i) Interacts with environment.
        (ii) Saves interaction event.
        (iii) At the end or after a fixed time, creates, yields and resets experience. 

        NOTE: + If Config.PLAY and Config.RENDER are set to True, it will render the game.

        TODO:  + Experience/Memory is created twice. First, the experience is created in `ShortTermMemory`.
                 Second, the memory is written into the experience queue in an appropriate batch size.
                 In principle, this could all be handled in the memory already.

        Yields:
            memory (:obj:`list` of :obj:`tuple`): List of experiences gathered at that point.
            total_reward (int): The total sum of rewards at that point.
            time_step (int): The current time step.
            done (bool): The information about whether or not an episode is finished.
        """
        self.done = False
        observation = self.env.reset()
        observation = self._reshape_observation(observation)
        past_observation = observation
        time_step = 0
        total_reward = 0
        if Config.PLAY and Config.RENDER:
            self.env.render()

        while not self.done:
            # (i) Play.
            h_values = self.predict(observation)
            action, is_saved = self.select_action(h_values, observation, time_step)
            observation, reward, self.done = self.env.step(action)
            if reward < 0.:
                reward = 0.
            observation = self._reshape_observation(observation)
            total_reward += reward
            if Config.PLAY and Config.RENDER:
                self.env.render()

            # (ii) Cummulate history.
            action_enc = self.actions[action].reshape(1, len(self.actions[action]))
            event = (past_observation, action_enc, h_values[action], reward)
            self.memory.add_event(event, flag=is_saved)
            past_observation = observation

            time_step += 1
            # (iii) Create and yield experiences from history. Reset memory but not history.
            if self.done or time_step % Config.MAX_TIME == 0:
                self.memory.create_experience()
                yield self.memory.memory, total_reward, time_step, self.done
                self.memory.reset_memory()

    def run(self):
        """
        Runs the process worker.
        (i.1) Plays an episode with environment.
        (i.2) Appends data to the experience queue.
        (i.3) Log data at the end of an episode.
        (ii) Reset memory and history.
        (iii) Repeat until terminated.
        """
        print(f"Spawning worker #{self.id} of type {self.env_id}")
        sys.stdout.flush()
        count = 0
        while not self.exit_flag.value:
            # (i.1) Play an episode.
            for a, total_reward, time_step, done in self.generate_episode():
                # (i.2) Add data to experience queue.
                self._flush_memory()
                self._flush_naught_memory()
                # (i.3) Log data at the end of episode.
                if done:
                    if Config.TRAIN_MODE == 'policy':
                        self.episode_log_q.put((datetime.now(), self.env_id, total_reward, time_step))
            
            # (ii) Reset memory and history.
            self.memory.reset()
        print(f'Closed worker #{self.id} of type {self.env_id}')
        sys.stdout.flush()

    # ----------------- helper methods ---------------------------------------------------------------------

    def _reshape_observation(self, observation):
        """
        Reshapes an observation in form of an numpy array to a torch tensor.

        Args:
            observation (np.ndarray): Observation as received from a gym environment.
        
        Returns:
            observation (torch.Tensor): Observation which can be used as input to a pytorch model.

        """
        observation=observation.reshape((1, Config.INPUT_SIZE))
        observation=torch.Tensor(observation)
        return observation

    def _flush_memory(self):
        """
        Writes memory to shared experience in the appropriate batch size. 
        Memory is consumed but history remains.
        
        In 'policy' mode we train on observation-action pairs.
        In 'selection' mode we accumulate sequences of actions for a single observation.
        
        NOTE: + Minibatching is performed by agent.
        """
        action_label = 0
        action_seq = torch.zeros((1,Config.NUM_ACTIONS))
        if Config.TRAIN_MODE == 'selection':
            action_seq = torch.zeros((1,Config.NUM_ACTIONS*Config.NUM_ACTIONS_PREDICT))
        while len(self.memory.memory):
            action_label = action_label % Config.NUM_ACTIONS_PREDICT
            try:
                o_, a_, t_ = self.memory.memory.popleft()
                action_seq[0,Config.NUM_ACTIONS*action_label:Config.NUM_ACTIONS*(action_label+1)] = a_
                a_ = action_seq
                if action_label == 0:
                    o__ = o_
                if Config.TRAIN_MODE == 'selection':
                    action_label += 1
            except IndexError:
                break
            if self.batch_size == 0:
                self.o_batch = o__; self.a_batch = a_; self.t_batch = t_
            else:
                self.o_batch = torch.cat((self.o_batch, o__), dim=0)
                self.a_batch = torch.cat((self.a_batch, a_), dim=0)
                self.t_batch = torch.cat((self.t_batch, t_), dim=0)
            self.batch_size += 1
            if self.batch_size == Config.MINIBATCH_SIZE:
                batch = (self.o_batch, self.a_batch, self.t_batch)
                self.experience_q.put(batch)
                self.batch_size = 0

    def _add_naught_experience(self, observation, h_values):
        """
        Adds experience for non-optimal actions, i.e. actions which have a 
        chance to be sampled that is less than random.
    
        Args:
            observation (np.ndarray): Observation as received from a gym environment.
            h_values (torch.Tensor): List of weights which define the probability distribution  
                                     over the action space. 
        """
        h_values_naught = (h_values < 1./Config.NUM_ACTIONS).float()
        action = torch.multinomial(h_values_naught, 1).item()
        action_naught = self.actions[action].reshape(1, len(self.actions[action]))
        self.memory.add_experience_fake(observation, action_naught, 0.)

    def _flush_naught_memory(self):
        """
        Writes the fake memory to shared experience in the appropriate batch size.
        The memory is consumed. There is no history. This should only be relevant for
        'selection' mode.
        """
        action_label = 0
        action_seq = torch.zeros((1,Config.NUM_ACTIONS*Config.NUM_ACTIONS_PREDICT))
        while len(self.memory.memory_fake):
            action_label = action_label % Config.NUM_ACTIONS_PREDICT
            try:
                o_, a_, t_ = self.memory.memory_fake.popleft()
                action_seq[0,Config.NUM_ACTIONS*action_label:Config.NUM_ACTIONS*(action_label+1)] = a_
                a_ = action_seq
                if action_label == 0:
                    o__ = o_
                action_label += 1
            except IndexError:
                break
            if self.batch_size == 0:
                self.o_batch = o__; self.a_batch = a_; self.t_batch = t_
            else:
                self.o_batch = torch.cat((self.o_batch, o__), dim=0)
                self.a_batch = torch.cat((self.a_batch, a_), dim=0)
                self.t_batch = torch.cat((self.t_batch, t_), dim=0)
            self.batch_size += 1
            if self.batch_size == Config.MINIBATCH_SIZE:
                batch = (self.o_batch, self.a_batch, self.t_batch)
                self.experience_q.put(batch)
                self.batch_size = 0

