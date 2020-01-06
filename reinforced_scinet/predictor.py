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

from datetime import datetime
import numpy as np
import sys
import warnings

import time

from config import Config

class PredictorProcess(Process):
    def __init__(self, agent, observation_q, prediction_qs, env_id, predictor_id):
        """
        Predictors gather observations from agents and make predictions.

        Args:
            agent (:class:`base_networks.DeepPS`): The deep PS model for RL.
            observation_q (:class:`mp.Queue`): Shared memory queue with observations of agents of the same type.
            prediction_qs (:obj:`list` of :class:`mp.Queue`): Shared memory queues containing predictions.
            env_id (str): The identifier for the environment type.
            predictor_id (int): The id of the trainer process.
        """
        super(PredictorProcess, self).__init__()
        self.agent = agent
        self.observation_q = observation_q
        self.prediction_qs = prediction_qs
        self.env_id = env_id
        self.id = predictor_id

        #int: Signal for process exit.
        self.exit_flag = Value('i', 0)
        #torch.Tensor of float: Array of actions in one-hot encoding.
        self.actions = torch.Tensor(np.eye(Config.NUM_ACTIONS)).to(Config.DEVICE)

    def predict(self, observation_batch):
        """ 
        Predict h-values with neural network model.

        Forward pass through the network. 
         
        Args: 
            observation_batch (torch.Tensor): Tensor which represents the batched states of the environments. 
 
        Returns: 
            h_values (torch.Tensor): List of weights which define h-values in the PS framework. 
        """ 
        
        # Forward pass.
        with torch.no_grad():
            h_values = torch.empty(0, device=observation_batch.device)
            for action in self.actions:  
                h_val_batch = self.agent.forward_no_selection(observation_batch, action.reshape(1, len(action)))  
                h_values = torch.cat((h_values, h_val_batch), dim=1)  

        return h_values

    def run(self):
        """
        Runs the process predictor.
        (i) Gets observation data from prediction queue.
        (ii) Send data to device.
        (iii) Gathers predictions with shared model.
        (iv) Distributes predictions to agents.
        """
        print(f'Spawning predictor #{self.id} for type {self.env_id}')
        sys.stdout.flush()
        while not self.exit_flag.value:

            # (i) Get observation data.
            id_batch = torch.zeros(Config.PREDICTION_BATCH_SIZE, dtype=torch.int32)
            observation_batch = torch.zeros((Config.PREDICTION_BATCH_SIZE, Config.INPUT_SIZE))
            # "EOFError" on some clusters can be removed by mp.Event() or mp.set_sharing_strategy('file_system')
            Id, observation = self.observation_q.get() # fix error with mp.set_sharing_strategy('file_system')
            id_batch[0], observation_batch[0] = Id.item(), observation

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.observation_q.empty():
                Id, observation = self.observation_q.get()
                id_batch[size], observation_batch[size] = Id.item(), observation
                size += 1

            # (ii) Resize data and Transfer to device.
            id_batch = id_batch[:size]
            observation_batch = observation_batch[:size].to(Config.DEVICE)

            #(iii) Make prediction.
            h_values = self.predict(observation_batch).to('cpu')

            # (iv) Add to queue.
            for index, i in enumerate(id_batch):
                # "RuntimeError: received 0 items of ancdata" can be fixed with mp.set_sharing_strategy('file_system')
                self.prediction_qs[i].put(h_values[index]) # fix error with mp.set_sharing_strategy('file_system')

        print(f'Closed predictor #{self.id}')
        sys.stdout.flush()
