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
from torch.multiprocessing import Process
import sys, os

from config import Config

class DataProcess(Process):
    def __init__(self, experience_q, env_id, data_id):
        """
        This process generates the data from a trained RL agent to train selection neurons 
        in an environment.

        Args:
            experience_q (mp.Queue): Shared memory queue containing experiences across workers of the same type.
            env_id (str): The id of the environment instance this data process is creating data for.
            data_id (int): The id of the data process.
        """
        super(DataProcess, self).__init__()
        self.experience_q = experience_q
        self.env_id = env_id
        self.id = data_id

        # torch.Tensor: The training data for selection.
        self.o_train_data = torch.empty((Config.DATA_TRAIN_SIZE, Config.INPUT_SIZE))
        # torch.Tensor: The training data for actions.
        self.a_train_data = torch.empty((Config.DATA_TRAIN_SIZE, Config.NUM_ACTIONS * Config.NUM_ACTIONS_PREDICT))
        # torch.Tensor: The training data for selection.
        self.t_train_data = torch.empty((Config.DATA_TRAIN_SIZE, 1), dtype=torch.double)

    def run(self):
        """
        Here we generate the data which is used for training of selection neurons.
        The process is as follows,

        (i) Generates data that lies within the policy of an agent. Colelcts data from experience queue.
        (ii) Save data to the file system.
        """
        print(f'Starting training data generation #{self.id}.')
        sys.stdout.flush()
        # (i) Get data for training from optimal policy of agents.
        self._get_data()

        # (ii) Save data
        torch.save(self.o_train_data, 'data/'+Config.ENV_NAME+'_o_'+self.env_id+'.pth')
        torch.save(self.a_train_data, 'data/'+Config.ENV_NAME+'_a_'+self.env_id+'.pth')
        torch.save(self.t_train_data, 'data/'+Config.ENV_NAME+'_t_'+self.env_id+'.pth')

        print(f'Finished data generation #{self.id}')
        sys.stdout.flush()

    # ----------------- helper methods ---------------------------------------------------------------------

    def _get_data(self):
        """
        Creates data which lies within the optimal policy of a learned agent.
        To this end, we just collect the data from the `experience_q` of the workers.
        """
        index = 0
        batch_cut = Config.DATA_TRAIN_SIZE
        while index < Config.DATA_TRAIN_SIZE:
            o_batch, a_batch, t_batch = self.experience_q.get()
            batch_size = len(o_batch)
            self.o_train_data[index:index+batch_size] = o_batch[:batch_cut]
            self.a_train_data[index:index+batch_size] = a_batch[:batch_cut]
            self.t_train_data[index:index+batch_size] = t_batch[:batch_cut]
            index += batch_size
            batch_cut -= batch_size
