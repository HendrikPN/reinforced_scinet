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
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import sys
from datetime import datetime
import time

from config import Config

class TrainerProcess(Process):
    def __init__(self, autoencoder, optimizer_ae, agent, optimizer_ps, env_id, 
                 select_data, experience_q, training_count, ae_loss_log_q, trainer_id):
        """
        Trainers gather experiences and train respective models model. 

        Args:
            autoencoder (:class:`base_networks.DenseAutoencoder`): The Server.autoencoder model.
            optimizer_ae (:class:`optim.Adam`): The Server.optimizer_ae for encoder.
            agent (:class:`base_networks.DeepPS`): The deep PS model for RL.
            optimizer_ps (:class:`optim.Adam`): The Server.optimizer_ps for deep PS.
            env_id (str): The id of the environment/agent instance this trainer is using.
            select_data (tuple): The data used for training in 'selection' mode.
            experience_q (:class:`mp.Queue`): Shared memory queue containing experiences for training.
            training_count (:class:`mp.Value`): Shared memory value which counts the number of trainings.
            ae_loss_log_q (:class:`mp.Queue`): Shared memory queue containing loss of decoder.
            trainer_id (int): The id of the trainer process.
        """
        super(TrainerProcess, self).__init__()
        self.autoencoder = autoencoder
        self.optimizer_ae = optimizer_ae
        self.agent = agent
        self.optimizer_ps = optimizer_ps
        self.env_id = env_id
        self.experience_q = experience_q
        self.training_count = training_count
        self.ae_loss_log_q = ae_loss_log_q
        self.id = trainer_id

        if Config.TRAIN_MODE == 'selection':
            o_batch, a_batch, t_batch = select_data
            #torch.Tensor: The observation training data set.
            self.o_batch = o_batch.to(Config.DEVICE)
            #torch.Tensor: The action training data set.
            self.a_batch = a_batch.to(Config.DEVICE)
            #torch.Tensor: The target training data set.
            self.t_batch = t_batch.to(Config.DEVICE)

        #int: Signal for process exit.
        self.exit_flag = Value('i', 0)

    
    def train(self, observation_batch, action_batch, target_batch):
        """
        Trains the shared models with batches.
        (0) Creates batch of random numbers in 'selection' mode.
        (i) Move batches to device.
        (ii) Train the agent and decoder model.
        (iii) Count trainings.
        (iv) Save losses to logger queue.

        Args:
            observation_batch (torch.Tensor): Batched observations from experience queue.
            action_batch (torch.Tensor): Batched actions from experience queue.
            target_batch (torch.Tensor): Batched target values from experience queue.
        """
        # (0) Create batch of random numbers for selection neurons
        if Config.TRAIN_MODE == 'selection':
            rand_batch = torch.randn((observation_batch.size(0), Config.LATENT_SIZE),device=torch.device(Config.DEVICE))
        # (i) Move batches to device if required
        elif Config.TRAIN_MODE == 'policy':
            observation_batch = observation_batch.to(Config.DEVICE)
            action_batch = action_batch.to(Config.DEVICE)
            target_batch = target_batch.to(Config.DEVICE)

        # (ii) Train device.
        # (ii.1) Forward
        if Config.TRAIN_MODE == 'policy':
            out_agent = self.agent.forward_no_selection(observation_batch, action_batch)
        elif Config.TRAIN_MODE == 'selection':
            latent = self.autoencoder.encoder(observation_batch)
            out_dec = self.autoencoder.decoder(latent)
            out_agent = self.agent(latent, action_batch, rand_batch)

        # (ii.2) Calculate losses.
        loss_agent = F.smooth_l1_loss(out_agent, target_batch.float())
        if Config.TRAIN_MODE == 'policy':
            loss = loss_agent * Config.AGENT_DISCOUNT
        elif Config.TRAIN_MODE == 'selection':
            loss_ae = F.binary_cross_entropy(out_dec, observation_batch)
            loss_select = - self.agent.selection.selectors.sum()
            loss = loss_agent * Config.AGENT_PREDICT_DISCOUNT + \
                   loss_select * Config.SELECTION_DISCOUNT + \
                   loss_ae * Config.AE_DISCOUNT

        # (ii.4) Train
        self.optimizer_ps.zero_grad()
        if Config.TRAIN_MODE == 'selection':
            self.optimizer_ae.zero_grad()
        loss.backward()
        self.optimizer_ps.step()
        if Config.TRAIN_MODE == 'selection':
            self.optimizer_ae.step()

        # (iii) Count trainings for statistics.
        with self.training_count.get_lock():
            self.training_count.value += observation_batch.size(0)
            count = self.training_count.value

        # (iv) Save agent loss.
        if Config.TRAIN_MODE == 'selection':
            loss_stats = loss_agent.detach().to('cpu')
            loss_stats_ae = loss_ae.detach().to('cpu')
            count = self.training_count.value + observation_batch.size(0)
            self.ae_loss_log_q.put((datetime.now(), 'Policy', self.env_id, loss_stats, count))
            self.ae_loss_log_q.put((datetime.now(), 'Autoencoder', self.env_id, loss_stats_ae, count))

    def run(self):
        """
        Runs the process trainer.
        (i.1) Gets training data from experience queue for RL or...
        (i.2) Get training data for prediction.
        (ii) Trains shared models.
        """
        print(f'Spawning trainer #{self.id} for type {self.env_id}')
        sys.stdout.flush()
        while not self.exit_flag.value:
            # (i) Get training data.
            # (i.1) Get training data in an appropriate batch for RL. 
            if Config.TRAIN_MODE == 'policy':
                batch_size = 0
                while batch_size < Config.BATCH_SIZE:
                    o_, a_, t_ = self.experience_q.get()
                    if batch_size == 0:
                        o__ = o_; a__ = a_; t__ = t_
                    else:
                        o__ = torch.cat((o__, o_), dim=0)
                        a__ = torch.cat((a__, a_), dim=0)
                        t__ = torch.cat((t__, t_), dim=0)
                    batch_size = o__.size(0)

            # (i.2) Get training data in an appropriate batch for prediction. 
            if Config.TRAIN_MODE == 'selection':
                random_indices = torch.randint(self.o_batch.size(0), 
                                (Config.DATA_BATCH_SIZE,), 
                                dtype=torch.int64, 
                                device=torch.device(Config.DEVICE)
                                )
                o__ = torch.index_select(self.o_batch, 0, random_indices)
                a__ = torch.index_select(self.a_batch, 0, random_indices)
                t__ = torch.index_select(self.t_batch, 0, random_indices)

            # (ii) Train model.
            if Config.TRAIN_MODEL:
                self.train(o__, a__, t__)

        print(f'Closed trainer #{self.id}')
        sys.stdout.flush()
