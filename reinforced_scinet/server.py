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
from torch.multiprocessing import Queue, Value, Process
import torch.optim as optim
import sys, os
import warnings
import time

from config import Config
from trainer import TrainerProcess
from worker import WorkerProcess
from predictor import PredictorProcess
from data import DataProcess
from statistics import StatProcess
from networks.base_networks import DenseAutoencoder, DeepPS

class Server:
    def __init__(self):
        """
        Server that spawns worker, trainer, predictors, data and statistics processes and exits them.
        """
        if not Config.TRAIN_MODE in ['policy', 'selection']:
            raise NotImplementedError("We currently only support 'policy' and 'selection' training modes. "+
                                      f"You specified '{Config.TRAIN_MODE}'.")

        if not (torch.cuda.is_available() and 'cuda' in Config.DEVICE):
            warnings.warn('Not running on a GPU will significantly slow down learning and'+
                          'has not been thoroughly tested.')

        #:obj:`dict` of :class:`mp.Queue`: Dictionary of shared memory queues containing experiences for training. 
        self.experience_q = {} 
        for env_id in Config.ENV_IDS: 
            self.experience_q[env_id] = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        #:obj:`dict` of :class:`mp.Queue`: Dictionary of shared memory queues containing observations for predictions. 
        self.observation_q = {}
        for env_id in Config.ENV_IDS:
            self.observation_q[env_id] = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        #:obj:`dict` of :obj:`list` of :class:`mp.Queue`: Dictionary of list of shared memory queues with predictions.
        self.prediction_qs = {}
        for env_id in Config.ENV_IDS:
            prediction_q = []
            for w in range(Config.WORKERS[env_id]):
                prediction_q.append(Queue(maxsize=1))
            self.prediction_qs[env_id] = prediction_q

        #:class:`base_networks.DenseAutoencoder`: Autoencoder to produce latent representation.
        self.autoencoder = DenseAutoencoder(*Config.AE_PARAMS).to(Config.DEVICE)
        self.autoencoder.share_memory() if Config.DEVICE == 'cpu' else None
        #:class:`optim.Adam`: Optimizer for Autoencoder model.
        self.optimizer_ae = optim.Adam(self.autoencoder.parameters(),
                                    lr=Config.LEARNING_RATE_PREDICT, amsgrad=True, weight_decay=1e-5
                                    )

        #:obj:`dict`: Dictionary of DPS models for RL.
        self.agents = {}  
        for env_id in Config.ENV_IDS:
            #:class:`base_networks.DeepPS`: Deep PS model for RL.
            self.agents[env_id] = DeepPS(Config.INPUT_SIZE, Config.NUM_ACTIONS, 
                                         dim_dense=Config.PS_DENSE_SIZES[env_id],
                                         is_predictor=False
                                        ).to(Config.DEVICE) 
            self.agents[env_id].share_memory() if Config.DEVICE == 'cpu' else None

        #:obj:`dict`: Dictionary of optimizers for DPS models.
        self.optimizers_ps = {}
        for env_id in Config.ENV_IDS: 
            #:class:`optim.Adam`: Optimizer for DPS model.
            self.optimizers_ps[env_id] = optim.Adam(self.agents[env_id].parameters(), 
                                                    lr=Config.LEARNING_RATE, 
                                                    amsgrad=True, weight_decay=1e-5
                                                   )

        if Config.TRAIN_MODE == 'selection':
            for env_id in Config.ENV_IDS:
                #:class:`base_networks.DeepPS`: DPS model for representation learning.
                self.agents[env_id + '_sel'] = DeepPS(Config.LATENT_SIZE, Config.NUM_ACTIONS*Config.NUM_ACTIONS_PREDICT, 
                                                     dim_dense=Config.PS_PREDICT_DENSE_SIZES[env_id],
                                                     is_predictor=True
                                                    ).to(Config.DEVICE)
                self.agents[env_id + '_sel'].share_memory() if Config.DEVICE == 'cpu' else None
                #:class:`optim.Adam`: DPS model's optimizer.
                self.optimizers_ps[env_id + '_sel'] = optim.Adam(self.agents[env_id+'_sel'].parameters(), 
                                        lr=Config.LEARNING_RATE_PREDICT, 
                                        amsgrad=True, weight_decay=1e-5
                                        )

        #:obj:`dict`: Data for prediction of agents.
        self.select_data = {}

        if Config.DEVICE == 'cpu':
            self._share_adam_optimizers() # put optimizers into shared memory
        self._load_states() #load state dictionaries
        self._fix_selections() #fix selection neurons

        #:class:`mp.Value`: Shared memory value which counts the number of trainings.
        self.training_count = Value('i', 0)
        #int: Number of available worker processes.
        self.num_workers = self._count_worker()
        #list: Empty list for worker processes.
        self.workers = []
        #list: Empty list for trainer processes.
        self.trainers = []
        #list: Empty list for prediction processes.
        self.predictors = []
        #list: Empty list for data processes.
        self.data = []

        models = ()
        if Config.TRAIN_MODE == 'selection':
            models = (self.agents[env_id+'_sel'] for env_id in Config.ENV_IDS)
        #:class:`Process`: Process for statistics that saves and logs statistics.
        self.stats = StatProcess(*models)
    
    def add_worker(self, env_id, agent_id):
        """
        Starts a worker process.

        Args: 
            env_id (str): The id of the environment instance this worker is interacting with.
        """
        self.workers.append(WorkerProcess(self.experience_q[env_id], self.prediction_qs[env_id][agent_id], 
                                          self.observation_q[env_id], env_id, self.stats.episode_log_q, agent_id))
        self.workers[-1].start()

    def rm_worker(self, index):
        """
        Passes exit flag to a specific worker process.
        """
        self.workers[index].exit_flag.value = 1
    
    def add_trainer(self, env_id):
        """
        Starts a trainer process.

        Args: 
            env_id (str): The id of the environment/agent instance this trainer is using.
        """
        if Config.TRAIN_MODE == 'policy':
            self.trainers.append(TrainerProcess(self.autoencoder, self.optimizer_ae, 
                                                self.agents[env_id], self.optimizers_ps[env_id], 
                                                env_id, (),
                                                self.experience_q[env_id], self.training_count, 
                                                self.stats.ae_loss_log_q, len(self.trainers)
                                                ))
        elif Config.TRAIN_MODE == 'selection':
            self.trainers.append(TrainerProcess(self.autoencoder, self.optimizer_ae, 
                                                self.agents[env_id+'_sel'], self.optimizers_ps[env_id+'_sel'], 
                                                env_id, self.select_data[env_id],
                                                self.experience_q[env_id], self.training_count, 
                                                self.stats.ae_loss_log_q, len(self.trainers)
                                                ))
        self.trainers[-1].start()
    
    def rm_trainer(self):
        """
        Removes last trainer process.
        """
        self.trainers[-1].exit_flag.value = 1
        self.trainers[-1].join()
        self.trainers.pop()
    
    def add_predictor(self, env_id):
        """
        Starts a prediction process.
        """
        self.predictors.append(PredictorProcess(self.agents[env_id],
                                                self.observation_q[env_id], self.prediction_qs[env_id], 
                                                env_id, len(self.predictors)
                                                ))
        self.predictors[-1].start()
    
    def rm_predictor(self, index):
        """
        Passes exit flag to a specific prediction process.
        """
        self.predictors[index].exit_flag.value = 1
    
    def add_data(self, env_id):
        """
        Start a data generation process.
        """
        self.data.append(DataProcess(self.experience_q[env_id], env_id, len(self.data)))
        self.data[-1].start()

    def wait_data(self):
        """
        Waits for last data processes to finish.
        """
        self.data[-1].join()
        self.data.pop()

    def main(self):
        """
        Runs the main program.

        (i) Starts the statistics process which gather and save statistics of all workers and trainers.
        (ii) Starts the prediction process which makes predictions with the shared models.
        (iii) Starts the worker processes which interact with their respective environments.
        (iv) Gets the data used for training in 'selection' mode by either loading it or starting data processes.
        (v) Starts the trainer processes which train the shared models.
        (vi) Saves models to file system every n episodes.
        (vii) Ends processes after a certain number of episodes has finished.
        """
        # (i) Start Process: Statistics
        self.stats.start()

        # (ii) Start Process: Predictors
        if Config.TRAIN_MODE == 'policy' or (Config.TRAIN_MODE == 'selection' and not Config.LOAD_DATA):
            for env_id in Config.ENV_IDS:
                for _ in range(Config.PREDICTORS[env_id]):
                    self.add_predictor(env_id)

        # (iii) Start Process: Workers
        if Config.TRAIN_MODE == 'policy' or (Config.TRAIN_MODE == 'selection' and not Config.LOAD_DATA):
            for env_id in Config.ENV_IDS:
                for i in range(Config.WORKERS[env_id]):
                    self.add_worker(env_id, i)

        # (iv) Load selection data or start Process: Data
        if Config.TRAIN_MODE == 'selection':
            if not Config.LOAD_DATA:
                for env_id in Config.ENV_IDS:
                    self.add_data(env_id)
                while self.data:
                    self.wait_data()
                self._close_processes()
            self._load_selection_data()
            

        # (v) Start Process: Trainers 
        for env_id in Config.ENV_IDS:
            for _ in range(Config.TRAINERS[env_id]):
                self.add_trainer(env_id)

        if Config.TRAIN_MODE == 'selection':
            self.stats.episode_count.value = 0
        while self.stats.episode_count.value < Config.MAX_EPISODES:

            # (vi) Save all models to file system.
            if Config.SAVE_MODELS:
                self._save_models()
            
            time.sleep(0.1)

        # (vii) Remove all Processes.
        self._close_processes()
        print('All processes have been closed, terminating statistics and end program.')
        # Terminate stats which is likely waiting for some queue.
        self.stats.terminate()

    # ----------------- helper methods ---------------------------------------------------------------------

    def _count_worker(self):
        """
        Counts the number of workers in Config.

        Returns:
            counter (int): Number of workers.
        """
        counter = 0
        for env_id in Config.ENV_IDS:
            for i in range(Config.WORKERS[env_id]):
                counter += 1
        return counter

    def _save_models(self):
        """
        Saves the models' parameters to the file system.
        """
        if self.stats.model_save.value:
            torch.save(self.autoencoder.state_dict(), Config.AE_FILE)
            torch.save(self.optimizer_ae.state_dict(), Config.AE_OPT_FILE)
            if Config.TRAIN_MODE == 'policy':
                for env_id in Config.ENV_IDS:
                    torch.save(self.agents[env_id].state_dict(), Config.PS_FILE[env_id])
                    torch.save(self.optimizers_ps[env_id].state_dict(), Config.PS_OPT_FILE[env_id])
                self.stats.model_save.value = 0
            elif Config.TRAIN_MODE == 'selection':
                for env_id in Config.ENV_IDS:
                    torch.save(self.agents[env_id+'_sel'].state_dict(), Config.PS_FILE[env_id])
                    torch.save(self.optimizers_ps[env_id+'_sel'].state_dict(), Config.PS_OPT_FILE[env_id])
                self.stats.model_save.value = 0

    def _share_adam_optimizers(self):
        """
        Puts the parameters of all ADAM optimizers into shared memory.

        TODO:   + improve code
        """
        for group in self.optimizer_ae.param_groups:
            for p in group['params']:
                state = self.optimizer_ae.state[p]
                if len(state) > 0:
                    # state['step'].share_memory_()
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()
                    if group['amsgrad']:
                        state['max_exp_avg_sq'].share_memory_()
        for env_id in Config.ENV_IDS:
            for group in self.optimizers_ps[env_id].param_groups:
                for p in group['params']:
                    state = self.optimizers_ps[env_id].state[p]
                    if len(state) > 0:
                        # state['step'].share_memory_()
                        state['exp_avg'].share_memory_()
                        state['exp_avg_sq'].share_memory_()
                        if group['amsgrad']:
                            state['max_exp_avg_sq'].share_memory_()
        if Config.TRAIN_MODE == 'selection':
            for env_id in Config.ENV_IDS:
                for group in self.optimizers_ps[env_id+'_sel'].param_groups:
                    for p in group['params']:
                        state = self.optimizers_ps[env_id+'_sel'].state[p]
                        if len(state) > 0:
                            # state['step'].share_memory_()
                            state['exp_avg'].share_memory_()
                            state['exp_avg_sq'].share_memory_()
                            if group['amsgrad']:
                                state['max_exp_avg_sq'].share_memory_()

    def _load_states(self):
        """
        Loads the state dictionaries as specified by the :class:`Config`.

        TODO:   + improve code

        (i) Load the autoencoder if required.
        (ii) Load optimizer for autoencoder if required.
        (iii) Load deep PS models for RL if required.
        (iv) Load optimizers for PS models for RL if required.
        (v) Load deep PS models for prediction and selection if required.
        (vi) Load optimizers for PS models for prediction if required.
        """
        # (i) Load autoencoder model.
        if Config.LOAD_AE:
            loaded_dict = torch.load(Config.LOAD_AE_FILE, map_location=torch.device(Config.DEVICE))
            self.autoencoder.load_state_dict(loaded_dict)

        # (ii) Load optimizer for autoencoder.
        if Config.LOAD_AE_OPTIMIZER:
            loaded_opt = torch.load(Config.LOAD_AE_OPT_FILE, map_location=torch.device(Config.DEVICE)) 
            self.optimizer_ae.load_state_dict(loaded_opt)
        
        for env_id in Config.ENV_IDS:
            # (iii) Load DPS models for RL.
            if Config.LOAD_PS[env_id]:
                loaded_dict = torch.load(Config.LOAD_PS_FILE[env_id], map_location=torch.device(Config.DEVICE))
                self.agents[env_id].load_state_dict(loaded_dict)
            # (iv) Load optimizer for DPS models for RL.
            if Config.LOAD_PS_OPTIMIZER[env_id]:
                loaded_opt = torch.load(Config.LOAD_PS_OPT_FILE[env_id], map_location=torch.device(Config.DEVICE)) 
                self.optimizers_ps[env_id].load_state_dict(loaded_opt)
            # (v) Load DPS models for prediction.
            if Config.LOAD_PREDICT[env_id]:
                loaded_dict = torch.load(Config.LOAD_PREDICT_FILE[env_id], map_location=torch.device(Config.DEVICE))
                self.agents[env_id+'_sel'].load_state_dict(loaded_dict)
            # (vi) Load optimizer for DPS models for prediction.
            if Config.LOAD_PREDICT_OPTIMIZER[env_id]:
                loaded_opt = torch.load(Config.LOAD_PREDICT_OPT_FILE[env_id], map_location=torch.device(Config.DEVICE)) 
                self.optimizers_ps[env_id+'_sel'].load_state_dict(loaded_opt)
    
    def _fix_selections(self):
        """
        Fixes selection neurons as specified by the :class:`Config`.
        """
        if Config.TRAIN_MODE == 'policy':
            for env_id in Config.ENV_IDS:  
                self.agents[env_id].selection.selectors.requires_grad = False
        elif Config.TRAIN_MODE == 'selection':
            for env_id in Config.ENV_IDS:
                self.agents[env_id].selection.selectors.requires_grad = False
                self.agents[env_id+'_sel'].selection.selectors.requires_grad = True

    def _load_selection_data(self):
        """
        Loads selection data from file system for training of selection neurons.
        """
        for env_id in Config.ENV_IDS:
            o_data = torch.load('data/'+Config.ENV_NAME+'_o_'+env_id+'.pth')
            a_data = torch.load('data/'+Config.ENV_NAME+'_a_'+env_id+'.pth')
            t_data = torch.load('data/'+Config.ENV_NAME+'_t_'+env_id+'.pth')
            self.select_data[env_id] = (o_data, a_data, t_data)

    def _close_processes(self):
        """
        Closes all processes by passing exit flags and generating dummy data.

        The order of closing is as follows
        (i) Trainers
        (ii) Workers
        (iii) Predictors
        (iv) Statistics
        """
        print('Start closing processes...')
        sys.stdout.flush()

        # (0) Pass exit flag to statistics to ignore errors.
        self.stats.exit_flag.value = 1

        # (i) Closing trainers.
        while self.trainers:
            self.rm_trainer()
        print('Training processes have been closed.')
        sys.stdout.flush()

        # (ii) Closing workers.
        for i in range(len(self.workers)):
            self.rm_worker(i)
        print('All worker processes have received exit flags.')
        # Flushes experience queue while workers are still alive.
        while self._check_alive('worker'):
            for env_id in Config.ENV_IDS:
                if self.experience_q[env_id].full():
                    try:
                        self.experience_q[env_id].get()
                    except (FileNotFoundError, ConnectionResetError) as error:
                        warnings.warn(f'Ignored error while trying to flush a full queue: {error}')
            time.sleep(0.1)
        print('All worker processes have been closed.')

        # (iii) Closing predictors.
        for j in range(len(self.predictors)):
            self.rm_predictor(j)
        print('All prediction processes have received exit flags.')
        sys.stdout.flush()
        # Flushes prediction queue and supplies dummy data while predictors are still alive.
        while self._check_alive('predictor'):
            for env_id in Config.ENV_IDS:
                if self.observation_q[env_id].empty():
                    self.observation_q[env_id].put((torch.LongTensor([0]), torch.zeros((1, Config.INPUT_SIZE))))
                for w in range(Config.WORKERS[env_id]):
                    if self.prediction_qs[env_id][w].full():
                        try:
                            self.prediction_qs[env_id][w].get()
                        except (FileNotFoundError, ConnectionResetError) as error:
                            warnings.warn(f'Ignored error while trying to flush a full queue: {error}')
            time.sleep(0.1)
        print('All prediction processes have been closed.')


    def _check_alive(self, process):
        """
        Checks whether there is any process of a certain type that is still alive.

        Args:
            process (str): The type of process to be considered. Either 'predictor' or 'worker'.

        Returns:
            alive (bool): Whether or not any process is still alive.
        """
        alive = False
        if process == 'predictor':
            delete = []
            for i in range(len(self.predictors)):
                if not self.predictors[i].is_alive():
                    delete.append(i)
                else:
                    alive = True
            for j in reversed(delete):
                del self.predictors[j]
        elif process == 'worker':
            delete = []
            for i in range(len(self.workers)):
                if not self.workers[i].is_alive():
                    delete.append(i)
                else:
                    alive = True
            for j in reversed(delete):
                del self.workers[j]
        return alive
