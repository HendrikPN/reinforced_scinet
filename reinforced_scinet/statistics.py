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
from torch.multiprocessing import Process, Queue, Value
import queue
from datetime import datetime
import sys
import time
import warnings

from config import Config

class StatProcess(Process):
    def __init__(self, *args):
        """
        Statistics process saves the statistics obtained from workers.
        In particular, the shared models are saved every Config.MODEL_SAVE_FREQUENCY episodes.
        Moreover, some statistics are logged every Config.LOG_STATS_FREQUENCY episodes.
        """
        super(StatProcess, self).__init__()
        self.episode_log_q = Queue(maxsize=Config.MAX_STATS_QUEUE_SIZE)
        self.ae_loss_log_q = Queue(maxsize=Config.MAX_STATS_QUEUE_SIZE)
        self.episode_count = Value('i', 0)
        self.model_save = Value('i', 0)
        self.exit_flag = Value('i', 0)

        #:obj:`dict`: Dictionary of DPS models for RL.
        self.agents = {}
        for model, env_id in zip(args, Config.ENV_IDS):
            self.agents[env_id] = model
        #float: Time at start for logging.
        self._start_time = time.time()
    
    def run(self):
        """
        Runs the statistics process.
        (i) Get statistics from shared memory queue.
            If process cannot find data for some time, it may time out.
        (ii) Saves statistics to file.
        (iii) Increments episode count.
        (iv) Communicates to server that model may be saved after n episodes.
        (v) Logs current episode statistics after m episodes.
        """
        print('Start gathering statistics.')
        sys.stdout.flush()
        with open(Config.RESULTS_FILE, 'a') as results_logger, \
             open(Config.RESULTS_LOSS_FILE, 'a') as loss_logger, \
             open(Config.SELECTION_FILE, 'a') as select_logger:
            while True:
                # (i) Get statistics. Ignore errors when exiting.
                try:
                    if Config.TRAIN_MODE == 'policy':
                        # Get episode log.
                        episode_time, env_id, \
                        total_reward, length = self.episode_log_q.get(timeout=Config.WAIT_STATS_INTERRUPT)
                    loss_q_empty = self.ae_loss_log_q.empty()
                    if Config.TRAIN_MODE == 'selection' and not loss_q_empty:
                        # Get loss log.
                        training_time, loss_type, env_id_loss, \
                        loss, training_count = self.ae_loss_log_q.get(timeout=Config.WAIT_STATS_INTERRUPT)
                        self.episode_count.value += 1
                except (FileNotFoundError, ConnectionResetError) as error:
                    if self.exit_flag.value:
                        warnings.warn(f'Ignored error in statistics while trying to close: {error}')
                    else:
                        raise error                 

                # (ii) Saves statistics.
                if Config.TRAIN_MODE == 'policy':
                    # Save episode log.
                    results_logger.write('%s, %s, %10.4f, %d\n' 
                                        % (episode_time.strftime("%Y-%m-%d %H:%M:%S"), 
                                        env_id, total_reward, length))
                    results_logger.flush()
                if Config.TRAIN_MODE == 'selection' and not loss_q_empty:
                    # Save loss log.
                    loss_logger.write('%s, %s, %s, %d, %10.8f\n' 
                                      % (training_time.strftime("%Y-%m-%d %H:%M:%S"), 
                                      loss_type, env_id_loss, training_count, loss))
                    loss_logger.flush()
                if (Config.TRAIN_MODE == 'selection' and 
                    self.episode_count.value % Config.SELECTION_SAVE_FREQUENCY == 0 and 
                    self.episode_count.value != 0 and not loss_q_empty):
                    # Save selection log.
                    for env_id in Config.ENV_IDS:
                        selection = self.agents[env_id].selection.selectors.data.tolist()
                        select_logger.write('%s, %s\n' % (env_id, str(selection)))
                    select_logger.flush()

                # (iii) Increments episode count.
                if Config.TRAIN_MODE == 'policy':
                    self.episode_count.value += 1

                # (iv) Tells server to save model.
                if Config.SAVE_MODELS and self.episode_count.value % Config.MODEL_SAVE_FREQUENCY == 0:
                    self.model_save.value = 1

                # (v) Logs some statistiscs.
                if Config.TRAIN_MODE == 'policy' and self.episode_count.value % Config.LOG_STATS_FREQUENCY == 0:
                    print(
                        '[ Time: %8d ]         '
                        '[ Environment type: %5s ]    '
                        '[ Episode #%8d with total Score %10.4f and length %8d. ]'
                        % (int(time.time()-self._start_time),
                          env_id,
                          self.episode_count.value, total_reward, length)
                    )
                if Config.TRAIN_MODE == 'selection' and not loss_q_empty:
                    print(
                        '[ Training #%12d ] '
                        '[ Episode #%8d ] '
                        '[ Loss for type: %6s ]    '
                        '[ Trainer for type: %5s ]    '
                        '[ Loss: %10.8f. ]'
                        % (training_count,
                        self.episode_count.value,
                        loss_type,
                        env_id_loss,
                        loss)
                    )
                sys.stdout.flush()
        print('Statistics have been closed.')
        sys.stdout.flush()
