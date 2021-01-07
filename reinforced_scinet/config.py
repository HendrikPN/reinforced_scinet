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

class Config:
    ####################################################################################################################
    # SERVER SETTINGS
    ####################################################################################################################

    MAX_EPISODES = 5000000 # maximum number of episodes that are to be played

    MAX_QUEUE_SIZE = 40 # maximum queue size for experiences

    ENV_IDS = ['env1', 'env2', 'env3']#, 'env2', 'env3'] # the IDs of the different environment types

    WORKERS = {'env1': 21, 'env2': 21, 'env3': 21} # the number of workers associated with the env types

    TRAINERS = {'env1': 1, 'env2': 1, 'env3': 1} # the number of trainers associated with the env types

    PREDICTORS = {'env1': 2, 'env2': 2, 'env3': 2} # the number of predictors associated with the env types

    DEVICE = 'cuda:0' # the device on which the network is trained

    LOAD_AE = False # load autoencoder

    LOAD_AE_FILE = 'load_models/ae_select.pth' # relative path where autoencoder is loaded from

    LOAD_AE_OPTIMIZER = False # load optimizer for autoencoder

    LOAD_AE_OPT_FILE = 'load_models/optim_ae_select.pth' # relative path where AE optimizer is loaded from
    
    LOAD_PS = {'env1': True, 'env2': True, 'env3': True} # load PS model 

    LOAD_PS_FILE = {'env1': 'load_models/ps_pretrain_1.pth',
                    'env2': 'load_models/ps_pretrain_2.pth',
                    'env3': 'load_models/ps_pretrain_3.pth'} # relative path where PS is loaded from
    
    LOAD_PS_OPTIMIZER = {'env1': False, 'env2': False, 'env3': False} # load optimizer for PS

    LOAD_PS_OPT_FILE = {'env1': 'load_models/optim_ps_pretrain_1.pth',
                        'env2': 'load_models/optim_ps_pretrain_2.pth',
                        'env3': 'load_models/optim_ps_pretrain_3.pth'} # relative path where PS is loaded from

    LOAD_PREDICT = {'env1': False, 'env2': False, 'env3': False} # load PS model for prediction 

    LOAD_PREDICT_FILE = {'env1': 'load_models/ps_pretrain_1.pth',
                         'env2': 'load_models/ps_pretrain_2.pth',
                         'env3': 'load_models/ps_pretrain_3.pth'} # relative path where PS is loaded from
    
    LOAD_PREDICT_OPTIMIZER = {'env1': False, 'env2': False, 'env3': False} # load optimizer for predicting PS

    # relative path where PS optimizer is loaded from
    LOAD_PREDICT_OPT_FILE = {'env1': 'load_models/optim_ps_pretrain_1.pth',
                             'env2': 'load_models/optim_ps_pretrain_2.pth',
                             'env3': 'load_models/optim_ps_pretrain_3.pth'} 

    LOAD_DATA = True # load selection data

    PLAY = False # test PS mode
    
    PLAY_AND_LEARN = False # test PS mode with training

    RENDER = False # render PS performance

    ####################################################################################################################
    # GAME SETTINGS
    ####################################################################################################################
    import gym_subgridworld # the game we want to play

    ENV_NAME = 'subgridworld-v0' # ID of the environment

    #######################################    
    # 3D subsystem gridworld 
    # environment specific parameters

    ENV_PARAMS = {'env1': {'grid_size': [12, 12, 12], 
                           "plane": [1,1,0], 
                           'max_steps': 400, 
                           "plane_walls": [], 
                           'reward_pos': [6,11]
                           },     
                  'env2': {'grid_size': [12, 12, 12], 
                           "plane": [0,1,1], 
                           'max_steps': 400, 
                           "plane_walls": [], 
                           'reward_pos': [11,6]
                           },
                  'env3': {'grid_size': [12, 12, 12], 
                           "plane": [1,0,1], 
                           'max_steps': 400, 
                           "plane_walls": [], 
                           'reward_pos': [6,6]
                           },  
                  }
    
    INPUT_SIZE = 36 # FROM ENV SPEC: sum(grid_size)   
    
    LATENT_SIZE = 8 # FROM ENV SPEC: minimum number of neurons to reproduce img    
    
    NUM_ACTIONS = 6 # FROM ENV SPEC: number of available actions 

    ####################################################################################################################
    # MODEL SETTINGS
    ####################################################################################################################

    TRAIN_MODEL = True # training mode on/off

    TRAIN_MODE = 'selection' # 'policy' trains reinforcement learning agents | 'selection' trains agent for selection

    ############################################################################   
    # 3D subsystem gridworld 
    
    # parameters of the encoder
    ENC_DENSE_SIZE = [128, 128, 64, 32]

    # parameters of the decoder
    DEC_DENSE_SIZE = [32, 64, 128, 128, 128]

    # gathered set of encoder parameters 
    AE_PARAMS = (INPUT_SIZE, ENC_DENSE_SIZE, DEC_DENSE_SIZE, LATENT_SIZE)

    ENC_PARAMS = (INPUT_SIZE, ENC_DENSE_SIZE, LATENT_SIZE)

    # number of neurons in dense layers for PS
    PS_DENSE_SIZES = {'env1': [128, 128, 128, 128, 64, 32], 
                      'env2':  [128, 128, 128, 128, 64, 32], 
                      'env3': [128, 128, 128, 128, 64, 32]
                      } 
    
    # number of neurons in dense layers for predictor PS
    PS_PREDICT_DENSE_SIZES = {'env1': [64, 128, 128, 128, 128, 64, 32], 
                              'env2':  [64, 128, 128, 128, 128, 64, 32], 
                              'env3': [64, 128, 128, 128, 128, 64, 32]
                              } 

    DATA_TRAIN_SIZE = 200000 # the size of the training data used for selection

    MAX_TIME = 401 # max time steps within an episode until experience is generated for training

    NUM_ACTIONS_PREDICT = 1 # The max number of actions an agent will need to predict
    
    ############################################################################

    BATCH_SIZE = 200 # max number elements per training batch

    DATA_BATCH_SIZE = 200 # batch size for the random training data for autoencoder

    MINIBATCH_SIZE = 50 # max number element per minibatch as created by worker

    PREDICTION_BATCH_SIZE = 100 # max number of elements per prediction batch

    GLOW = 0.1 # PS glow parameter

    GAMMA = 0.01 # regularization for PS

    SOFTMAX = 0.5 # softmax beta parameter

    LEARNING_RATE_PREDICT = 0.0001 # optimizer learning rate for 'selection' mode

    LEARNING_RATE = 0.00005 # optimizer learning rate for 'policy' mode

    AGENT_DISCOUNT = 1.0 # the discount rate of the agent's loss

    AGENT_PREDICT_DISCOUNT = 1.0 # the discount rate of the agent's loss in 'selection' mode

    SELECTION_DISCOUNT = 0.04 # the discount rate of the selection neuron's loss

    MIN_DISCOUNT = 0.02 # the discount rate of the representation minimization

    REWARD_RESCALE = 10.0 # factor by which the discounted reward is rescaled

    AE_DISCOUNT = 10.0 # the discount rate of the AE's loss in 'selection' mode

    REWARD_CLIP_VALUE = 1.0e-7 # the value at which the discounted reward is neglected.

    POLICY_STEPS = 0 # max number of steps to be taken before training data for 'selection' mode is recorded

    ####################################################################################################################
    # STATISTICS SETTINGS
    ####################################################################################################################

    MAX_STATS_QUEUE_SIZE = 100 # maximum queue size for episode log

    RESULTS_FILE = 'results_log/results.txt' # relative path where results are saved

    RESULTS_LOSS_FILE = 'results_log/results_loss.txt' # relative path where AE results are saved

    SELECTION_FILE = 'results_log/selection.txt' # relative path where selection neurons are saved

    SELECTION_SAVE_FREQUENCY = 1000 # freqeuency at which selection neuron values are saved
    
    SAVE_MODELS = True # save the models

    AE_FILE = 'saved_models/ae_select.pth' # relative path where network is saved

    AE_OPT_FILE = 'saved_models/optim_ae_select.pth' # relative path where optimizer is saved
    
    PS_FILE = {'env1': 'saved_models/ps_select_1.pth',
               'env2': 'saved_models/ps_select_2.pth',
               'env3': 'saved_models/ps_select_3.pth'} # relative path where PS is saved

    PS_OPT_FILE = {'env1': 'saved_models/optim_ps_select_1.pth',
                   'env2': 'saved_models/optim_ps_select_2.pth',
                   'env3': 'saved_models/optim_ps_select_3.pth'} # relative path where optimizer for PS saved

    LOG_STATS_FREQUENCY = 10 # frequency at which stats are logged

    MODEL_SAVE_FREQUENCY = 10000 # frequency at which network is saved

    # time in sec until a waiting stat process (accessing log queue) is interrupted
    WAIT_STATS_INTERRUPT = 600
