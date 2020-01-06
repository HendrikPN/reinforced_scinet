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

import torch.multiprocessing as mp
# the following might fix an EOFError and RuntimeError in `PredictorProcess` if applicable
# mp.set_sharing_strategy('file_system')
import sys

from config import Config
from server import Server

# Parse arguments.
for i in range(1, len(sys.argv)):
    # Config arguments should be in format of CONFIG=Value
    # For setting booleans to False use CONFIG=
    x, y = sys.argv[i].split('=')
    setattr(Config, x, type(getattr(Config, x))(y))

# Checking whether we are playing/learning with agent.
if Config.PLAY:
    Config.RENDER = True
    Config.ENV_IDS = ['env1', 'env2', 'env3'] 
    Config.WORKERS = {'env1': 1, 'env2': 1, 'env3': 1}
    Config.TRAIN_MODEL = False
    Config.SAVE_MODELS = False
    Config.LOAD_AE = False
    Config.LOAD_AE_OPTIMIZER = False
    Config.LOAD_PS = {'env1': True, 'env2': True, 'env3': True} 
    Config.LOAD_PS_OPTIMIZER = {'env1': False, 'env2': False, 'env3': False} 
    Config.TRAINERS = {'env1': 1, 'env2': 1, 'env3': 1}
    Config.PREDICTORS = {'env1': 1, 'env2': 1, 'env3': 1}
    Config.DEVICE = 'cpu'
    Config.PLAY_AND_LEARN = False
elif Config.PLAY_AND_LEARN:
    Config.PLAY = True
    Config.RENDER = True
    Config.ENV_IDS = ['env1'] 
    Config.WORKERS = {'env1': 1, 'env2': 1, 'env3': 1}
    Config.TRAIN_MODEL = True
    Config.SAVE_MODELS = False
    Config.LOAD_AE = False
    Config.LOAD_AE_OPTIMIZER = False
    Config.LOAD_PS = {'env1': True, 'env2': True, 'env3': True} 
    Config.LOAD_PS_OPTIMIZER = {'env1': True, 'env2': True, 'env3': True} 
    Config.TRAINERS = {'env1': 1, 'env2': 1, 'env3': 1}
    Config.PREDICTORS = {'env1': 1, 'env2': 1, 'env3': 1}
    Config.DEVICE = 'cpu'


# Run program.
if __name__=="__main__":
    mp.set_start_method('spawn')
    Server().main()
