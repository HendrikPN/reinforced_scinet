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

from collections import deque
import torch

from config import Config

class ShortTermMemory():
    def __init__(self, glow):
        """
        Short term memory contains memorized experience as observation, action and target h-value. All as torch.Tensors.
        It also contains the history of observations, actions, h-values and rewards.
        Memories are added from the history through backpropagation of the rewad via a glow mechanism. 

        Args:
            glow (float): Glow value which determines how much a reward is backpropagated.
        """
        self.glow = glow
        #:class:`collections.deque` of :obj:`tuple`: Short term memory. Short term experiences are stored here.
        self.memory = deque()
        #:class:`collections.deque` of :obj:`tuple`: Short term memory. Fake short term experiences are stored here.
        self.memory_fake = deque()
        #:class:`list` of :obj:`tuple`: History of observations, actions, h-values and rewards.
        self.history = list()
        #:class:`collections.deque` of :obj:`bool`: Queue of flags associated with experience to be remembered.
        self.flags = deque()

    def create_experience(self):
        """
        Creates experience from history by backpropagating rewards and adds experience to the memory.
        Does not reset history or memory before or after.
        """
        current_reward = 0
        for hist in reversed(self.history):
            observation, action, hval, reward = hist
            is_saved = self.flags.pop()
            current_reward *= (1-self.glow)
            if reward:
                current_reward += reward
            if abs(current_reward) > Config.REWARD_CLIP_VALUE and is_saved:
                if Config.TRAIN_MODE == 'policy':
                    target = (1 - Config.GAMMA) * hval + current_reward
                elif Config.TRAIN_MODE == 'selection':
                    target = torch.DoubleTensor([current_reward * Config.REWARD_RESCALE])
                self.memory.appendleft((observation, action, target.view(1,1)))

    def add_event(self, event, flag=True):
        """
        Adds an event to the history.

        Args:
            event (tuple): Tuple consisting of observation, action, h-value and reward.
            flag (bool): Whether or not this event will be used for training. Default is True.
        """
        self.history.append(event)
        self.flags.append(flag)

    def add_experience_fake(self, observation, action, reward):
        """
        Creates fake experience by hand. Appends an event to deque.

        Args:
            observation (torch.Tensor): State of environment.
            action (torch.Tensor): Encoded action that is being performed.
            reward (float): Discounted reward for this experience.
        """
        target = torch.DoubleTensor([0.])
        self.memory_fake.append((observation, action, target.view(1,1)))

    def reset_memory(self):
        """
        Resets memory.
        """
        self.memory = deque()
    
    def reset_history(self):
        """
        Resets history.
        """
        self.history = list()
        
    def reset(self):
        """
        Resets both history and memory.
        """
        self.memory = deque()
        self.memory_fake = deque()
        self.history = list()
        self.flags = deque()
