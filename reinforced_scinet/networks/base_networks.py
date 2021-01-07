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
import torch.nn as nn
from torch.autograd import Variable

class DenseEncoder(nn.Module):
    """
    A dense neural network which has a low-dimensional output 
    representing latent variables of an abstract representation with adjustable
    dimensionality through :class:`Selection` neurons.

    Args:
        dim_input (int): The size of the input.
        dim_dense (:obj:`list` of :obj:`int`): Number of neurons per layer.
        dim_latent (int): The size of the output.
    """
    def __init__(self, dim_input, dim_dense, dim_latent):
        super(DenseEncoder, self).__init__()
        # Building input layer.
        self.input = nn.Sequential(
                            nn.Linear(dim_input, dim_dense[0]),
                            nn.ELU()
                        )
            
        # Building hidden, dense layers.
        self.encoding = nn.ModuleList()
        for l in range(len(dim_dense)-1):
            modules = []
            modules.append(nn.Linear(dim_dense[l], dim_dense[l+1]))
            if l != len(dim_dense)-2:
                modules.append(nn.ELU())
            self.encoding.append(
                nn.Sequential(*modules)
            )

        # Building dense output/abstraction layer.
        self.abstraction = nn.Linear(dim_dense[-1], dim_latent)

        # Building selection neurons.
        self.selection = Selection(dim_latent)

    def forward(self, x):
        """
        The forward pass through the encoder without selection neurons. 

        Args:
            x (torch.Tensor): The input array.
        
        Returns:
            out (torch.Tensor): The latent representation of the encoder.
        """
        # (i) Input
        x = self.input(x)
        # (ii) Dense
        for dense in self.encoding:
            x = dense(x)
        # (iii) Abstraction
        out = self.abstraction(x)
        return out

    def forward_sel(self, x, rand):
        """
        The forward pass through the encoder with selection neurons.

        Args:
            x (torch.Tensor): The input array.
            rand (torch.Tensor): Random samples from standard normal distribution of size (batch_size, size_latent).
        
        Returns:
            out (torch.Tensor): The latent representation of the encoder after selection.
        """
        if x.size(0) == 1:
            raise ValueError(f'Input batch size is incorrect: {x.size()}. Requires batch size > 1.')
        # (i) Input
        x = self.input(x)
        # (ii) Dense
        for dense in self.encoding:
            x = dense(x)
        # (iii) Abstraction
        x = self.abstraction(x)
        # (iv) Pass through selection neurons.
        out = self.selection(x, rand)
        return out

class DenseDecoder(nn.Module):
    """
    A dense neural network which has a low-dimensional input 
    representing latent variables of an abstract representation and producing
    a high-dimensional output.

    Args:
        dim_latent (int): The size of the output.
        dim_dense (:obj:`list` of :obj:`int`): Number of neurons per layer.
        dim_output (int): The size of the input.
    """
    def __init__(self, dim_latent, dim_dense, dim_output):
        super(DenseDecoder, self).__init__()
        # Building input layer.
        self.unabstraction = nn.Sequential(
                             nn.Linear(dim_latent, dim_dense[0]),
                             nn.ELU()
                            )
            
        # Building hidden, dense decoding layers.
        dim_hidden = dim_dense + [dim_output]
        self.decoding = nn.ModuleList()
        for l in range(len(dim_hidden)-1):
            modules = []
            modules.append(nn.Linear(dim_hidden[l], dim_hidden[l+1]))
            if l != len(dim_hidden)-2:
                modules.append(nn.ELU())
            else:
                modules.append(nn.Sigmoid())
            self.decoding.append(
                nn.Sequential(*modules)
            )

    def forward(self, x):
        """
        The forward pass through the encoder. 

        Args:
            x (torch.Tensor): The input array.
        
        Returns:
            out (torch.Tensor): The latent representation of the encoder.
        """
        # (i) Input
        x = self.unabstraction(x)
        # (ii) Dense
        for dense in self.decoding:
            x = dense(x)

        return x
    
class DenseAutoencoder(nn.Module):
    """
    A dense autoencoder consisting of a :class:`DenseEncoder` and :class:`DenseDecoder`.
    
    Args:
        dim_input (:obj:`list` of :obj:`int`): The x,y size of the input image.
        dim_dense_enc (:obj:`list` of :obj:`int`): Sizes of hidden, dense layers for Encoder.
        dim_dense_dec (:obj:`list` of :obj:`int`): Sizes of hidden, dense layers for Decoder
        dim_latent (int): The size of the latent representation space.
    """
    def __init__(self, dim_input, dim_dense_enc, dim_dense_dec, dim_latent):
        super(DenseAutoencoder, self).__init__()

        #:class:`DenseEncoder`: Dense encoder.
        self.encoder = DenseEncoder(dim_input, dim_dense_enc, dim_latent)
        #:class:`DenseDecoder`: Dense decoder.
        self.decoder = DenseDecoder(dim_latent, dim_dense_dec, dim_input)
    
    def forward(self, x):
        """
        The forward pass through the autoencoder.

        NOTE: This method is unused since we just call the individual parts, i.e. `encoder` and `decoder`.

        Args:
            x (torch.Tensor): The input array of shape (batch_size, #channel, x-size, y-size).

        Returns:
            dec_out (torch.Tensor): The decoder output.
            latent (torch.Tensor): The latent representation of the encoder.
        """

        latent = self.encoder(x)
        dec_out = self.decoder(latent)

        return dec_out, latent



class Selection(nn.Module):
    """
    Selection neurons to sample from a latent representation for a decoder agent.
    An abstract representation :math:`l_i` is disturbed by a value :math:`r_i` sampled from a normal 
    standard distribution which is scaled by the selection neuron :math:`s_i`.

    ..math::
        n_i \sim l_i + \sigma_{l_i} \times \exp(s_i) \times r_i

    where :math:`\sigma_{l_i}` is the standard deviation over the batch. 
    If the selection neuron has a low (i.e. negative) value, the latent variable is passed to the agent. 
    If the selection neuron has a high value (i.e. close to zero), the latent variable is rendered useless to the agent.

    Args:
        num_selectors (int): Number of selection neurons, i.e. latent variables.

        **kwargs:
            init_selectors (float): Initial value for selection neurons. Default: -10.
    """

    def __init__(self, num_selectors, init_selectors=-10.):
        super(Selection, self).__init__()
        select = torch.Tensor([init_selectors for _ in range(num_selectors)])
        # torch.nn.parameter.Parameter: The selection neurons.
        self.selectors = nn.Parameter(select)

    def forward(self, x, rand, std_dev=None):
        """
        The forward pass for the selection neurons.

        Args:
            x (torch.Tensor): The input array of shape (batch_size, size_latent).
            rand (torch.Tensor): Random samples from standard normal distribution of size (batch_size, size_latent).

            **kwargs:
                std_dev (:class:`torch.Tensor` or :class:`NoneType`): The standard deviation calculated throughout 
                                                                      episodes. Needs to be specified for prediction. 
                                                                      Default: None.
        
        Returns:
            sample (torch.Tensor): Sample from a distribution around latent variables.
        """
        selectors = self.selectors.expand_as(x)
        if std_dev is None:
            std = x.std(dim=0).expand_as(x)
        else:
            std = std_dev
        sample = x + std * torch.exp(selectors) * rand
        return sample

class DeepPS(nn.Module):
    """
    A dense neural network for reinforcement learning with Projective Simulation 
    on a low-dimensional input space which represents an abstract state 
    space representation and a discrete action space.
    The agent can preselect part of the latent space to reduce the information of the input.
    The agent network may also be used to predict discounted rewards.

    Args:
        dim_latent (int): The size of the input abstract representation.
        dim_action (int): The size of the action space.
        dim_dense (:obj:`list` of :obj:`int`, optional): Sizes of hidden, dense layers. Default: [100, 64].
        init_kaiman (bool, optional): If this is set to True, network is initialized with weights sampled 
                                      from a uniform distribution according to the kaiming_uniform method.
                                      Default: True.
        is_predictor (bool, optional): If this is set to True, this network is used as a predictor and not for RL.
                                       Default: False.
    """
    def __init__(self, dim_latent, dim_action, dim_dense=[100, 64], init_kaiman=True, is_predictor=False):
        super(DeepPS, self).__init__()
        # Building selection neurons.
        self.selection = Selection(dim_latent)

        # Building input layer.
        self.visible = nn.Sequential(
                            nn.Linear(dim_latent+dim_action, dim_dense[0]),
                            nn.ReLU()
                        )
        self.visible.apply(self._init_weights) if init_kaiman else None

        # Building hidden, dense layers.
        self.dense = nn.ModuleList()
        for l in range(len(dim_dense)-1):
            self.dense.append(
                nn.Sequential(
                    nn.Linear(dim_dense[l], dim_dense[l+1]),
                    nn.ReLU()
                )
            )
            self.dense[l].apply(self._init_weights) if init_kaiman else None

        # Building (constant) output layer.
        self.output = nn.Linear(dim_dense[-1], 1)
        if is_predictor:
            nn.init.constant_(self.output.weight, 1.)
            nn.init.constant_(self.output.bias, 0.)
        else:
            nn.init.constant_(self.output.weight, 1.)
            nn.init.constant_(self.output.bias, 1.)
            self.output.weight.requires_grad = False
            self.output.bias.requires_grad = False

    def _init_weights(self, layer):
        """
        Initializes weights with kaiming_uniform method for ReLu from PyTorch.

        Args:
            layer (nn.Linear):  Layer to be initialized.
        """
        if type(layer) == nn.Linear:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward_predict(self, x, action, rand, std_dev):
        """
        The forward pass through the DeepPS for prediction. Passes a single input (batch size: 1) 
        through
        (i) selection neurons, which preselect part of the latent space, to
        (ii) a concatenation layer with action input to
        (iii) dense layers to
        (iv) an output layer.

            Args:
                x (torch.Tensor): The input array of shape (batch_size, dim_latent).
                action (torch.Tensor): The input action array of shape (batch_size, dim_action).
                rand (torch.Tensor): Random samples from standard normal distribution of size (batch_size, size_latent).
                std_dev (torch.Tensor): The standard deviation over multiple interactions.
        """
        # (i) Pass through selection neurons.
        action = action.expand((x.size(0), -1))
        std_dev = std_dev.expand_as(x)
        x = self.selection(x, rand, std_dev=std_dev)
        # (ii) Concatenate action and input.
        x = torch.cat((x, action), dim=1)
        # (iii) Dense layers
        x = self.visible(x)
        for l in range(len(self.dense)):
            x = self.dense[l](x)
        # (iv) Constant output layer which just sums inputs.
        x = self.output(x)
        return x
    
    def forward_no_selection(self, x, action):
        """
        The forward pass through the DeepPS w/o selection neurons.

        Args:
            x (torch.Tensor): The input array.
            action (torch.Tensor): The input action array of shape (batch_size, dim_action).
        """
        # (i) Concatenate action and input.
        action = action.expand((x.size(0), -1))
        x = torch.cat((x, action), dim=1)
        # (ii) Dense layers
        x = self.visible(x)
        for l in range(len(self.dense)):
            x = self.dense[l](x)
        # (iii) Constant output layer which just sums inputs.
        x = self.output(x)
        return x 

    def forward(self, x, action, rand):
        """
        The forward pass through the DeepPS. Passes input batch (size > 1) through 
        (i) selection neurons, which preselect part of the latent space, to
        (ii) a concatenation layer with action input to
        (iii) dense layers to
        (iv) an output layer.

            Args:
                x (torch.Tensor): The input array of shape (batch_size, dim_latent). Assumes batch_size > 1.
                action (torch.Tensor): The input action array of shape (batch_size, dim_action).
                rand (torch.Tensor): Random samples from standard normal distribution of size (batch_size, size_latent). 
        """
        if x.size(0) == 1:
            raise ValueError(f'Input batch size is incorrect: {x.size()}. Requires batch size > 1.')
        # (i) Pass through selection neurons.
        x = self.selection(x, rand)
        # (ii) Concatenate action and input.
        x = torch.cat((x, action), dim=1)
        # (iii) Dense layers
        x = self.visible(x)
        for l in range(len(self.dense)):
            x = self.dense[l](x)
        # (iv) Constant output layer which just sums inputs.
        x = self.output(x)
        return x
