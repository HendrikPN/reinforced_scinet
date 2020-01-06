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

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np

def plot_grad_flow(named_parameters, path):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    NOTE: This code is unused but might be helpful for analysis.

    Args:
        named_parameters (nn.Module.named_parameters): The named parameters of the model.
        path (str): The path where the plot should be saved.
    '''
    ave_grads = []
    median_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            median_grads.append(p.grad.abs().median()) 
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.bar(np.arange(len(max_grads)), median_grads, alpha=0.1, lw=1) 
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.hlines(0, 0, len(median_grads)+1, lw=2, color="k" ) 
    plt.xticks(range(0,len(median_grads), 1), layers, rotation="vertical") 
    plt.xlim(left=0, right=len(median_grads)) 
    plt.ylim(bottom = -0.001, top=0.01) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([lines.Line2D([0], [0], color="c", lw=4),
                lines.Line2D([0], [0], color="b", lw=4),
                lines.Line2D([0], [0], lw=4), 
                lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'median-gradient', 'zero-gradient']) 
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(path)
    plt.show()

def get_no_grad_params(optimizer, model):
    """
    Collects the named parameters of the model for which the optimizer obtained no
    gradients.

    NOTE: This code is unused but might be helpful for analysis.

    Args:
        optimizer (:class:`optim.Adam`): The adam optimizer for the model.
        model (:class:torch.nn.Module): The NN model.

    Returns:
        no_grad_params (:obj:`list` of :obj:`str`): The list of named parameters with no gradient.
    """
    no_grad_params = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                for n, q in model.named_parameters():
                    if q.size() == p.size():
                        if torch.all(torch.eq(q, p)):
                            no_grad_params.append(n)
    return no_grad_params

def check_models_eq(model1, model2):
    """
    Checks whether two models have the same weights.

    NOTE: This code is unused but might be helpful for analysis.

    Args:
        model1 (:class:torch.nn.Module): The first model.
        model2 (:class:torch.nn.Module): The second model.
    
    Returns:
        (bool): Whether or not the models have the same parameters.
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True
 