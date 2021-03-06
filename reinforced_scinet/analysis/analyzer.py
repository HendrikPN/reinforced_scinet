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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os, sys
import gym
import gym_subgridworld
import cv2

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils.data_generator import TrainDataGenerator
from config import Config
sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from base_networks import DenseAutoencoder

class AnalyzerSubGridWorld():
    def __init__(self, env_id, load_model=False):
        """
        The analyzer is used to generate plots for the sub-grid world environment which...

            ...display the performance of the RL agent.
            ...display the performance of the prediction.
            ...displays behavior of latent variables w.r.t. a test set.

        Args:
            env_id (str): The environment Id specifying which environment to use given a Config. 
            load_model (bool, optional): Whether or not to load the autoencoder as specified in Config. Default: False.
        """
        if not Config.ENV_NAME == 'subgridworld-v0':
            raise NotImplementedError("This analyzer is tailored to the sub-grid world environment. "+
                                      f"You used it for '{Config.ENV_NAME}'. "+
                                      "You need to create your own Analyzer class.")
        self.env_id = env_id
        #str: The name of the environment.
        self.env_name = Config.ENV_NAME
        self.env = gym.make(Config.ENV_NAME, **Config.ENV_PARAMS[self.env_id])
        if load_model:
            #:class:``: DenseEncoder model
            self.encoder = DenseAutoencoder(*Config.AE_PARAMS)
            loaded_dict = torch.load("../"+Config.LOAD_AE_FILE, map_location=torch.device('cpu'))
            self.encoder.load_state_dict(loaded_dict)
            self.encoder = self.encoder.encoder

    def plot_latent_space(self, limit=[-0.5, 0.5]):
        """
        This method plots the responses of all latent neurons to combinations of x,y,z positions.

        Args:
            limit (list, optional): The minimal and maximal value that are being plotted. Default: [-0.5, 0.5]
        """
        latent_neurons = range(Config.LATENT_SIZE)
        axes_label = ['x', 'y', 'z'] # all axes
        show_axes = [[0,1], [1,2], [0,2]] # combinations of axes to be plotted
        ignore_axes = [2,0,1] # axes which are to be ignored per plot
        
        fig = plt.figure("Latent variables", figsize=(30,30))
        index = 0

        # create plots
        for neuron in latent_neurons:
            for i, axes in enumerate(show_axes):
                index += 1
                # get points which are to be analyzed
                x,y = self._get_points_subgridworld(axes)
                # get responses for points
                z = self._get_data_subgridworld(x, y, axes, ignore_axes[i], neuron)

                ax = fig.add_subplot(Config.LATENT_SIZE, 3, index, projection='3d')
                surf = ax.plot_surface(x, y, z, cmap=cm.inferno,
                                    linewidth=0, antialiased=False, vmin=limit[0], vmax=limit[1])

                fs = 22 * 5/Config.LATENT_SIZE
                ax.set_xlabel(f"{axes_label[axes[0]]}-axis", fontsize=fs, labelpad=15)
                ax.set_ylabel(f"{axes_label[axes[1]]}-axis", fontsize=fs, labelpad=15)
                ax.set_zlabel(f"Latent variable #{neuron}", fontsize=fs, labelpad=15)
                ax.set_zlim(limit[0], limit[1])

        plt.tight_layout(pad=1.5)
        plt.savefig('results/latent_space.png')
        plt.show()

    def plot_selection_figure(self, plot_neurons=[6, 1, 0], limit=[-0.3, 0.3]):
        """
        Plot for the figure as it appears in the whitepaper. 

        Args:
            plot_neurons (list): The indices of neurons to be plotted.
            limit (list): The minimal and maximal value that are being plotted.
        """
        latent_neurons = plot_neurons # the relevant neurons
        axes_label = ['x', 'y', 'z'] # all axes
        show_axes = [[0,1], [1,2], [0,2]] # combinations of axes to be plotted
        ignore_axes = [2,0,1] # axes which are to be ignored per plot

        # create figure
        fig = plt.figure("Latent variables", figsize=(25,15))
        index = 0 # keep track of index in figure
        plot_label = 0 # keep track of index of current plot

        # create plots of latent neuron responses
        for i, axes in enumerate(show_axes):
            for neuron in latent_neurons:
                plot_label+=1
                # skip some plots to not be included in figure
                if plot_label in [5,6,7]:
                    continue
                index += 1

                # get points which are to be analyzed
                x,y = self._get_points_subgridworld(axes)
                # get responses for points
                z = self._get_data_subgridworld(x, y, axes, ignore_axes[i], neuron)
                # add plot to figure
                ax = fig.add_subplot(3, 3, index, projection='3d')
                surf = ax.plot_surface(x, y, z, cmap=cm.inferno,
                                    linewidth=0, antialiased=False, vmin=limit[0], vmax=limit[1])

                # style changes: labels, ticks, limits,...
                ax.set_xlabel(f"{axes_label[axes[0]]}-axis", fontsize=22, labelpad=15)
                ax.set_ylabel(f"{axes_label[axes[1]]}-axis", fontsize=22, labelpad=15)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.zaxis.set_major_locator(plt.MaxNLocator(5))
                ax.view_init(50, 40)

                # additional labels to figure on side
                if index in [1,2,3]:
                    ax.set_title(f'Latent neuron {plot_label}',fontsize=30, pad=30)
                if index in [1]:
                    ax.text2D(0.01, 0.8, 'Latent neuron\nactivation', fontsize=25, transform=plt.gcf().transFigure)
                if index in [4]:
                    ax.text2D(0.01, 0.5, 'Latent neuron\nactivation', fontsize=25, transform=plt.gcf().transFigure)
                ax.set_zlim(limit[0], limit[1])

        # create plots of selection noise
        for neuron in latent_neurons:
            index += 1
            self._plot_selection(fig, neuron, index)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig('results/selection_figure.pdf')
        plt.show()

    def plot_results_figure(self, avg_mod=200, limit=[0., 400.]):
        """
        Plots reinforcement learning results for the various agents in the sub-grid world environment.
        Plots the results from one results file.

        Args:
            avg_mod (int, optional): The number of elements to be averaged over. Default: 200.
            limit (list, optional): The y-limit for the plot. Default: [0., 400.].
        """
        fig = plt.figure("Reinforcement learning results", figsize=(25,6))

        # style changes
        plt.rc('font', size=18)
        params = {'axes.linewidth': 1.}
        plt.rcParams.update(params)

        for index, env_id in enumerate(Config.ENV_IDS):
            self._plot_results(fig, index+1, env_id, avg_mod, limit)
        plt.tight_layout()
        plt.savefig('results/results_rl.png')
        plt.show()

    def plot_loss_figure(self, avg_mod=100, limitDec=[0.,1.], limitAE=[0.,0.01]):
        """
        Plots prediction loss results for the various decoders (and autoencoders) in the sub-grid world environment.
        Plots the results from one loss results file.

        Args:
            avg_mod (int, optional): The number of elements to be averaged over. Default: 100.
            limitDec (list, optional): The y-limit for the decoders. Default: [0., 1.].
            limitAE (list, optional): The y-limit for the autoencoder. Default: [0., 0.01].
        """
        fig = plt.figure("Prediction loss", figsize=(25,6))

        # style changes
        plt.rc('font', size=20)
        params = {'axes.linewidth': 1.}
        plt.rcParams.update(params)
        self._plot_results_loss(fig, 1, None, 'Autoencoder', avg_mod, limitDec, limitAE)
        for index, env_id in enumerate(Config.ENV_IDS):
            self._plot_results_loss(fig, index+2, env_id, 'Policy', avg_mod, limitDec, limitAE)
        plt.tight_layout()
        plt.savefig('results/results_loss.png')
        plt.show()

    # ----------------- helper methods ---------------------------------------------------------------------

    def _get_points_subgridworld(self, axes):
        """
        Generate data points in the sub-grid world environment for a 3D plot.

        Args:
            axes (list): List of axes that are adressed in the sub grid-world environment.

        Returns:
            x,y (np.meshgrid): Coordinate array for 3D plot.
        """
        x_max, y_max = (self.env._grid_size[i] for i in axes)
        x = [i for i in range(x_max)]
        y = [i for i in range(y_max)]
        x,y = np.meshgrid(x,y)

        return x,y

    def _get_data_subgridworld(self, x, y, axis_show, axis_ignore, latent_neuron):
        """
        Get the latent responses of one neuron to two variable positions in the grid and the third fixed to 6.

        Args:
            x (np.ndarray): Array of position values.
            y (np.ndarray):
            axis_show (list): List of axes which are adressed.
            axis_ignore (int): Axis which is set to 6.
            latent_neuron (int): The label of the latent neuron that is analyzed.

        Returns:
            data (np.ndarray): Latent responses for specified latent neuron.
        """
        data = torch.zeros(x.shape)
        for index_x, index_y in np.ndindex(x.shape):
            x_,y_ = x[index_x,index_y], y[index_x,index_y]
            nn_input = torch.zeros((1, sum(self.env._grid_size)))
            pos = [0,0,0]
            pos[axis_show[0]] = x_
            pos[axis_show[1]] = y_
            pos[axis_ignore] = 6
            for index, i in enumerate(pos):
                nn_input[0, index*12 + i] = 1.
            with torch.no_grad():
                data[index_x, index_y] = self.encoder(nn_input)[0][latent_neuron]
            
        return data.numpy()

    def _plot_selection(self, fig, selection_neuron, index, limit=[-10,10]):
        """
        Plots the selection noise in the figure for the whitepaper.

        Args:
            fig (plt.figure): The figure in which the plot is to be displayed.
            selection_neuron (int): The index of the selection neuron to be plotted.
            index (int): The index in the figure.
            limit (list, optional): The y-limit for the figure. Default: [-10, 10].
        """
        # get data points from file
        downsample = 10 # do not take all points from file
        with open('../results_log/selection.txt', 'r') as data:
            results_full = np.array([line.split() for line in data])
            results_id = np.delete(results_full, [i for i in range(1, Config.LATENT_SIZE+1)], 1).flatten()
            results_full = np.delete(results_full, [0], 1)#.flatten()
        data.close()

        # style changes
        plt.rc('font', size=18)
        params = {'legend.fontsize': 25, 'legend.markerscale': 1.5, 'legend.numpoints': 100, 'axes.linewidth': 1.}
        plt.rcParams.update(params)

        # create plot
        ax = fig.add_subplot(3,3,index)
        for env_id in Config.ENV_IDS + ['enc']:
            # get lines for specific env ID
            positions = np.where(results_id == env_id+',')
            results = results_full[positions]
            size_results = len(results)
            results = np.array([results[i] for i in range(0, size_results, downsample)])

            # get values from lines
            plot_data = results[:,selection_neuron]
            if selection_neuron == 0:
                #remove bracket at beginning and comma at end
                plot_data = [point[1:-1] for point in plot_data]
            else:
                #remove bracket or comma at end
                plot_data = [point[:-1] for point in plot_data]
            plot_data = list(map(float, plot_data))
            # get associated x-values
            x_axis = [i*Config.SELECTION_SAVE_FREQUENCY for i in range(0, size_results, downsample)]
            # plot values
            ax.plot(x_axis,plot_data, marker='o',markersize=3.)
            # style changes
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.tick_params(axis='both', which='major', labelsize=18)
            # remove y label for some
            if index > 7:
                plt.setp(ax.get_yticklabels(), visible=False)

        # add additional labels and legend on sides
        if index in [7]:
            ax.text(0.01, 0.15, 'Selection\nneuron\nactivation', fontsize=25, transform=plt.gcf().transFigure)
        if index in [9]:
            legend = ax.legend(['Decoder 1','Decoder 2', 'Decoder 3', 'Encoder'], 
                                loc="upper right", bbox_to_anchor=(1.09, 0.58))
            legend.get_frame().set_linewidth(2.0)
        plt.xlabel("Training Episodes", fontsize=25, labelpad=15)
        ax.grid(axis='y')
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        plt.ylim(limit[0],limit[1])


    def _plot_results(self, fig, label, env_id, avg_mod, limit):
        """
        Plots the results for one agent with a specified environment id.
        The plot is compressed by averaging over `avg_mod` number of steps.

        Args:
            fig (plt.figure): The figure in which the plot is to be displayed.
            label (int): The index of the plot in figure.
            env_id (str): The id of the environment for which the results are to be displayed.
            avg_mod (int): The number of elements to be averaged over.
            limit (list): The y-limit for the plot.
        """
        ax = fig.add_subplot(1,3,label)
        # get results
        with open('../'+Config.RESULTS_FILE, 'r') as data:
            results = np.array([line.split() for line in data])
            indices = np.where(results==env_id+',')[0]
            results = results[indices]
            results = np.delete(results, [0,1,2,3], 1)
            results = results.astype(int)
        data.close()

        # modify results by averaging over a certain length
        avg_results = []
        avg = 0
        for index, value in enumerate(results):
            avg += value
            if index%avg_mod == 0:
                avg = float(avg/avg_mod)
                avg_results.append(avg)
                avg = 0
        if not index%avg_mod == 0:
            avg = avg/(index%avg_mod)
            avg_results.append(avg)
        # get associated x-axis
        x_axis = (np.arange(len(avg_results))+avg_mod)*avg_mod
        # plot values
        ax.plot(x_axis, avg_results)

        # style changes
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel("trial")
        if label == 1:
            # add label on side
            plt.ylabel("number of steps\nuntil reward was found", labelpad=15)
        else:
            # remove y label
            plt.setp(ax.get_yticklabels(), visible=False)
        plt.title(f"Agent #{label}")
        plt.grid(True)
        plt.ylim(limit[0],limit[1])

    def _plot_results_loss(self, fig, label, env_id, loss_type, avg_mod, limitDec, limitAE):
        """
        Plots the prediction loss for decoders and autoencoder.

        Args:
            fig (plt.figure): The figure in which the plot is to be displayed.
            label (int): The index of the plot in figure.
            env_id (str): The id of the environment for which the results are to be displayed.
            loss_type (str): The type of loss coming either from 'Autoencoder' or 'Policy'
            avg_mod (int): The number of elements to be averaged over.
            limitDec (list): The y-limit for the decoder plot.
            limitAE (list): The y-limit for the autoencoder plot.

        """
        ax = fig.add_subplot(1,4,label)
        downsample = 100 # do not take all points from file

        # get results
        with open('../results_log/results_loss.txt', 'r') as data:
            results_ae = np.array([line.split() for line in data])#.flatten().astype(float)
            results_id = np.delete(results_ae, [0,1,2,4,5], 1).flatten()
            results_type = np.delete(results_ae, [0,1,3,4,5], 1).flatten()
            results_ae = np.delete(results_ae, [0,1,2,3,4], 1).flatten()
            results_ae = results_ae.astype(float)
        data.close()
        # get results associated with id or type
        positions_ae = np.where(results_type == 'Autoencoder,')
        if loss_type == 'Autoencoder':
            positions = positions_ae
        else:
            positions = np.where(results_id == env_id+',')
            positions = np.setdiff1d(positions, positions_ae)
        results_ae = results_ae[positions]
        
        # modify results by averaging over a certain length
        avg_results = []
        avg = 0
        for index, value in enumerate(results_ae):
            avg += value
            if index%avg_mod == 0:
                avg = float(avg/avg_mod)
                avg_results.append(avg)
                avg = 0
        if not index%avg_mod == 0:
            avg = avg/(index%avg_mod)
            avg_results.append(avg)
        # get associated x-axis
        x_axis = (np.arange(len(avg_results))+avg_mod)*avg_mod
        
        # plot results
        ax.plot(x_axis, avg_results)

        # style changes
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel("Training Episodes")
        if label == 2:
            # add label on side
            plt.ylabel("smooth L1 loss", labelpad=15)
        elif label > 1:
            # remove y label
            plt.setp(ax.get_yticklabels(), visible=False)
        plt.title(f"Decoder #{label-1}")
        plt.ylim(limitDec[0], limitDec[1])
        if label == 1:
            plt.title(f"Autoencoder")
            plt.ylim(limitAE[0], limitAE[1])
            plt.ylabel("BCE loss", labelpad=15)
        plt.grid(True)
