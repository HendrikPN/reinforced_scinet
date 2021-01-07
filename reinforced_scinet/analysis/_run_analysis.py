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

from analyzer import AnalyzerSubGridWorld 

PLOT_LATENT = False # plot the latent variable's behavior
PLOT_RESULTS = False # plot the performance of the RL agent
PLOT_RESULTS_LOSS = False # plot the performance of pretrainer
PLOT_FIGURE =  True # plot the figure from the whitepaper

ENV_ID = 'env2' # the environment id to be used (usually not relevant)

if __name__ == "__main__":
    analyzer = AnalyzerSubGridWorld(ENV_ID, load_model=True)
    if PLOT_LATENT:
        analyzer.plot_latent_space(limit=[-0.3, 0.3])
    if PLOT_RESULTS:
        analyzer.plot_results_figure(avg_mod=200, limit=[0., 400.])
    if PLOT_RESULTS_LOSS:
        analyzer.plot_loss_figure(avg_mod=100, limitDec=[0., 2.], limitAE=[0., 0.02])
    if PLOT_FIGURE:
        analyzer.plot_selection_figure(plot_neurons=[6, 1, 0], limit=[-0.3, 0.3])