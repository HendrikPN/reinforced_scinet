# Analysis

The methods that were used to generate the figures from the `README` and paper.
The methods are tailored to the sub-grid world environment and cannot be used for anything else.
You need to make your own analysis tools if you intend to apply this code to another problem.

## analyzer

The class that comprises all the methods to plot the results can be found in `analyzer.py`.
To generate the plots you may adjust the parameters in `_run_analysis.py` and execute it. 

+ `plot_latent_space`: A figure that displays the responses of latent neurons.
+ `plot_results_figure`: A figure that displays the reinforcement learning results.
+ `plot_loss_figure`: A figure that displays the losses during representation learning.
+ `plot_selection_figure`: The results figure from the Sec. 6 of the paper.