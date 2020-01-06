#!/bin/bash

# create required directories
[ ! -d "saved_models" ] && { echo "Create network directory"; mkdir saved_models; }
[ ! -d "results_log" ] && { echo "Create result directory"; mkdir results_log; }

# check whether results exist and delete
[ -f "results_log/results.txt" ] && { echo "Remove previous results"; rm results_log/results.txt; }
[ -f "results_log/results_loss.txt" ] && { echo "Remove previous AE results"; rm results_log/results_loss.txt; }
[ -f "results_log/selection.txt" ] && { echo "Remove previous selection results"; rm results_log/selection.txt; }

# run main
python main.py
