### This file is not intended to be used to run an experiment. It provides an example on how to train WGAN-GP on MNIST using this codebase. ###

#!/bin/bash

## Prepare virtual environment
source path/to/requirements.txt

## Settings
folder="mnist_results"
temp_folder="path/to/temp_folder"
dataset="mnist"
data="path/to/mnist_dataset" # Folder containing MNIST/raw/... used by the torchvision built-in MNIST dataset
test_data_path="path/to/mnisttest" # Folder containing MNIST test dataset images saved individually (used for FID evaluation with the pytorch_fid package) 
iterations=50000
checkpoints=25
generator='infogan'
critic='infogan'
bs=128
glr=0.001

## Train and evaluate
python train.py --folder=$folder --temp_folder=$temp_folder --dataset=$dataset --data=$data --test_data_path=$test_data_path --iterations=$iterations --checkpoints=$checkpoints --generator=$generator --critic=$critic --bs=$bs --glr=$glr
