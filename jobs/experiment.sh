#!/bin/bash

source /lustre/home/ruizhang/miniconda3/etc/profile.d/conda.sh
conda activate py36
python /lustre/home/ruizhang/DeepGMM/our_methods/nn_model.py "$@"
