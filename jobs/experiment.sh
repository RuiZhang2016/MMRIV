#!/bin/bash

source /lustre/home/ruizhang/miniconda3/etc/profile.d/conda.sh
conda activate py36
python /lustre/home/ruizhang/MMR/our_methods/rkhs_model_leave_M_out_nystr2.py "$@"
