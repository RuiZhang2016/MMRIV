#!/bin/bash

source /lustre/home/ruizhang/miniconda3/etc/profile.d/conda.sh
conda activate py36
# python /lustre/home/ruizhang/MMR/our_methods/rkhs_model_leave_M_out_nystr2.py "$@" 200
# python /lustre/home/ruizhang/MMR/run_zoo_experiments_baselines.py "$@" 2000
# python /lustre/home/ruizhang/MMR/run_zoo_experiments_ours.py "$@" 2000
# python /lustre/home/ruizhang/MMR/our_methods/nn_model.py "$@" 10000
# python /lustre/home/ruizhang/MMR/run_mnist_experiments_baselines.py "$@"
# python /lustre/home/ruizhang/MMR/run_mnist_experiments_ours.py "$@"
# python /lustre/home/ruizhang/MMR/our_methods/precomp_matrix_mnist.py "$@"
# python /lustre/home/ruizhang/MMR/our_methods/rkhs_model_leave_M_out_nystr_mnist3.py "$@"
python /lustre/home/ruizhang/MMR/run_mendelian_experiments_baselines.py "$@"
# python /lustre/home/ruizhang/MMR/our_methods/nn_model_mendelian.py "$@"
# python /lustre/home/ruizhang/MMR/our_methods/rkhs_model_leave_M_out_nystr_mendelian.py "$@"
# python /lustre/home/ruizhang/MMR/run_mendelian_experiments_ours.py "$@"
