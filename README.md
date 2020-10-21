# Maximum-Moment-Restriction
Python implementation of Maximum Moment Restriction for Instrumental Variable Regression (MMRIV).

# Steps to run the experiments
1. Install virtual environment: conda create -n MMR python=3.6
2. Activate environment: conda activate MMR
3. Install requirements: pip install -r requirements.txt
4. Download data from [the anonymous dropbox link](https://www.dropbox.com/sh/qxyh1ixpaceywf7/AAAk04Ls2VMDn0dqhn-4CSm-a?dl=0) to [the repo path] as [the repo path]/data/ without changing the data folder structure.
5. Run DeepGMM on low-dimensional (or MNIST or Mendelian) data: (1) cd [the repo path]/DeepGMM_scripts/ (2) python run_zoo(or mnist or mendelian)_experiments_deepgmm.py 
6. Run KernelIV on low-dimensional (or Mendelian) data: (1) cd [the repo path]/KernelIV/KIV/ (2) [use matlab run] main_zoo(or mendelian).m
7. Run other baselines on low-dimensional (or MNIST or Mendelian) data: (1) cd [the repo path]/other_baselines_scripts/ (2) python run_zoo(or mnist or mendelian)_experiments_more_baselines.py
8. Run MMR-IV (Nystr\"om) on low-dimensional (or MNIST or Mendelian) data: (1) cd [the repo path]/MMR_IVs/ (2) python rkhs_model_LMO_nystr_zoo.py (or python precomp_matrix_mnist.py; python rkhs_model_LMO_nystr_mnist.py or python precomp_matrix_mendelian.py; python rkhs_model_LMO_nystr_mendelian.py )
9. Run MMR-IV (NN) on low-dimensional (or MNIST or Mendelian) data: (1) cd [the repo path]/MMR_IVs/ (2) python nn_model_zoo.py (or python precomp_matrix_mnist.py; python nn_model_mnist.py or python precomp_matrix_mendelian.py; python nn_model_mendelian.py)
10. Run MMR-IV (Nystr\"om) on Vitamin D data: (1) cd [the repo path]/MMR_IVs/ (2) python rkhs_model_LMO_nystr_vitD.py

Note: precomp_matrix_*.py saves lots of time on redandunt computation by storing some intermediate results locally (tmp/, mendelian_precomp/ and mnist_precomp/). Running these files once is enough and files in tmp/ can be removed after.
