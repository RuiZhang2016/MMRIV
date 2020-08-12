import os
os.system('rm job_*')
# os.system('rm ../our_methods/results/zoo/*/*_m.pdf')
out_str = os.popen('condor_submit run_experiments.sub').read()
ind = (out_str.split())[-1]
ind = ind[:-1]
os.system('condor_prio -p 20 '+ind)
