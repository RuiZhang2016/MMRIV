import torch, add_path
import numpy as np
import os
from scenarios.abstract_scenario import AbstractScenario
from methods.toy_model_selection_method import ToyModelSelectionMethod
import sys
from tabulate import tabulate
from MMR_IVs.util import ROOT_PATH, load_data
import random
random.seed(527)

def run_experiment(scenario_name, repid, datasize):
    # set random seed
    seed = 527
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_reps = 10

    print("\nLoading " + scenario_name + "...")
    train, dev, test = load_data(ROOT_PATH+'/data/zoo/'+scenario_name+'_{}.npz'.format(datasize),Torch=True)

    folder = ROOT_PATH+"/results/zoo/" + scenario_name + "/"
    os.makedirs(folder, exist_ok=True)
    for rep in range(repid,repid+1):
        method = ToyModelSelectionMethod(enable_cuda=torch.cuda.is_available())
        time = method.fit(train.x, train.z, train.y, dev.x, dev.z, dev.y,
                   g_dev=dev.g, verbose=True)
        np.save(folder+"deepgmm_%d_%d_time.npy" %(rep,train.x.shape[0]),time)

        g_pred_test =  method.predict(test.x)
        mse =  float(((g_pred_test - test.g) ** 2).mean())

        print("--------------- "+str(rep))
        print("MSE on test:", mse)
        print("")
        print("saving results...")
        file_name = "deepgmm_%d_%d.npz" %(rep,train.x.shape[0])
        save_path = os.path.join(folder, file_name)
        np.savez(save_path, x=test.w, y=test.y, g_true=test.g,
                 g_hat=g_pred_test.detach())


if __name__ == "__main__":
    scenarios = np.array(["abs", "linear", "sin", "step"])

    for datasize in [200,2000]:
        for sid in range(4):
            for repid in range(10):
                run_experiment(scenarios[sid],repid,datasize)

        means = []
        times = []
        for scenario_name in scenarios:
            folder = ROOT_PATH+"/results/zoo/" + scenario_name + "/"
            means2 = []
            times2 = []
            for rep in range(10):
                save_path = folder+"deepgmm_%d_%d.npz" % (rep,datasize)
                res = np.load(save_path)
                means2 +=  [np.mean((res['g_hat']-res['g_true'])**2)]
                time_path = folder+"deepgmm_%d_%d_time.npy" % (rep,datasize)
                res = np.load(time_path)
                times2 += [res]
            means += [means2]
            times += [times2]
        print(times)
        print('times',np.mean(times,axis=1),np.std(times,axis=1))
        mean = np.mean(means,axis=1)
        std = np.std(means,axis=1)
        rows = ["{:.3f} $pm$ {:.3f}".format(mean[i],std[i]) for i in range(len(mean))]
        print(tabulate(np.vstack((scenarios,rows)), headers='firstrow',tablefmt='latex'))


