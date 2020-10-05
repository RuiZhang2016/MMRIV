import torch
import numpy as np
import os
from scenarios.abstract_scenario import AbstractScenario
from methods.toy_model_selection_method import ToyModelSelectionMethod
import sys
from scipy import io
from tabulate import tabulate
from our_methods.util import ROOT_PATH, load_data
import random
random.seed(527)

def run_experiment(scenario_name, repid):
    # set random seed
    seed = 527
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\nLoading " + scenario_name + "...")
    matname = ROOT_PATH + "/data/mendelian/"+scenario_name+ '.mat'
    if not os.path.exists(matname):
        # set load_data(Torch=False)
        train, dev, test = load_data(ROOT_PATH + "/data/mendelian/" + scenario_name+'.npz',Torch=False)
        io.savemat(matname, mdict={'X_train': train.x, 'Y_train': train.y, 'Z_train': train.z,'X_dev': dev.x, 'Y_dev': dev.y, 'Z_dev': dev.z,'X_test': test.x, 'g_test': test.g})
    
    train, dev, test = load_data(ROOT_PATH + "/data/mendelian/" + scenario_name+'.npz',Torch=True)
    folder = ROOT_PATH+"/results/mendelian/"+scenario_name+"/"
    os.makedirs(folder, exist_ok=True)
    for rep in range(repid,repid+1):
        method = ToyModelSelectionMethod(f_input=train.z.shape[1], enable_cuda=torch.cuda.is_available())
        time = method.fit(train.x, train.z, train.y, dev.x, dev.z, dev.y,
                   g_dev=dev.g, verbose=True)
        np.save(folder+"deepgmm_%d_time.npy" %(rep),time)
        g_pred_test =  method.predict(test.x)
        mse =  float(((g_pred_test - test.g) ** 2).mean())

        print("--------------- "+str(rep))
        print("MSE on test:", mse)
        print("")
        print("saving results...")
        file_name = "deepgmm_%d.npz" %(rep)
        save_path = os.path.join(folder, file_name)
        np.savez(save_path, x=test.w, y=test.y, g_true=test.g,
                 g_hat=g_pred_test.detach())



if __name__ == "__main__":
    scenarios = ["mendelian_{}_{}_{}".format(s, i, j) for s in [8, 16, 32] for i, j in [[1, 1]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[1, 0.5], [1, 2]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[0.5, 1], [2, 1]]]

    for sce in scenarios:
        for repid in range(10):
            run_experiment(sce, repid)

    means = []
    times = []
    for scenario_name in scenarios:
        folder = ROOT_PATH+"/results/mendelian/" + scenario_name + "/"
        means2 = []
        times2 = []
        for rep in range(10):
            file_name = "deepgmm_%d.npz" % (rep)
            save_path = folder+file_name
            res = np.load(save_path)
            means2 +=  [np.mean((res['g_hat']-res['g_true'])**2)]
            time_path = folder+"Ours_%d_time.npy" % (rep)
            res = np.load(time_path)
            times2 += [res]
        means += [means2]
        times += [times2]
    print(times)
    print('times',np.mean(times,axis=1),np.std(times,axis=1))
    mean = np.mean(means,axis=1)
    std = np.std(means,axis=1)
    sizes = [8,16,32]
    rows = ["({},{:.3f}) +- ({:.3f},{:.3f})".format(scenarios[j].split('_')[-2],mean[j],std[j],std[j]) for j in range(len(mean))]
    print('\n'.join(rows))

