import torch
import numpy as np
import os,sys
from methods.mnist_x_model_selection_method import MNISTXModelSelectionMethod
from methods.mnist_xz_model_selection_method import MNISTXZModelSelectionMethod
from methods.mnist_z_model_selection_method import MNISTZModelSelectionMethod
from scenarios.abstract_scenario import AbstractScenario
from joblib import Parallel, delayed
from our_methods.util import ROOT_PATH, load_data
import random

random.seed(527)

SCENARIOS_NAMES = ["mnist_x", "mnist_z", "mnist_xz"]
SCENARIO_METHOD_CLASSES = {
    "mnist_x": MNISTXModelSelectionMethod,
    "mnist_z": MNISTZModelSelectionMethod,
    "mnist_xz": MNISTXZModelSelectionMethod,
}

RESULTS_FOLDER = ROOT_PATH + "/results/mnist/"


def run_experiment(scenario_name,repid,model_id,training=False):
    # set random seed
    seed = 527
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_reps = 10

    print("\nLoading " + scenario_name + "...")
    train, dev, test = load_data(ROOT_PATH+'/data/'+scenario_name+'/main.npz',Torch=True,verbal=True)
    means = []
    for rep in range(num_reps):
        method_class = SCENARIO_METHOD_CLASSES[scenario_name]
        method = method_class(enable_cuda=torch.cuda.is_available())
        if training:
            if rep < repid:
                continue
            elif rep >repid:
                break
            else:
                pass
            print('here')
            method.fit(train.x, train.z, train.y, dev.x, dev.z, dev.y,
                       g_dev=dev.g,rep=rep,model_id=model_id, verbose=True)
            g_pred_test = method.predict(test.x)
            mse = float(((g_pred_test - test.g) ** 2).mean())

            print("---------------")
            print("finished running methodology on scenario ",scenario_name)
            print("MSE on test:", mse)
            print("")
            print("saving results...")
            folder = ROOT_PATH+"/results/mnist/" + scenario_name + "/"
            file_name = "Ours_%d.npz" % rep
            save_path = os.path.join(folder, file_name)
            os.makedirs(folder, exist_ok=True)
            np.savez(save_path, x=test.w, y=test.y, g_true=test.g,
                     g_hat=g_pred_test.detach())
        else:
            folder = ROOT_PATH+"/results/mnist/" + scenario_name + "/"
            file_name = "Ours_%d.npz" % rep
            save_path = os.path.join(folder, file_name)
            if os.path.exists(save_path):
                res = np.load(save_path)
                means += [((res['g_true']-res['g_hat'])**2).mean()]
            else:
                print(save_path, ' not exists')
    return means

def main():
    for scenario in SCENARIOS_NAMES:
        run_experiment(scenario)


if __name__ == "__main__":
    if len(sys.argv)>1:
        ind = int(sys.argv[1])
        sid, ind = divmod(ind,10)
        model_id, repid = divmod(ind,10)
        model_id = -1
        # sid, repid = divmod(ind,10)
        run_experiment(SCENARIOS_NAMES[sid], repid,model_id, training=True)
        # Parallel(n_jobs=20)(delayed(run_experiment)(SCENARIOS_NAMES[sid], repid, 3,training=True)  for sid in range(3) for repid in range(10))
    else:
        for s in SCENARIOS_NAMES:
            means = run_experiment(s, 0, 3, training=False)
            print(means)
            mean = np.mean(means)
            std = np.std(means)
            print("{} {:.3f} $pm$ {:.3f}".format(s, mean,std))
