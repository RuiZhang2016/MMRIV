import torch
import numpy as np
from baselines.all_baselines import Poly2SLS, Vanilla2SLS, DirectNN, \
    GMM, DeepIV, AGMM
import os
from scenarios.abstract_scenario import AbstractScenario
import tensorflow
from tabulate import tabulate
import scipy

def eval_model(model, test):
    g_pred_test = model.predict(test.x)
    mse = float(((g_pred_test - test.g) ** 2).mean())
    return mse


def save_model(model, save_path, test):
    g_pred = model.predict(test.x)
    np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=g_pred)


def run_experiment(scenario_name, num_reps=10, seed=527):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensorflow.set_random_seed(seed)

    scenario_path = "data/zoo/" + scenario_name + "_2000.npz"
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    # scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    # scipy.io.savemat(scenario_name + '_data_200.mat', mdict={'X_train': train.x, 'Y_train': train.y, 'Z_train': train.z,
    # 'X_dev': dev.x, 'Y_dev': dev.y, 'Z_dev': dev.z,'X_test': test.x, 'g_test': test.g})
    # return

    # result folder
    folder = "results/zoo/" + scenario_name + "/"
    os.makedirs(folder, exist_ok=True)

    means = []

    for rep in range(num_reps):
        # Not all methods are applicable in all scenarios
        methods = []

        # baseline methods
        # methods += [("DirectNN", DirectNN())]
        # methods += [("Vanilla2SLS", Vanilla2SLS())]
        # methods += [("Poly2SLS", Poly2SLS())]
        # methods += [("GMM", GMM(g_model="2-layer", n_steps=20))]
        methods += [("AGMM", AGMM())]
        methods += [("DeepIV", DeepIV())]

        means_rep = []
        for method_name, method in methods:
            print("Running " + method_name +" " + str(rep))
            file_name = "%s_%d_2000.npz" % (method_name, rep)
            save_path = os.path.join(folder, file_name)
            if os.path.exists(save_path) and False:
                res = np.load(save_path)
                means_rep += [float(((res['g_hat'] - res['g_true']) ** 2).mean())]
                # print('mse: {}'.format(float(((res['g_hat'] - res['g_true']) ** 2).mean())))
                continue
            else:
                model = method.fit(train.x, train.y, train.z, None)
                save_model(model, save_path, test)
                test_mse = eval_model(model, test)
                model_type_name = type(model).__name__
                print("Test MSE of %s: %f" % (model_type_name, test_mse))
        means += [means_rep]
    return means

def main():
    scenarios = np.array(["abs", "linear", "sin", "step"])
    rows = []
    for scenario in scenarios:
        print("\nLoading " + scenario + "...")
        means = run_experiment(scenario)
        # mean = np.mean(means,axis=0)
        # std = np.std(means,axis=0)
        # rows += [["{:.3f}$pm${:.3f}".format(mean[i],std[i]) for i in range(len(mean))]]
    return
    methods = np.array(["DirectNN","Vanilla2SLS","Poly2SLS","GMM","AGMM","DeepIV"])[:,None]
    print(methods,np.array(rows).T)
    rows = np.hstack((methods,np.array(rows).T))
    print('Tabulate Table:')
    print(tabulate(np.vstack((np.append([""],scenarios),rows)), headers='firstrow',tablefmt='latex'))


if __name__ == "__main__":
    main()

