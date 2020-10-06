import torch,add_path
import numpy as np
from baselines import all_baselines
from baselines.all_baselines import Poly2SLS, Vanilla2SLS, DirectNN, \
    DirectMNIST, GMM
import os
import tensorflow
from tabulate import tabulate
from MMR_IVs.util import ROOT_PATH, load_data
import random
random.seed(527)


def eval_model(model, test):
    g_pred_test = model.predict(test.x)
    mse = float(((g_pred_test - test.g) ** 2).mean())
    return mse


def save_model(model, save_path, test):
    g_pred = model.predict(test.x)
    np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=g_pred)


def run_experiment(scenario_name, mid, repid, num_reps=10, seed=527,training=False):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensorflow.set_random_seed(seed)
    
    train, dev, test = load_data(ROOT_PATH+'/data/'+scenario_name+'/main.npz',verbal=True)
    print(np.mean(train.x,axis=1))

    means = []
    for rep in range(num_reps):
        # Not all methods are applicable in all scenarios

        methods = []

        # baseline methods
        poly2sls_method = Poly2SLS(poly_degree=[1],
                                   ridge_alpha=np.logspace(-5, 3, 5))
        direct_method = None
        gmm_method = None
        deep_iv = all_baselines.DeepIV()
        if scenario_name == "mnist_z":
            deep_iv = all_baselines.DeepIV(treatment_model="cnn")
            gmm_method = GMM(
                g_model="2-layer", n_steps=10, g_epochs=10)
            direct_method = DirectNN()
        elif scenario_name == "mnist_x":
            gmm_method = GMM(g_model="mnist", n_steps=10, g_epochs=1)
            direct_method = DirectMNIST()
        elif scenario_name == "mnist_xz":
            deep_iv = all_baselines.DeepIV(treatment_model="cnn")
            gmm_method = GMM(g_model="mnist", n_steps=10, g_epochs=1)
            direct_method = DirectMNIST()

        methods += [("DirectNN", direct_method)]
        methods += [("Vanilla2SLS", Vanilla2SLS())]
        methods += [("Ridge2SLS", poly2sls_method)]
        methods += [("GMM", gmm_method)]
        methods += [("DeepIV", deep_iv)]
        
        if training:
            if rep < repid:
                continue
            elif rep >repid:
                break
            else:
                pass
            
            for method_name, method in methods[mid:mid+1]:
                print("Running " + method_name)
                model,time = method.fit(train.x, train.y, train.z, None)
                folder = ROOT_PATH+"/results/mnist/" + scenario_name + "/"
                np.save(folder+'{}_{}_time.npy'.format(method_name,rep),time)
                file_name = "%s_%d.npz" % (method_name, rep)
                save_path = os.path.join(folder, file_name)
                os.makedirs(folder, exist_ok=True)
                save_model(model, save_path, test)
                test_mse = eval_model(model, test)
                model_type_name = type(model).__name__
                print("Test MSE of %s: %f" % (model_type_name, test_mse))
        else:
            mean_rep = []
            for method_name, method in methods:
                folder = ROOT_PATH+"/results/mnist/" + scenario_name + "/"
                file_name = "%s_%d.npz" % (method_name, rep)
                save_path = os.path.join(folder, file_name)
                res = np.load(save_path)
                mean_rep += [((res['g_true']-res['g_hat'])**2).mean()]
            print(mean_rep)
            means += [mean_rep]
    return means



if __name__ == "__main__":
    scenarios = np.array(["mnist_z", "mnist_x", "mnist_xz"])
    for s in scenarios:
        for mid in range(5):
            for repid in range(10):
                run_experiment(s, mid, repid, num_reps=10, seed=527,training=True)

    rows = []
    for s in scenarios:
        means = run_experiment(s, 0, 0, training=False)
        mean = np.mean(means,axis=0)
        std = np.std(means,axis=0)
        rows += [["{:.3f} $pm$ {:.3f}".format(mean[i],std[i]) for i in range(len(mean))]]

    methods = np.array(["DirectNN","Vanilla2SLS","Ridge2SLS","GMM+NN","DeepIV"])[:,None]
    rows = np.hstack((methods,np.array(rows).T))
    print(tabulate(np.vstack((np.append([""],scenarios),rows)), headers='firstrow',tablefmt='latex'))
