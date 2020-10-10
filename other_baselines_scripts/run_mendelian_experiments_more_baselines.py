import torch, add_path
import numpy as np
from baselines.all_baselines import Poly2SLS, Vanilla2SLS, DirectNN, \
    GMM, DeepIV, AGMM
import os
import tensorflow
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


def run_experiment(scenario_name,mid,repid, num_reps=10, seed=527,training=False):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensorflow.set_random_seed(seed)

    train, dev, test = load_data(ROOT_PATH + "/data/mendelian/" + scenario_name+'.npz')

    # result folder
    folder = ROOT_PATH + "/results/mendelian/"+scenario_name+"/"
    os.makedirs(folder, exist_ok=True)
    means = []
    times = []
    for rep in range(num_reps):
        # Not all methods are applicable in all scenarios
        methods = []

        # baseline methods
        methods += [("DirectNN", DirectNN())]
        methods += [("Vanilla2SLS", Vanilla2SLS())]
        methods += [("Poly2SLS", Poly2SLS())]
        methods += [("GMM", GMM(g_model="2-layer", n_steps=20))]
        methods += [("AGMM", AGMM())]
        methods += [("DeepIV", DeepIV())]

        if training:
            if rep < repid:
                continue
            elif rep >repid:
                break
            else:
                pass
            for method_name, method in methods[mid:mid+1]:
                print("Running " + method_name +" " + str(rep))
                file_name = "%s_%d.npz" % (method_name, rep)
                save_path = os.path.join(folder, file_name)
                
                model, time = method.fit(train.x, train.y, train.z, None)
                np.save(folder+"%s_%d_time.npy" % (method_name, rep),time)
                save_model(model, save_path, test)
                test_mse = eval_model(model, test)
                model_type_name = type(model).__name__
                print("Test MSE of %s: %f" % (model_type_name, test_mse))
        else:
            means2 = []
            times2 = []
            for method_name, method in methods:
                # print("Running " + method_name +" " + str(rep))
                file_name = "%s_%d.npz" % (method_name, rep)
                save_path = os.path.join(folder, file_name)
                if os.path.exists(save_path):
                    res = np.load(save_path)
                    mse = float(((res['g_hat'] - res['g_true']) ** 2).mean())
    #                print('mse: {}'.format(mse))
                    means2 += [mse]
                else:
                    print(save_path, ' not exists')
                time_path = folder+"%s_%d_time.npy" % (method_name, rep)
                if os.path.exists(time_path):
                    res = np.load(time_path)
                    times2 += [res]
                else:
                    print(time_path, ' not exists')
            if len(means2) == len(methods):
                means += [means2]
            if len(times2) == len(methods):
                times += [times2]
    #print('means',np.mean(np.array(means),axis=0))
    #print('std',np.std(np.array(means),axis=0))
    return means,times


if __name__ == "__main__":
    scenarios = ["mendelian_{}_{}_{}".format(s, i, j) for s in [8,16,32] for i,j in [[1,1]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[1, 0.5],[1, 2]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j)for i, j in [[0.5, 1],[2, 1]]]

    for sce in scenarios:
        for mid in range(6):
            for repid in range(10):
                run_experiment(sce, mid, repid, training=True)

    rows = []
    for i in range(len(scenarios)):
        s = scenarios[i]
        means,times = run_experiment(s,0,0,training=False)
        mean = np.mean(means,axis=0)
        std = np.std(means,axis=0)
        rows += [["({},{:.4f}) +- ({:.3f},{:.3f})".format(s,mean[j],std[j],std[j]) for j in range(len(mean))]]
        print('time: ',np.mean(times,axis=0),np.std(times,axis=0))

    # methods = np.array(["DirectNN","Vanilla2SLS","Poly2SLS","GMM","AGMM","DeepIV"])[:,None]
    rows = np.array(rows)
    #rows = np.vstack((methods,rows))
    print('addplot+[mark=*,error bars/.cd, y dir=both,y explicit] coordinates'.join(['{'+'\n'.join(e)+'};\n' for e in rows.T]))
    print('Tabulate Table:')
    # print(tabulate(np.vstack((np.append([""],scenarios),rows)), headers='firstrow',tablefmt='latex'))

