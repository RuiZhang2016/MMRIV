import os,sys,torch,add_path
import torch.autograd as ag
import torch.optim as optim
import numpy as np
from scenarios.abstract_scenario import AbstractScenario
from early_stopping import EarlyStopping
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy
from joblib import Parallel, delayed
from util import get_median_inter_mnist, Kernel, data_generate, load_data, ROOT_PATH,_sqdist


def run_experiment_nn(sname,indices=[],seed=527,training=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(indices)==2:
        lr_id, dw_id = indices
    elif len(indices)==3:
        lr_id, dw_id,W_id = indices
    # load data
    folder = ROOT_PATH+"/MMR_IVs/results/mendelian/"+sname+"/"
    train, dev, test = load_data(ROOT_PATH+"/data/mendelian/"+sname+'.npz',Torch=True)

    n_train = train.x.shape[0]
    # training settings
    n_epochs = 1000
    batch_size = 1000 if train.x.shape[0]>1000 else train.x.shape[0]

    # kernel
    kernel = Kernel('rbf',Torch=True)
    # training loop
    lrs = [2e-4,1e-4,5e-5] #[2e-5,1e-5,5e-6] # [3,5]
    decay_weights = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6] # [11,5]
    os.makedirs(folder, exist_ok=True)

    def my_loss(output, target, indices, K):
        d = output - target
        if indices is None:
            W = K
        else:
            W = K[indices[:, None], indices]
            # print((kernel(Z[indices],None,a,1)+kernel(Z[indices],None,a/10,1)+kernel(Z[indices],None,a*10,1))/3-W)
        loss = d.T @ W @ d / (d.shape[0]) ** 2
        return loss[0, 0]

    def fit(x,y,z,dev_x,dev_y,dev_z,lr,decay_weight,n_epochs=n_epochs):
        train_K = np.load(ROOT_PATH+'/mendelian_precomp/{}_train_K.npy'.format(sname))
        dev_K = np.load(ROOT_PATH+'/mendelian_precomp/{}_dev_K.npy'.format(sname))
        train_K = torch.from_numpy(train_K).float()
        dev_K = torch.from_numpy(dev_K).float()

        n_data = x.shape[0]
        net = Net(x.shape[1])
        es = EarlyStopping(patience=5)
        optimizer = optim.Adam(list(net.parameters()), lr=lr, weight_decay=decay_weight)

        for epoch in range(n_epochs):
            permutation = torch.randperm(n_data)

            for i in range(0, n_data, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = x[indices], y[indices]

                # training loop
                def closure():
                    optimizer.zero_grad()
                    pred_y = net(batch_x)
                    loss = my_loss(pred_y, batch_y, indices, train_K)
                    loss.backward()
                    return loss

                optimizer.step(closure)  # Does the update
            if epoch % 5 == 0 and epoch >= 5 and dev_x is not None: # 5, 10 for small # 5,50 for large 
                g_pred = net(test.x.float())
                test_err = ((g_pred-test.g.float())**2).mean()
                dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
                print('test',test_err,'dev',dev_err)
                if es.step(dev_err):
                    break
        return es.best, epoch, net

    
    if training is True:
        print('training')
        for rep in range(10):
            save_path = os.path.join(folder, 'mmr_iv_nn_{}_{}_{}_{}.npz'.format(rep,lr_id,dw_id,train.x.shape[0]))
            # if os.path.exists(save_path):
            #    continue
            lr,dw = lrs[lr_id],decay_weights[dw_id]
            print('lr, dw', lr,dw)
            t0 = time.time()
            err,_,net = fit(train.x.float(),train.y.float(),train.z.float(),dev.x.float(),dev.y.float(),dev.z.float(),lr,dw)
            t1 = time.time()-t0
            np.save(folder+'mmr_iv_nn_{}_{}_{}_{}_time.npy'.format(rep,lr_id,dw_id,train.x.shape[0]),t1)
            g_pred = net(test.x.float()).detach().numpy()
            test_err = ((g_pred-test.g.numpy())**2).mean()
            np.savez(save_path,err=err.detach().numpy(),lr=lr,dw=dw, g_pred=g_pred,test_err=test_err)
    else:
        print('test')
        opt_res = []
        times = []
        for rep in range(10):
            res_list = []
            other_list = []
            times2 = []
            for lr_id in range(len(lrs)):
                for dw_id in range(len(decay_weights)):
                    load_path = os.path.join(folder, 'mmr_iv_nn_{}_{}_{}_{}.npz'.format(rep,lr_id,dw_id,train.x.shape[0]))
                    if os.path.exists(load_path):
                        res = np.load(load_path)
                        res_list += [res['err'].astype(float)]
                        other_list += [[res['lr'].astype(float),res['dw'].astype(float),res['test_err'].astype(float)]]
                    time_path = folder+'mmr_iv_nn_{}_{}_{}_{}_time.npy'.format(rep,lr_id,dw_id,train.x.shape[0])
                    if os.path.exists(time_path):
                        t = np.load(time_path)
                        times2 += [t]
            res_list = np.array(res_list)
            other_list = np.array(other_list)
            other_list = other_list[res_list>0]
            res_list = res_list[res_list>0]
            optim_id = np.argsort(res_list)[0]# np.argmin(res_list)
            print(rep,'--',other_list[optim_id],np.min(res_list))
            opt_res += [other_list[optim_id][-1]]
            # times += [times2[optim_id]]
            # lr,dw = [torch.from_numpy(e).float() for e in params_list[optim_id]]
            # _,_,net = fit(X[:2000],Y[:2000],Z[:2000],X[2000:],Y[2000:],Z[2000:],a,lr,dw)
            # g_pred = net(test_X).detach().numpy()
            # test_err = ((g_pred-test_G.numpy())**2).mean()
            # print(test_err)
            # np.savez(save_path,g_pred=g_pred,g_true=test.g,x=test.w)
        print('time: ', np.mean(times),np.std(times))
        mean,std = np.mean(opt_res),np.std(opt_res)

        print("({},{:.3f}) +- ({:.3f},{:.3f})".format((sname.split('_'))[1],mean,std,std))



if __name__ == '__main__':
    scenarios = ["mendelian_{}_{}_{}".format(s, i, j) for s in [8, 16, 32] for i, j in [[1, 1]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[1, 0.5], [1, 2]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[0.5, 1], [2, 1]]]
    # index = int(sys.argv[1])
    # sid,index = divmod(index,21)
    # lr_id, dw_id = divmod(index,7)

    for s in scenarios:
        for lr_id in range(3):
            for dw_id in range(7):
                run_experiment_nn(s, [lr_id,dw_id])
    for s in scenarios:
        print(s)
        run_experiment_nn(s,[1, 0],training=False)
