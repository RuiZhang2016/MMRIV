import os,sys,torch,add_path
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scenarios.abstract_scenario import AbstractScenario
from early_stopping import EarlyStopping
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy
from joblib import Parallel, delayed
from util import get_median_inter, get_median_inter_mnist, Kernel, data_generate, load_data, ROOT_PATH,_sqdist


class Net(nn.Module):

    def __init__(self,input_size):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def run_experiment_nn(sname,datasize,indices=[],seed=527,training=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(indices)==2:
        lr_id, dw_id = indices
    elif len(indices)==3:
        lr_id, dw_id,W_id = indices
    # load data
    if 'mnist' in sname:
        folder = ROOT_PATH+"/our_methods/results/" + sname + "/"
    else:
        folder = ROOT_PATH+"/our_methods/results/zoo/" + sname + "/"
    
    if 'mnist' not in sname:
        train, dev, test = load_data(ROOT_PATH+"/data/zoo/"+sname+'/main_{}.npz'.format(datasize),Torch=True)
    else:
        train, dev, test = load_data(ROOT_PATH+"/data/"+sname+'/main.npz',Torch=True)
    X,Z,Y = torch.cat((train.x,dev.x),dim=0).float(),torch.cat((train.z,dev.z),dim=0).float(),torch.cat((train.y,dev.y),dim=0).float()
    test_X, test_G = test.x.float(),test.g.float()
    n_train = train.x.shape[0]
    # training settings
    n_epochs = 1000
    batch_size = 1000 if train.x.shape[0]>1000 else train.x.shape[0]

    # kernel
    kernel = Kernel('rbf',Torch=True)
    if Z.shape[1] < 5:
        a = get_median_inter_mnist(train.z)
    else:
        # a = get_median_inter_mnist(train.z)
        # np.save('../mnist_precomp/{}_ak.npy'.format(sname),a)
        a = np.load(ROOT_PATH+'/mnist_precomp/{}_ak.npy'.format(sname))
    a = torch.tensor(a).float()
    # training loop
    lrs = [2e-4,1e-4,5e-5] # [3,5]
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

    def fit(x,y,z,dev_x,dev_y,dev_z,a,lr,decay_weight,n_epochs=n_epochs):
        if 'mnist' in sname:
            train_K = torch.eye(x.shape[0])
        else:
            train_K = (kernel(z, None, a, 1)+kernel(z, None, a/10, 1)+kernel(z, None, a*10, 1))/3
        if dev_z is not None:
            if 'mnist' in sname:
                dev_K = torch.eye(x.shape[0])
            else:
                dev_K = (kernel(dev_z, None, a, 1)+kernel(dev_z, None, a/10, 1)+kernel(dev_z, None, a*10, 1))/3
        n_data = x.shape[0]
        net = Net(x.shape[1]) if sname not in ['mnist_x','mnist_xz'] else CNN()
        es = EarlyStopping(patience=5) # 10 for small
        optimizer = optim.Adam(list(net.parameters()), lr=lr, weight_decay=decay_weight)
        # optimizer = optim.SGD(list(net.parameters()),lr=1e-1, momentum=0.9)
        # optimizer = optim.Adadelta(list(net.parameters()))

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
            if epoch % 5 == 0 and epoch >= 50 and dev_x is not None: # 5, 10 for small # 5,50 for large 
                g_pred = net(test_X)
                test_err = ((g_pred-test_G)**2).mean()
                if epoch == 50 and 'mnist' in sname:
                    if z.shape[1] > 100:
                        train_K = np.load(ROOT_PATH+'/mnist_precomp/{}_train_K0.npy'.format(sname))
                        train_K = (torch.exp(-train_K/a**2/2)+torch.exp(-train_K/a**2*50)+torch.exp(-train_K/a**2/200))/3
                        dev_K = np.load(ROOT_PATH+'/mnist_precomp/{}_dev_K0.npy'.format(sname))
                        dev_K = (torch.exp(-dev_K/a**2/2)+torch.exp(-dev_K/a**2*50)+torch.exp(-dev_K/a**2/200))/3
                    else:
                        train_K = (kernel(z, None, a, 1)+kernel(z, None, a/10, 1)+kernel(z, None, a*10, 1))/3
                        dev_K = (kernel(dev_z, None, a, 1)+kernel(dev_z, None, a/10, 1)+kernel(dev_z, None, a*10, 1))/3

                dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
                print('test',test_err,'dev',dev_err)
                if es.step(dev_err):
                    break
        return es.best, epoch, net

    
    if training is True:
        print('training')
        for rep in range(10):
            save_path = os.path.join(folder, 'our_method_nn_{}_{}_{}_{}.npz'.format(rep,lr_id,dw_id,train.x.shape[0]))
            # if os.path.exists(save_path):
            #    continue
            lr,dw = lrs[lr_id],decay_weights[dw_id]
            print('lr, dw', lr,dw)
            t0 = time.time()
            err,_,net = fit(X[:n_train],Y[:n_train],Z[:n_train],X[n_train:],Y[n_train:],Z[n_train:],a,lr,dw)
            t1 = time.time()-t0
            np.save(folder+'our_method_nn_{}_{}_{}_{}_time.npy'.format(rep,lr_id,dw_id,train.x.shape[0]),t1)
            g_pred = net(test_X).detach().numpy()
            test_err = ((g_pred-test_G.numpy())**2).mean()
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
                    load_path = os.path.join(folder, 'our_method_nn_{}_{}_{}_{}.npz'.format(rep,lr_id,dw_id,datasize))
                    if os.path.exists(load_path):
                        res = np.load(load_path)
                        res_list += [res['err'].astype(float)]
                        other_list += [[res['lr'].astype(float),res['dw'].astype(float),res['test_err'].astype(float)]]
                    time_path = folder+'our_method_nn_{}_{}_{}_{}_time.npy'.format(rep,lr_id,dw_id,datasize)
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
        print(np.mean(opt_res),np.std(opt_res))



if __name__ == '__main__': 
    scenarios = ['mnist_z','mnist_x','mnist_xz']#['mnist_z','mnist_x','mnist_xz'] # ["step", "sin", "abs", "linear"]

        # index = int(sys.argv[1])
        # datasize = int(sys.argv[2])
        # sid,index = divmod(index,21)
        # lr_id, dw_id = divmod(index,7)
    datasize = 10000
    for s in scenarios:
        for lr_id in range(3):
            for dw_id in range(7):
                run_experiment_nn(s,datasize, [lr_id,dw_id])

    for s in scenarios:
        run_experiment_nn(s,datasize,[1, 0],training=False)
