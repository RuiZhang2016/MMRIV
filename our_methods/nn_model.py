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
from util import get_median_inter, get_median_inter_mnist, Kernel, data_generate


class Net(nn.Module):

    def __init__(self,input_size):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 64)  # 6*6 from image dimension
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

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


def run_experiment_nn(sname,indices=[],seed=527,training=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(indices)==2:
        lr_id, dw_id = indices
    
    # load data
    print("\nGenerating " + sname + "...")
    if 'mnist' in sname:
        folder = "/home/ruizhang/MMR/our_methods/results/" + sname + "/"
    else:
        folder = "/home/ruizhang/MMR/our_methods/results/zoo/" + sname + "/"

    # train, dev, test = load_data(sname)
    # x,z,y = torch.from_numpy(train.x).float(),torch.from_numpy(train.z).float(),torch.from_numpy(train.y).float()
    # dev_x, dev_z, dev_y, test_x = [torch.from_numpy(e).float() for e in [dev.x, dev.z,dev.y, test.x]]
    n_train, n_test = 4000, 2000
    use_x_images = sname in ['mnist_x','mnist_xz']
    use_z_images = sname in ['mnist_z','mnist_xz']
    X, Y, Z, test_X, test_G = [e.float() if torch.is_tensor(e) else torch.from_numpy(e).float() for e in data_generate(sname,n_train, n_test, use_x_images, use_z_images)]
    # training settings
    n_epochs = 100
    batch_size = 2000

    # kernel
    kernel = Kernel('rbf',Torch=True)
    kernel2 = Kernel('rbf',Torch=False)
    if Z.shape[1] < 5:
        a = get_median_inter_mnist(Z)
    else:
        # a = get_median_inter_mnist(np.vstack((:train.z,dev.z)))
        a = np.load('tmp/{}_k_params.npy'.format(sname))
    a = torch.tensor(a).float()
    # training loop
    lrs = [0.001,0.01,0.001,0.0001]
    decay_weights = [1e-3,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
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
        train_K = (kernel(z, None, a, 1)+kernel(z, None, a/10, 1)+kernel(z, None, a*10, 1))/3
        if dev_z is not None:
            dev_K = (kernel(dev_z,None,a,1)+kernel(dev_z,None,a/10,1)+kernel(dev_z,None,a*10,1))/3
        n_data = x.shape[0]

        net = Net(x.shape[1]) if sname not in ['mnist_x','mnist_xz'] else CNN()
        es = EarlyStopping(patience=5)
        optimizer = optim.Adam(list(net.parameters()), lr=lr, weight_decay=decay_weight)

        for epoch in range(n_epochs):
            permutation = torch.randperm(n_data)

            for i in range(0, n_data, batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x[indices], y[indices]

                # training loop
                def closure():
                    optimizer.zero_grad()
                    pred_y = net(batch_x)
                    loss = my_loss(pred_y, batch_y, indices, train_K)
                    loss.backward()
                    return loss

                optimizer.step(closure)  # Does the update
            if epoch % 2 == 0 and epoch > 20 and dev_x is not None:
                dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
                g_pred = net(test_X)
                test_err = ((g_pred-test_G)**2).mean()
                print('test',test_err,'dev',dev_err)
                if es.step(dev_err):
                    break
        return es.best, epoch, net

    def loop(lr,dw):
        err1,epoch1,net = fit(X[:2000],Y[:2000],Z[:2000],X[2000:],Y[2000:],Z[2000:],a,lr,dw)
        # err2,epoch2,_ = fit(dev_x, dev_y, dev_z, x, y, z,a,lr,dw)
        return err1,net
    
    if training is True:
        print('training')
        for rep in range(10):
            save_path = os.path.join(folder, 'our_method_nn_{}_{}_{}.npz'.format(rep,lr_id,dw_id))
            # results = Parallel(n_jobs=len(lrs)*len(decay_weights))(delayed(loop)(lr,dw) for lr in lrs for dw in decay_weights)
            lr,dw = lrs[lr_id],decay_weights[dw_id]
            err,_,net = fit(X[:2000],Y[:2000],Z[:2000],X[2000:],Y[2000:],Z[2000:],a,lr,dw)# loop(lrs[lr_id],decay_weights[dw_id])
            g_pred = net(test_X).detach().numpy()
            test_err = ((g_pred-test_G.numpy())**2).mean()
            np.savez(save_path,err=err.detach().numpy(),lr=lr,dw=dw, g_pred=g_pred,test_err=test_err)
            #print('running {} rep: {} lr: {} dw: {} '.format(sname,rep,lrs[lr_id],decay_weights[dw_id]))
    else:
        print('test')
        for rep in range(10):
            res_list = []
            other_list = []
            save_path = os.path.join(folder, 'our_method_nn_{}.npz'.format(rep))
            for lr_id in range(3):
                for dw_id in range(6):
                    load_path = os.path.join(folder, 'our_method_nn_{}_{}_{}.npz'.format(rep,lr_id,dw_id))
                    res = np.load(load_path)
                    res_list += [res['err'].astype(float)]
                    other_list += [[res['lr'].astype(float),res['dw'].astype(float),res['test_err'].astype(float)]]
            print(res_list)
            print(other_list)
            optim_id = np.argmin(res_list)
            print(other_list[optim_id])
            # lr,dw = [torch.from_numpy(e).float() for e in params_list[optim_id]]
            # _,_,net = fit(X[:2000],Y[:2000],Z[:2000],X[2000:],Y[2000:],Z[2000:],a,lr,dw)
            # g_pred = net(test_X).detach().numpy()
            # test_err = ((g_pred-test_G.numpy())**2).mean()
            # print(test_err)
            # np.savez(save_path,g_pred=g_pred,g_true=test.g,x=test.w)



if __name__ == '__main__': 
    # index = int(sys.argv[1])
    # scenario_id,index = divmod(index,120)
    # rep, index = divmod(index,12)
    # lr_id, dw_id = divmod(index,4)
    scenarios = ["step", "sin", "abs", "linear"]# ['mnist_z','mnist_x','mnist_xz']# ["step", "sin", "abs", "linear"]
    # run_experiment_nn(scenarios[scenario_id],[rep,lr_id,dw_id],test=True)
    # Parallel(n_jobs=20)(delayed(run_experiment_nn)(scenarios[0],[lr_id,dw_id]) for lr_id in range(3) for dw_id in range(6))
    run_experiment_nn(scenarios[0],[0,0])
    # for s in scenarios[:1]:
    #     run_experiment_nn(s,[1, 0],training=False)
