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
from util import get_median_inter, get_median_inter_mnist, Kernel, data_generate,nystrom_decomp
from sklearn.decomposition import PCA

class Net(nn.Module):

    def __init__(self,input_size):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 100)  # 6*6 from image dimension
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
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
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


def run_experiment_nn(scenario_name,indices=[],seed=527,training=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(indices)==2:
        lr_id, dw_id = indices
    # load data
    print("\nLoading " + scenario_name + "...")
    
    k,l = Kernel('rbf',True), Kernel('rbf', True)#exp_sin_squared')# Kernel('rbf')
    use_x_images = scenario_name in ['mnist_x','mnist_xz']
    use_z_images = scenario_name in ['mnist_z','mnist_xz']
    # training settings
    n_epochs = 1000
    batch_size = 1000
    
    M = 2
    n_train, n_test =20000,20000

    # kernel
    # if dev_z.shape[1] < 5:
    #     a = get_median_inter(np.vstack((train.z,dev.z)))
    # else:
        # a = get_median_inter_mnist(np.vstack((train.z,dev.z)))
    #    a = np.load('tmp/{}_k_params.npy'.format(scenario_name))
    # train_K = kernel(z,z,a,1)
    # dev_K = kernel(dev_z,dev_z,a,1)
    # a = torch.from_numpy(a).float()
    # training loop
    lrs = [0.01,0.001,0.0001,1e-5]
    decay_weights = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
    # os.makedirs(folder, exist_ok=True)

    def my_loss(target, L,alpha,W):
        N2 = torch.tensor(L.shape[0]**2).float()
        c_y = L@alpha-target
        
        lmo_err = 0
        N = 0
        M = 2
        for ii in range(4):
            permutation = np.random.permutation(len(target))
            for i in range(0, len(target), M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices,indices)]*N2
                C_i = C[np.ix_(indices,indices)]
                c_y_i = c_y[indices]
                b_y = torch.inverse(torch.eye(M).float()-C_i.float()@K_i)@c_y_i
                lmo_err += b_y.t()@K_i@b_y
                N += 1
        return lmo_err[0,0]/N/M/M
    
    def test_err(target,L,test_L,alpha):
        # N2= L.shape[0]**2
        pred_mean = test_L@alpha
        test_err = ((pred_mean-target)**2).mean()
        norm = 0#alpha.t() @ L @ alpha
        print('test_err, norm: ', test_err,norm)
        return test_err

    def fit(x,y,z,lr,decay_weight,n_epochs=n_epochs):
        n_data = x.shape[0]
        net1,net2 = CNN(), Net(1)
        bl = torch.tensor(1.0, requires_grad=True).float()
        alpha = torch.tensor([[1.0/n_train]]*n_train, requires_grad=True).float()
        # es = EarlyStopping(patience=10)
        
        optimizer = optim.Adam(list(net1.parameters()), lr=lr, weight_decay=decay_weight)
        # x= torch.from_numpy(np.random.randn(n_train,1)).float()
        for epoch in range(n_epochs):
            # training loop
            permutation = torch.randperm(n_test)
            for i in range(0, n_test, batch_size):
                indices = permutation[i:i + n_test]
                batch_x, batch_y, batch_z = test_X[indices], test_G[indices], None#x[indices], y[indices], z[indices]

                def closure():
                    optimizer.zero_grad()
                    feature_x = net1(batch_x)
                    # L = l(feature_x,None,1,bl)
                    # test_L = l(net1(test_X),feature_x,1,bl)
                    loss1 = 0# my_loss(batch_y,L,alpha, W)
                    # pred_y= net2(feature_x)
                    loss2 = 0# (y-L@alpha).t()@W@(y-L@alpha)
                    # print(epoch,feature_x,loss2)
                    test_error = ((feature_x-batch_y)**2).mean()# test_err(test_G,L,test_L,alpha)                    # (loss1+loss2).backward()
                    test_error.backward()
                    print(test_error)
                    return test_error

            optimizer.step(closure)  # Does the update
            # if epoch % 5 == 0 and epoch > 0 and dev_x is not None:
            #    dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
            #    if es.step(dev_err):
            #        break
        return 


    X, Y, Z, test_X, test_G = [torch.from_numpy(e).float() for e in data_generate('abs',n_train, n_test, use_x_images, use_z_images)]

    
    if training is True:
        print('training')
        for rep in range(10):
            # save_path = os.path.join(folder, 'our_method_nn_{}_{}_{}.npz'.format(rep,lr_id,dw_id))
            # results = Parallel(n_jobs=len(lrs)*len(decay_weights))(delayed(loop)(lr,dw) for lr in lrs for dw in decay_weights)
            err,_,net = fit(X,Y,Z,lrs[lr_id],decay_weights[dw_id])
            g_pred = net(test_X).detach().numpy()
            test_err = ((g_pred-test_G)**2).mean()
            # np.savez(save_path,err=err.detach().numpy(),lr=lrs[lr_id],dw=decay_weights[dw_id],g_pred=g_pred,test_err=test_err)
            # print('running {} rep: {} lr: {} dw: {} '.format(scenario_name,rep,lrs[lr_id],decay_weights[dw_id]))
    else:
        print('test')
        for rep in range(10):
            res_list = []
            params_list = []
            # save_path = os.path.join(folder, 'our_method_nn_{}.npz'.format(rep))
            for lr_id in range(3):
                for dw_id in range(4):
                    load_path = os.path.join(folder, 'our_method_nn_{}_{}_{}.npz'.format(rep,lr_id,dw_id))
                    res = np.load(load_path)
                    res_list += [res['res'].astype(float)]
                    params_list += [[res['lr'].astype(float),res['dw'].astype(float)]]
            optim_id = np.argmin(res_list)
            lr,dw = [torch.from_numpy(e).float() for e in params_list[optim_id]]
            _,_,net = fit(x,y,z,dev_x,dev_y,dev_z,a,lr,dw)
            g_pred = net(test_x).detach().numpy()
            test_err = ((g_pred-test.g)**2).mean()
            print(test_err)
            # np.savez(save_path,g_pred=g_pred,g_true=test.g,x=test.w)



if __name__ == '__main__': 
    # index = int(sys.argv[1])
    # scenario_id,index = divmod(index,120)
    # rep, index = divmod(index,12)
    # lr_id, dw_id = divmod(index,4)
    # scenarios = ['mnist_z','mnist_x','mnist_xz']# ["step", "sin", "abs", "linear"]
    # run_experiment_nn(scenarios[scenario_id],[rep,lr_id,dw_id],test=True)
    # Parallel(n_jobs=10)(delayed(run_experiment_nn)(s,[1,dw_id]) for s in scenarios for dw_id in range(6))

    run_experiment_nn('mnist_x',[0,0])
