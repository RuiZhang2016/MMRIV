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
from util import get_median_inter, get_median_inter_mnist, Kernel, data_generate,nystrom_decomp,_sqdist
import time
nystr_M = 300

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
        self.fc1 = nn.Linear(16 * 4 * 4, 10)
        # self.fc2 = nn.Linear(100, 2)
        # self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x) # F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
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
    
    M = 2
    n_train, n_test =4000,10000

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
    lrs = [0.01,0.001,0.0001]
    decay_weights = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
    # os.makedirs(folder, exist_ok=True)

    def my_loss(target, L,eig_vec_K,inv_eig_val,W,W_nystr_Y):
        N2 = torch.tensor(L.shape[0]**2).float()
        # L_inv = chol_inv(L)
        print('11')
        tmp_mat = L@eig_vec_K
        C = L-tmp_mat@torch.inverse(eig_vec_K.T@tmp_mat/N2+inv_eig_val)@tmp_mat.T/N2
        c = C@W_nystr_Y*N2

        c_y = c-target
        
        lmo_err = 0
        N = 0
        M = 2
        for ii in range(1):
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
    
    def fit(x,y,z,dev_x,dev_y,dev_z,lr,decay_weight,n_epochs=0):
        n_data = x.shape[0]
        N2  = torch.tensor(len(y)**2).float()
        net1 = CNN()
        bl = torch.tensor(1.0, requires_grad=True).float()
        es = EarlyStopping(patience=10)
        optimizer = optim.Adam(list(net1.parameters())+[bl], lr=lr, weight_decay=decay_weight)
        ak = 3
        print('time',time.time())
        W = (k(z,None,ak,1)+k(z,None,ak*10,1)+k(z,None,ak/10,1))/3/N2
        random_indices = np.sort(np.random.choice(range(x.shape[0]),nystr_M,replace=False))
        eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices,True)
        W_nystr_Y = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T@y/N2
        inv_eig_val = torch.diag(1/eig_val_K/N2)
        print('time',time.time())
        for epoch in range(n_epochs):
            # training loop
            def closure():
                optimizer.zero_grad()
                feature_x = net1(x)
                # feature_test_x = net1(test_X)
                # feature_dev_x = net1(dev_x) # net1(x)
                L = l(feature_x,None,1,bl)
                # test_L = l(feature_test_x,feature_x,1,bl)
                loss1 = my_loss(y, L,eig_vec_K,inv_eig_val,W,W_nystr_Y)
                loss1.backward()
                return loss1

            optimizer.step(closure)  # Does the update

            if epoch % 1 == 0 and epoch > 0 and dev_x is not None:
                feature_x = net1(x)
                feature_dev_x = net1(dev_x)
                feature_test_x = net1(test_X)
                dev_L = l(feature_dev_x,feature_x,1,bl)
                test_L = l(feature_test_x,feature_x,1,bl)
                dev_W = (k(dev_z,None,ak,1)+k(dev_z,None,ak*10,1)+k(dev_z,None,ak/10,1))/3/dev_z.shape[0]**2
                L = l(feature_x,None,1,bl)
                tmp_mat = eig_vec_K.T@L
                alpha = torch.eye(L.shape[0])-eig_vec_K@torch.inverse(tmp_mat@eig_vec_K/N2+inv_eig_val)@tmp_mat/N2
                alpha = alpha@W_nystr_Y*N2
                pred_mean = test_L@alpha
                test_err = ((pred_mean.detach().numpy()-test_G.numpy())**2).mean()
                norm = alpha.T @ L @ alpha
                dev_err = (dev_y-dev_L@alpha).T@dev_W@(dev_y-dev_L@alpha)
                print(test_err,dev_err)
                if es.step(dev_err):
                    break
        return es.best, epoch, net1


    X, Y, Z, test_X, test_G = [torch.from_numpy(e).float() for e in data_generate('abs',n_train, n_test, use_x_images, use_z_images)]

    half_n = int(n_train/2)
    if training is True:
        print('training')
        for rep in range(10):
            # save_path = os.path.join(folder, 'our_method_nn_{}_{}_{}.npz'.format(rep,lr_id,dw_id))
            # results = Parallel(n_jobs=len(lrs)*len(decay_weights))(delayed(loop)(lr,dw) for lr in lrs for dw in decay_weights)
            
            err,_,net = fit(X[:half_n],Y[:half_n],Z[:half_n],X[half_n:],Y[half_n:],Z[half_n:],lrs[lr_id],decay_weights[dw_id],1000)
            # g_pred = net(test_X).detach().numpy()
            # test_err = ((g_pred-test_G)**2).mean()
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

    run_experiment_nn('mnist_x',[1,1])
