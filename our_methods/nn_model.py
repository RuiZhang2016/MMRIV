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

def Kernel(name):
    def poly(x,c,d):
        return (x @ x.T+c*c)**d

    def rbf(x,y,a,b):
        if y is None:
            y = x
        x,y = x/a, y/a
        x2,y2 = torch.sum(x*x,dim=1,keepdim=True),torch.sum(y*y,dim=1,keepdim=True)
        sqdist = x2+y2.T-2*x@y.T
        if y is None:
            sqdist = (sqdist+torch.abs(sqdist).T)/2
        out = b*b*torch.exp(-sqdist)
        return out

    def laplace(x,a):
        return 0

    def quad(x,y,a,b):
        x, y = x /a, y /a
        x2, y2 = torch.sum(x * x, dim=1, keepdim=True), torch.sum(y * y, dim=1, keepdim=True)
        sqdist = x2 + y2.T - 2 * x @ y.T
        out = (sqdist+1)**(-b)
        return out

    assert isinstance(name,str), 'name should be a string'
    kernel_dict = {'rbf':rbf,'poly':poly,'quad':quad}
    return kernel_dict[name]

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

def get_median_inter(x):
    n,m = x.shape
    def loop(a):
        A = a[:,None]# np.tile(x[:,[i]],[1,n])
        B = A.T
        dist = abs(A - B)
        dist = dist.flatten()
        med = np.median(dist)
        print(med)
        return med
    mat = np.array([loop(x[:,i]) for i in range(m)])
    return mat.reshape((1,-1))
    # b = x.T
    # b = b[:,:,np.newaxis]
    # c = b.reshape((b.shape[0],1,-1))
    # d = c-b
    # d = d.reshape((d.shape[0],-1))
    # d = np.abs(d)
    # return (np.median(d,axis=1))[:,None]

def get_median_inter_mnist(x):
    x2 = np.sum(x*x,axis=1,keepdims=True)
    sqdist = x2+x2.T-2*x@x.T
    dist = np.sqrt((sqdist+abs(sqdist).T)/2)
    return np.median(dist.flatten())



def run_experiment_nn(scenario_name,indices=[],seed=527,training=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(indices)==2:
        lr_id, dw_id = indices
    # load data
    print("\nLoading " + scenario_name + "...")
    if 'mnist' in scenario_name:
        scenario_path = "../data/" + scenario_name + "/main.npz"
        folder = "/home/ruizhang/DeepGMM/our_methods/results/" + scenario_name + "/"
    else:
        scenario_path = "../data/zoo/" + scenario_name + ".npz"
        folder = "/home/ruizhang/DeepGMM/our_methods/results/zoo/" + scenario_name + "/"

    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    x,z,y = torch.from_numpy(train.x).float(),torch.from_numpy(train.z).float(),torch.from_numpy(train.y).float()
    dev_x, dev_z, dev_y, test_x = [torch.from_numpy(e).float() for e in [dev.x, dev.z,dev.y, test.x]]

    # training settings
    n_epochs = 100
    batch_size = 1024

    # kernel
    kernel = Kernel('rbf')
    if dev_z.shape[1] < 5:
        a = get_median_inter(np.vstack((train.z,dev.z)))
    else:
        # a = get_median_inter_mnist(np.vstack((train.z,dev.z)))
        a = np.load('tmp/{}_k_params.npy'.format(scenario_name))
    # train_K = kernel(z,z,a,1)
    # dev_K = kernel(dev_z,dev_z,a,1)
    a = torch.from_numpy(a).float()
    # training loop
    lrs = [0.01,0.001,0.0001]
    decay_weights = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
    os.makedirs(folder, exist_ok=True)

    def my_loss(output, target, indices, K):
        d = output - target
        if indices is None:
            W = K
        else:
            W = K[indices[:, None], indices]
        loss = d.T @ W @ d / (d.shape[0]) ** 2
        return loss[0, 0]

    def fit(x,y,z,dev_x,dev_y,dev_z,a,lr,decay_weight,n_epochs=n_epochs):
        train_K = kernel(z, None, a, 1)
        if dev_z is not None:
            dev_K = kernel(dev_z,None,a,1)
        n_data = x.shape[0]

        net = Net(x.shape[1]) if scenario_name not in ['mnist_x','mnist_xz'] else CNN()
        es = EarlyStopping(patience=10)
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
            if epoch % 5 == 0 and epoch > 0 and dev_x is not None:
                dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
                if es.step(dev_err):
                    break
        return es.best, epoch, net

    def loop(lr,dw):
        err1,epoch1,net = fit(x,y,z,dev_x,dev_y,dev_z,a,lr,dw)
        # err2,epoch2,_ = fit(dev_x, dev_y, dev_z, x, y, z,a,lr,dw)
        return err1,net
    
    if training is True:
        print('training')
        for rep in range(10):
            save_path = os.path.join(folder, 'our_method_nn_{}_{}_{}.npz'.format(rep,lr_id,dw_id))
            # results = Parallel(n_jobs=len(lrs)*len(decay_weights))(delayed(loop)(lr,dw) for lr in lrs for dw in decay_weights)
            err,net = loop(lrs[lr_id],decay_weights[dw_id])
            g_pred = net(test_x).detach().numpy()
            test_err = ((g_pred-test.g)**2).mean()
            np.savez(save_path,err=err.detach().numpy(),lr=lrs[lr_id],dw=decay_weights[dw_id],g_pred=g_pred,test_err=test_err)
            #print('running {} rep: {} lr: {} dw: {} '.format(scenario_name,rep,lrs[lr_id],decay_weights[dw_id]))
    else:
        print('test')
        for rep in range(10):
            res_list = []
            params_list = []
            save_path = os.path.join(folder, 'our_method_nn_{}.npz'.format(rep))
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
    scenarios = ['mnist_z','mnist_x','mnist_xz']# ["step", "sin", "abs", "linear"]
    # run_experiment_nn(scenarios[scenario_id],[rep,lr_id,dw_id],test=True)
    Parallel(n_jobs=10)(delayed(run_experiment_nn)(s,[1,dw_id]) for s in scenarios for dw_id in range(6))
