import os,sys
ROOT_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(ROOT_PATH)
import torch
from scenarios.abstract_scenario import AbstractScenario
import autograd.numpy as np
import logging
from joblib import Parallel, delayed
import random
from torchvision import datasets, transforms
from collections import defaultdict
import math

def get_median_inter(x):
    n,m = x.shape
    def loop(a):
        A = a[:,None]# np.tile(x[:,[i]],[1,n])
        B = A.T
        dist = abs(A - B)
        dist = dist.flatten()
        med = np.median(dist)
        return med
    mat = np.array([loop(x[:,i]) for i in range(m)])
    return mat.reshape((1,-1))

def _sqdist(x,y,Torch=False):
    if y is None:
        y = x
    if Torch:
        diffs = torch.unsqueeze(x,1)-torch.unsqueeze(y,0)
        sqdist = torch.sum(diffs**2, axis=2, keepdim=False)
    else:
        diffs = np.expand_dims(x,1)-np.expand_dims(y,0)
        sqdist = np.sum(diffs**2, axis=2)
        del diffs
    return sqdist

def get_median_inter_mnist(x):
    # x2 = np.sum(x*x,axis=1,keepdims=True)
    # sqdist = x2+x2.T-2*x@x.T
    # sqdist = (sqdist+abs(sqdist).T)/2
    if x.shape[0]< 10000:
        sqdist = _sqdist(x,None)
    else:
        M = int(x.shape[0]/400)
        sqdist = Parallel(n_jobs=20)(delayed(_sqdist)(x[i:i+M],x) for i in range(0,x.shape[0],M))
    dist = np.sqrt(sqdist)
    return np.median(dist.flatten())

def load_data(scenario_path,verbal=False, Torch=False):
    # load data
    # print("\nLoading " + scenario_name + "...")
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    if verbal:
        scenario.info()
    if Torch:
        scenario.to_tensor()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    return train, dev, test

def Kernel(name, Torch=False):
    def poly(x,y,c,d):
        if y is None:
            y = x
            res = (x @ y.T+c*c)**d
            res = (res+res.T)/2
            return res
        else:
            return (x @ y.T+c*c)**d
    

    def rbf(x,y,a,b,Torch=Torch):
        if y is None:
            y = x
        # sqdist = x2+y2.T-2*np.matmul(x,y.T)
        if x.shape[0]< 60000:
            sqdist = _sqdist(x,y,Torch)/a/a
        else:
            M = int(x.shape[0]/400)
            sqdist = np.vstack([_sqdist(x[i:i+M],y,Torch) for i in range(0,x.shape[0],M)])/a/a
        # elements can be negative due to float errors
        out = torch.exp(-sqdist/2) if Torch else np.exp(-sqdist/2)
        return out*b*b
   
    def rbf2(x,y,a,b,Torch=Torch):
        if y is None:
            y = x
        x, y = x/a, y/a
        return b*b*np.exp(-_sqdist(x,y)/2)

    def mix_rbf(x,y,a,b,Torch=False):
        res = 0
        for i in range(len(a)):
            res += rbf(x,y,a[i],b[i],Torch)
        return res

    def laplace(x,a):
        return 0

    def quad(x,y,a,b):
        x, y = x/a, y/a
        x2, y2 = torch.sum(x * x, dim=1, keepdim=True), torch.sum(y * y, dim=1, keepdim=True)
        sqdist = x2 + y2.T - 2 * x @ y.T
        out = (sqdist+1)**(-b)
        return out
    
    def exp_sin_squared(x,y,a,b,c):
        if y is None:
            y = x
        diffs = np.expand_dims(x,1)-np.expand_dims(y,0)
        sqdist = np.sum(diffs**2, axis=2)
        assert np.all(sqdist>=0),sqdist[sqdist<0]
        out = b*b*np.exp(-np.sin(sqdist/c*np.pi)**2/a**2*2)
        return out
    # return the kernel function
    assert isinstance(name,str), 'name should be a string'
    kernel_dict = {'rbf':rbf,'poly':poly,'quad':quad, 'mix_rbf':mix_rbf,'exp_sin_squared':exp_sin_squared,'rbf2':rbf2}
    return kernel_dict[name]


def jitchol(A, maxtries=5):
    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise np.linalg.LinAlgError("not pd: non-positive diagonal elements")
    jitter = diagA.mean() * 1e-6
    num_tries = 1
    while num_tries <= maxtries and np.isfinite(jitter):
        try:
            L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
            return L
        except:
            jitter *= 10
        finally:
            num_tries += 1
    raise np.linalg.LinAlgError("not positive definite, even with jitter.")


def data_generate(sname, n_train, n_test, use_x_images, use_z_images):
    funcs = {'sin':lambda x: np.sin(x),
            'step':lambda x: 0* (x<0) +1* (x>=0),
            'abs':lambda x: np.abs(x),
            'linear': lambda x: x}

    digit_dict = None
    if use_x_images or use_z_images:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(ROOT_PATH+"/datasets", train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])), batch_size=60000)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(ROOT_PATH+"/datasets", train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])), batch_size=10000)
        train_data, test_data = list(train_loader), list(test_loader)
        images_list = [train_data[0][0].numpy(), test_data[0][0].numpy()]
        labels_list = [train_data[0][1].numpy(), test_data[0][1].numpy()]
        del train_data, test_data

        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        idx = list(range(images.shape[0]))
        random.shuffle(idx)
        images = images[idx]
        labels = labels[idx]
        digit_dict = defaultdict(list)
        for label, image in zip(labels, images):
            digit_dict[int(label)].append(image)
        del images, labels, idx

    Z,test_Z = np.random.uniform(-3,3,size=(n_train,2)),np.random.uniform(-3,3,size=(n_test,2))# np.random.normal(0,np.sqrt(2),size=(n_train,1))# np.random.uniform(-3,3,size=(n_train+n_test,2))
    cofounder,test_cofounder = np.random.normal(0,1,size=(n_train,1)),np.random.normal(0,1,size=(n_test,1))
    gamma,test_gamma = np.random.normal(0,.1,size=(n_train,1)),np.random.normal(0,.1,size=(n_test,1))
    delta, test_delta = np.random.normal(0,.1,size=(n_train,1)),np.random.normal(0,.1,size=(n_test,1))
    X = Z[:,[0]]+cofounder+gamma
    test_X = test_Z[:,[0]] + test_cofounder+test_gamma
    
    if sname in funcs.keys():
        func = funcs[sname]
    else:
        func = funcs['abs']
    Y =func(X) + cofounder+delta
    test_G = func(test_X)
    train_Y = Y[:int(n_train/2)]
    
    if use_x_images:
        X_digits = np.clip(1.5*X + 5.0, 0, 9).round()
        test_X_digits = np.clip(1.5*test_X + 5.0, 0, 9).round()
        X = np.stack([random.choice(digit_dict[int(d)]).flatten() for d in X_digits.flatten()], axis=0)
        test_X =  np.stack([random.choice(digit_dict[int(d)]).flatten() for d in test_X_digits.flatten()], axis=0)
        test_G = np.abs((test_X_digits - 5.0) / 1.5).reshape(-1, 1)
    if use_z_images:
        Z_digits = np.clip(1.5*Z[:, [0]] + 5.0, 0, 9).round()
        Z = np.stack([random.choice(digit_dict[int(d)]).flatten() for d in Z_digits.flatten()], axis=0)
    test_G = (test_G - train_Y.mean())/train_Y.std()
    Y = (Y-train_Y.mean())/train_Y.std()
    return X,Y,Z,test_X, test_G

def nystrom_decomp(G,ind, Torch=False):
    Gnm = G[:,ind]
    sub_G = (Gnm)[ind,:]
    
    if Torch:
        eig_val, eig_vec = torch.symeig(sub_G,eigenvectors=True)
        #eig_vec = math.sqrt(len(ind) / G.shape[0]) * Gnm@eig_vec/eig_val
    else:
        eig_val, eig_vec = np.linalg.eigh(sub_G)
        # eig_vec = np.sqrt(len(ind) / G.shape[0]) * Gnm@eig_vec/eig_val
    eig_vec = math.sqrt(len(ind) / G.shape[0]) * Gnm@eig_vec/eig_val
    eig_val /= len(ind) / G.shape[0]

    if Torch:
        return eig_val.float(), eig_vec.float()
    else:
        return eig_val, eig_vec

def remove_outliers(array):
    if not isinstance(array, np.ndarray):
        raise Exception('input type should be numpy ndarray, instead of {}'.format(type(array)))
    Q1 = np.quantile(array,0.25)
    Q3 = np.quantile(array,0.75)
    IQR = Q3 - Q1
    array = array[array<=Q3+1.5*IQR]
    array = array[ array>= Q1-1.5*IQR]
    return array
