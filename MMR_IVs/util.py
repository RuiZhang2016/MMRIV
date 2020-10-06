import os,sys
ROOT_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(ROOT_PATH)
import torch
from scenarios.abstract_scenario import AbstractScenario
import autograd.numpy as np
from joblib import Parallel, delayed
import torch.nn as nn
import torch.nn.functional as F
import autograd.scipy.linalg as splg

JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)

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

def remove_outliers(array):
    if not isinstance(array, np.ndarray):
        raise Exception('input type should be numpy ndarray, instead of {}'.format(type(array)))
    Q1 = np.quantile(array,0.25)
    Q3 = np.quantile(array,0.75)
    IQR = Q3 - Q1
    array = array[array<=Q3+1.5*IQR]
    array = array[ array>= Q1-1.5*IQR]
    return array


def nystrom_decomp(G,ind):
    Gnm = G[:,ind]
    sub_G = (Gnm)[ind,:]

    eig_val, eig_vec = np.linalg.eigh(sub_G)
    eig_vec = np.sqrt(len(ind) / G.shape[0]) * Gnm@eig_vec/eig_val
    eig_val /= len(ind) / G.shape[0]
    return eig_val, eig_vec

def nystrom_inv(W, ind):
    EYEN = np.eye(W.shape[0])
    eig_val, eig_vec = nystrom_decomp(W,ind)
    tmp = np.matmul(np.diag(eig_val),eig_vec.T)
    tmp = np.matmul(np.linalg.inv(JITTER*EYE_nystr +np.matmul(tmp,eig_vec)),tmp)
    W_inv = (EYEN - np.matmul(eig_vec,tmp))/JITTER
    return W_inv

def chol_inv(W):
    EYEN = np.eye(W.shape[0])
    try:
        tri_W = np.linalg.cholesky(W)
        tri_W_inv = splg.solve_triangular(tri_W,EYEN,lower=True)
        #tri_W,lower  = splg.cho_factor(W,lower=True)
        # W_inv = splg.cho_solve((tri_W,True),EYEN)
        W_inv = np.matmul(tri_W_inv.T,tri_W_inv)
        W_inv = (W_inv + W_inv.T)/2
        return W_inv
    except Exception as e:
        return False

class FCNN(nn.Module):

    def __init__(self,input_size):
        super(FCNN, self).__init__()
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
