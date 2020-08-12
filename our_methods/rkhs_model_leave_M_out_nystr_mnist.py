import add_path,os,sys
import numpy
import torch
import torch.optim as optim
import autograd.scipy.linalg as splg
import autograd.numpy as np
from scenarios.abstract_scenario import AbstractScenario
import matplotlib.pyplot as plt
from autograd import value_and_grad
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from util import get_median_inter,get_median_inter_mnist, Kernel, load_data, ROOT_PATH,jitchol,_sqdist, data_generate,nystrom_decomp
from joblib import Parallel,delayed
from early_stopping import EarlyStopping
from autograd import primitive
from scipy.sparse import csr_matrix
import autograd.scipy.stats.multivariate_normal as mvn
from torchvision import datasets, transforms
import random
from collections import defaultdict
from sklearn.decomposition import PCA

Nfeval = 1
seed = 527
np.random.seed(seed)
random.seed(seed)
JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)
__sparse_fmt = csr_matrix
opt_params = None
prev_norm = None
opt_test_err = None


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

def test_LMO_err(sname, seed,use_x_images,use_z_images, nystr=True):
    
    def LMO_err(params,M=2,verbal=False):
        n_params = len(params)
        # params = np.exp(params)
        al,bl = params # params[:int(n_params/2)], params[int(n_params/2):] #  [np.exp(e) for e in params]
        L = bl*bl*np.exp(-L0/al/al/2)+1e-6*EYEN # l(X,None,al,bl)+1e-6*EYEN# l(X,None,al,bl,cl)
        # L_inv = chol_inv(L)
        C = L-L@eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
        c = C@W_nystr@Y*N2
        c_y = c-Y
        lmo_err = 0
        N = 0
        for ii in range(4):
            permutation = np.random.permutation(X.shape[0])
            for i in range(0,X.shape[0],M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices,indices)]*N2
                C_i = C[np.ix_(indices,indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(np.eye(M)-C_i@K_i)@c_y_i
                # print(I_CW_inv.shape,c_y_i.shape)
                lmo_err += b_y.T@K_i@b_y
                N += 1
        return lmo_err[0,0]
    
    def callback0(params):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval % 1 == 0:
            n_params = len(params)
            al,bl = params # params[:int(n_params/2)], params[int(n_params/2):]
            # print('Nfeval: ', Nfeval, 'params: ', [al._value,bl._value, JITTER._value])
            L = bl*bl*np.exp(-L0/al/al/2)+1e-6*EYEN# np.exp(-L0/al/al/2)*bl*bl+1e-6*EYEN
            # L = l(X, None, al, bl)+1e-6*EYEN
            # L_inv = chol_inv(L+JITTER*EYEN)
            if nystr:
                alpha = EYEN-eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
                alpha = alpha@W_nystr@Y*N2
            else:
                LWL_inv = chol_inv(L@W@L+L/N2+JITTER*EYEN)
                alpha = LWL_inv@L@W@Y
            test_L = l(test_X,X,al,bl)# np.exp(-test_L0/al/al/2)*bl*bl # l(test_X, X, al,bl)# np.exp(-test_L0/al/al/2)*bl*bl
            pred_mean = test_L@alpha
            test_err = ((pred_mean-test_G)**2).mean()
            norm = alpha.T @ L @ alpha
        Nfeval += 1
        # if prev_norm is not None:
        #    if norm[0,0]/prev_norm >=3:
        #        if opt_test_err is None:
        #            prev_norm = norm[0,0]
        #            opt_test_err = test_err
        #            opt_params = params
        #        print(True,opt_params, opt_test_err,prev_norm)
        #        raise Exception
        
        #if prev_norm is None or norm[0,0]<= prev_norm:
        prev_norm = norm[0,0]
        opt_test_err = test_err
        opt_params = params
        print(True,opt_params, opt_test_err, prev_norm)

    funcs = {'sin':lambda x: np.sin(x),
            'step':lambda x: 0* (x<0) +1* (x>=0),
            'abs':lambda x: np.abs(x),
            'linear': lambda x: x}

    k,l = Kernel('rbf',False), Kernel('rbf', False)#exp_sin_squared')# Kernel('rbf')
    use_x_images = sname in ['mnist_x','mnist_xz']
    use_z_images = sname in ['mnist_z','mnist_xz']
    M = 2
    n_train, n_test =4000,20000
    X, Y, Z, test_X, test_G = [e.numpy() if torch.is_tensor(e) else e for e in data_generate('abs',n_train, n_test, use_x_images, use_z_images)]
    print(X[:5])
    # batch_size = 1000
    for _ in range(seed+1):
        random_indices = np.sort(np.random.choice(range(X.shape[0]),nystr_M,replace=False))
    params0 = [1.1,0.9]
    bounds =  [[1e-2,10],[1e-2,10]]
    EYEN = np.eye(Z.shape[0])
    N2 = X.shape[0]**2
    ak = get_median_inter_mnist(Z)
    W = (k(Z,None,ak,1)+k(Z,None,ak*10,1)+k(Z,None,ak/10,1))/3
    W /= N2
    L0, test_L0 = _sqdist(X,None), _sqdist(test_X,X)

    eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices)
    W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T/N2
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5,'disp':True},callback=callback0)

    #for epoch in range(100):
    #    batch_permutation = np.random.permutation(X.shape[0])
    #    for i in range(0, X.shape[0], batch_size):
    #        indices = batch_permutation[i:i + batch_size]
    #        Z_batch, X_batch, Y_batch = Z[indices], X[indices], Y[indices]
    #        EYEN = np.eye(Z_batch.shape[0])
    #        N2 = X_batch.shape[0]**2
    #        ak = 3 # get_median_inter_mnist(Z_batch)
    #        W = (k(Z_batch,None,ak,1)+k(Z_batch,None,ak*10,1)+k(Z_batch,None,ak/10,1))/3
    #        W /= N2
    #        L0, test_L0 = _sqdist(X_batch,None), _sqdist(test_X,X_batch)
    #        # np.save(ROOT_PATH+'/mnist_precomp/aks.npy',[ak,get_median_inter_mnist(Z)])
    #        eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices)
    #        W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T/N2
    #        obj_grad = value_and_grad(lambda params: LMO_err(params,Y_batch))
            #try:
    #        res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5,'disp':True},callback=callback0) 
            #except Exception as e:
            #    print(e)
    #        params0 = opt_params
        #PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
        #np.save(PATH+'LMO_errs_{}_nystr.npy'.format(seed),[opt_params,prev_norm,opt_test_err])

if __name__ == '__main__':
    snames = ['mnist_z', 'mnist_x','mnist_xz']
    if len(sys.argv)==2:
        ind = int(sys.argv[1])
        sid, seed = divmod(ind,100)
        test_LMO_err(snames[sid],seed, True, False)
    elif len(sys.argv)==1:
        pass
        # plot_res_2d('abs')
    else:
        inds = [int(e) for e in sys.argv[1:]]
        alids,blids = [],[]
        for ind in inds:
            alid,blid = divmod(ind,32)
            alids += [alid]
            blids += [blid]
        Parallel(n_jobs = min(len(inds),20))(delayed(test_LMO_err)(alids[i],blids[i]) for i in range(len(alids))) 
    
    
    
    
    
    
    
    
    
    
    

