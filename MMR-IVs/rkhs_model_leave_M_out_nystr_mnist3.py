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
from util import get_median_inter,get_median_inter_mnist, Kernel, load_data, ROOT_PATH,jitchol,_sqdist, data_generate,nystrom_decomp,remove_outliers
from joblib import Parallel,delayed
from early_stopping import EarlyStopping
from autograd import primitive
from scipy.sparse import csr_matrix
import autograd.scipy.stats.multivariate_normal as mvn
from torchvision import datasets, transforms
import random
from collections import defaultdict
from sklearn.decomposition import PCA
import time 
from sklearn.preprocessing import StandardScaler

Nfeval = 1
seed = 527
# np.random.seed(seed)
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

def test_LMO_err(sname, seed, nystr=True):
    
    def LMO_err(params,M=2,verbal=False):
        global Nfeval
        n_params = len(params)
        params = np.exp(params)
        al,bl = params[:-1],params[-1] # params[:int(n_params/2)], params[int(n_params/2):] #  [np.exp(e) for e in params]
        if train.x.shape[1]<5:
            train_L = bl**2*np.exp(-train_L0/al**2/2)+1e-4*EYEN
        #    dev_L = bl**2*np.exp(-dev_L0/al**2/2)
        else:
            train_L,dev_L = 0,0
            for i in range(len(al)):
                train_L += train_L0[i]/al[i]**2
         #       dev_L += dev_L0[i]/al[i]**2
            train_L = bl*bl*np.exp(-train_L/2)+1e-4*EYEN
         #   dev_L = bl*bl*np.exp(-dev_L/2)


        tmp_mat = train_L@eig_vec_K
        C = train_L-tmp_mat@np.linalg.inv(eig_vec_K.T@tmp_mat/N2+inv_eig_val)@tmp_mat.T/N2
        c = C @ W_nystr_Y*N2
        c_y = c-train.y # dev.y
        #if Nfeval >=0:
        #    return (c_y.T @ dev_W @ c_y)[0,0]
        lmo_err = 0
        N = 0
        t0 = time.time()
        for ii in range(1):
            permutation = np.random.permutation(train.x.shape[0])
            for i in range(0,train.x.shape[0],M):
                indices = permutation[i:i + M]
                K_i = train_W[np.ix_(indices,indices)]*N2
                C_i = C[np.ix_(indices,indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(np.eye(M)-C_i@K_i)@c_y_i
                # print(I_CW_inv.shape,c_y_i.shape)
                lmo_err += b_y.T@K_i@b_y
                N += 1
        return lmo_err[0,0]/M**2
    
    def callback0(params):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval %1 == 0:
            n_params = len(params)
            params = np.exp(params)
            print('params:',params)
            al,bl = params[:-1],params[-1] # params[:int(n_params/2)], params[int(n_params/2):]
            
            if train.x.shape[1]<5:
                train_L = bl**2*np.exp(-train_L0/al**2/2)+1e-4*EYEN
                test_L = bl**2*np.exp(-test_L0/al**2/2)
            else:
                train_L,test_L = 0,0
                for i in range(len(al)):
                    train_L += train_L0[i]/al[i]**2
                    test_L += test_L0[i]/al[i]**2
                train_L = bl*bl*np.exp(-train_L/2)+1e-4*EYEN
                test_L = bl*bl*np.exp(-test_L/2)

            if nystr:
                tmp_mat = eig_vec_K.T@train_L
                alpha = EYEN-eig_vec_K@np.linalg.inv(tmp_mat@eig_vec_K/N2+inv_eig_val)@tmp_mat/N2
                alpha = alpha@W_nystr_Y*N2
            else:
                LWL_inv = chol_inv(train_L@W@train_L+train_L/N2+JITTER*EYEN)
                alpha = LWL_inv@train_L@W@Y
            # test_L = l(test_X/al,X/al,1,bl)# np.exp(-np.sum(np.array([test_L0[i]/al[i]/al[i]/2 for i in range(len(al))]),axis=0))*bl*bl # l(test_X, X, al,bl)# np.exp(-test_L0/al/al/2)*bl*bl
            pred_mean = test_L@alpha
            test_err = ((pred_mean-test.g)**2).mean()
            norm = alpha.T @ train_L @ alpha
        Nfeval += 1
        if prev_norm is not None:
            if norm[0,0]/prev_norm >=3:
                if opt_test_err is None:
                    opt_test_err = test_err
                    opt_params = params
                print(True,opt_params, opt_test_err,prev_norm, norm[0,0])
                raise Exception
        
        if prev_norm is None or norm[0,0]<= prev_norm:
            prev_norm = norm[0,0]
        opt_test_err = test_err
        opt_params = params
        print(True,opt_params, opt_test_err, prev_norm, norm[0,0])

    # k,l = Kernel('rbf',False), Kernel('rbf', False)
    M = 2
    train, dev, test = load_data(ROOT_PATH+'/data/'+sname+'/main.npz')
    del dev 
    for _ in range(seed+1):
        random_indices = np.sort(np.random.choice(range(train.x.shape[0]),nystr_M,replace=False))

    EYEN = np.eye(train.x.shape[0])
    N2 = train.x.shape[0]**2
    if train.z.shape[1] < 5:
        ak = get_median_inter_mnist(train.z)
    else:
        # a = get_median_inter_mnist(train.z)
        # np.save('../mnist_precomp/{}_ak.npy'.format(sname),a)
        ak = np.load(ROOT_PATH+'/mnist_precomp/{}_ak.npy'.format(sname))
    train_W = np.load(ROOT_PATH+'/mnist_precomp/{}_train_K0.npy'.format(sname))# _sqdist(Z,None)
    train_W = (np.exp(-train_W/ak/ak/2)+np.exp(-train_W/ak/ak/200)+np.exp(-train_W/ak/ak*50))/3/N2
    # dev_W = np.load(ROOT_PATH+'/mnist_precomp/{}_dev_K0.npy'.format(sname))# _sqdist(Z,None)
    # dev_W = (np.exp(-dev_W/ak/ak/2)+np.exp(-dev_W/ak/ak/200)+np.exp(-dev_W/ak/ak*50))/3/N2
    if train.x.shape[1]<5:
        train_L0 = _sqdist(train.x,None)
        # dev_L0 = _sqdist(dev.x, train.x)
        test_L0 = _sqdist(test.x,train.x)
    else:
        L0s=np.load(ROOT_PATH+'/mnist_precomp/{}_Ls.npz'.format(sname))
        train_L0 = L0s['train_L0']
        # dev_L0 = L0s['dev_L0']
        test_L0 = L0s['test_L0']
        del L0s
    if train.x.shape[1]<5:
        params0 = np.random.randn(2)*0.1
    else:
        params0 = np.random.randn(len(train_L0)+1)*0.1
    bounds = None# [[1e-6,10]]*X.shape[1]+[[1e-1,10]]
    eig_val_K,eig_vec_K = nystrom_decomp(train_W*N2, random_indices)
    W_nystr_Y = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T@train.y/N2
    inv_eig_val = np.diag(1/eig_val_K/N2)
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5000,'disp':True,'ftol':0},callback=callback0)
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
    if 'mnist' in sname:
        PATH = ROOT_PATH + "/our_methods/results/"+ sname + "/"
    else:
        PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
    np.save(PATH+'LMO_errs_{}_nystr.npy'.format(seed),[opt_params,prev_norm,opt_test_err])

if __name__ == '__main__':
    snames = ['mnist_z', 'mnist_x','mnist_xz']
    # for s in snames:
    #     test_LMO_err(s,0, True, False)
    # assert 1 ==0
    if len(sys.argv)==2:
        ind = int(sys.argv[1])
        sid, seed = divmod(ind,100)
        test_LMO_err(snames[sid],seed)
    elif len(sys.argv)==1:
        for s in snames:
            print(s)
            PATH = ROOT_PATH + "/our_methods/results/"+ s + "/"
            ress = []
            for seed in range(100):
                filename = PATH+'LMO_errs_{}_nystr.npy'.format(seed)
                if os.path.exists(filename):
                    res = np.load(filename,allow_pickle=True)
                    if res[-1] is not None:
                        ress += [res[-1]]
            ress = np.array(ress)
            ress = remove_outliers(ress)
            #print(ress,len(ress))
            print(np.nanmean(ress),np.nanstd(ress))
        # plot_res_2d('abs')
    else:
        inds = [int(e) for e in sys.argv[1:]]
        alids,blids = [],[]
        for ind in inds:
            alid,blid = divmod(ind,32)
            alids += [alid]
            blids += [blid]
        Parallel(n_jobs = min(len(inds),20))(delayed(test_LMO_err)(alids[i],blids[i]) for i in range(len(alids))) 
    
    
    
    
    
    
    
    
    
    
    

