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
from util import get_median_inter,get_median_inter_mnist, Kernel, load_data, ROOT_PATH,jitchol,_sqdist,remove_outliers
from joblib import Parallel,delayed
from early_stopping import EarlyStopping
from autograd import primitive
from scenarios.abstract_scenario import AbstractScenario
import time


Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None

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

def test_LMO_err(sname, seed,datasize, nystr=False):
    
    def LMO_err(params,M=2,verbal=False):
        n_params = len(params)
        # params = np.exp(params)
        al,bl = np.exp(params)
        L = bl*bl*np.exp(-L0/al/al/2) +1e-6*EYEN # l(X,None,al,bl)# +1e-6*EYEN
        if nystr:
            tmp_mat = L@eig_vec_K
            C = L-tmp_mat@np.linalg.inv(eig_vec_K.T@tmp_mat/N2+inv_eig_val_K)@tmp_mat.T/N2
            c = C@W_nystr_Y*N2
        else:
            LWL_inv = chol_inv(L@W@L+L/N2 +JITTER*EYEN)# chol_inv(W*N2+L_inv) # chol_inv(L@W@L+L/N2 +JITTER*EYEN)
            C = L@LWL_inv@L/N2
            c = C@W@Y*N2
        c_y = c-Y
        lmo_err = 0
        N = 0
        for ii in range(1):
            permutation = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices,indices)]*N2
                C_i = C[np.ix_(indices,indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(np.eye(M)-C_i@K_i)@c_y_i
                # print(I_CW_inv.shape,c_y_i.shape)
                lmo_err += b_y.T@K_i@b_y
                N += 1
        return lmo_err[0,0]/N/M**2
    
    def callback0(params, timer=None):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval % 1 == 0:
            n_params = len(params)
            al,bl = params # params[:int(n_params/2)], params[int(n_params/2):]
            # print('Nfeval: ', Nfeval, 'params: ', [al._value,bl._value, JITTER._value])
            L = l(X,None,al,bl)+1e-6*EYEN
            # L_inv = chol_inv(L+JITTER*EYEN)
            if nystr:
                # L_W_inv = L-L@eig_vec_K@chol_inv(eig_vec_K.T@L@eig_vec_K/N2+1/eig_val_K/N2)@eig_vec_K.T@L/N2
                alpha = EYEN-eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
                alpha = alpha@W_nystr@Y*N2
            else:
                LWL_inv = chol_inv(L@W@L+L/N2+JITTER*EYEN)
                alpha = LWL_inv@L@W@Y
                # L_W_inv = chol_inv(W*N2+L_inv)
            test_L = l(test_X,X,al,bl)
            pred_mean = test_L@alpha
            if timer:
                return
            test_err = ((pred_mean-test_G)**2).mean() # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha

        Nfeval += 1
        if prev_norm is not None:
            if norm[0,0]/prev_norm >=3:
                if opt_params is None:
                    opt_test_err = test_err
                    opt_params = params
                print(True,opt_params, opt_test_err,prev_norm)
                raise Exception

        if prev_norm is None or norm[0,0]<= prev_norm:
            prev_norm = norm[0,0]
        opt_test_err = test_err
        opt_params = params
        print('params,test_err, norm: ',opt_params, opt_test_err, prev_norm)


    funcs = {'sin':lambda x: np.sin(x),
            'step':lambda x: 0* (x<0) +1* (x>=0),
            'abs':lambda x: np.abs(x),
            'linear': lambda x: x}
    k,l = Kernel('rbf'), Kernel('rbf')#exp_sin_squared')# Kernel('rbf')
    snames = ['step','sin','abs','linear']
    M = 2
    # n_train, n_test = 200,2000
    
    train,dev,test = load_data('{}_{}'.format(sname,datasize))    

    X = np.vstack((train.x,dev.x))
    Y = np.vstack((train.y,dev.y))
    Z = np.vstack((train.z,dev.z))
    test_X = test.x
    test_G = test.g
    
    t0 = time.time()
    EYEN = np.eye(X.shape[0])
    ak = get_median_inter_mnist(Z)
    N2 = X.shape[0]**2
    W0 = _sqdist(Z,None)
    W = (np.exp(-W0/ak/ak/2)+np.exp(-W0/ak/ak/200)+np.exp(-W0/ak/ak*50))/3/N2
    del W0
    L0, test_L0 = _sqdist(X,None), _sqdist(test_X,X)
    # callback0(np.random.randn(2)/10,True)
    # np.save(ROOT_PATH + "/our_methods/results/zoo/" + sname + '/LMO_errs_{}_nystr_{}_time.npy'.format(seed,train.x.shape[0]),time.time()-t0)
    # return
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    params0 =np.random.randn(2)/10
    bounds =  None # [[0.01,10],[0.01,5]]
    if nystr:
        for _ in range(seed+1):
            random_indices = np.sort(np.random.choice(range(W.shape[0]),nystr_M,replace=False))
        eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices)
        inv_eig_val_K = np.diag(1/eig_val_K/N2)
        W_nystr = eig_vec_K @ np.diag(eig_val_K)@eig_vec_K.T/N2
        W_nystr_Y = W_nystr@Y
    
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    try:
        res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5000},callback=callback0)    
    except Exception as e:
        print(e)
    PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
    np.save(PATH+'LMO_errs_{}_nystr_{}.npy'.format(seed,train.x.shape[0]),[opt_params,prev_norm,opt_test_err])

def plot_res_2d(sname,datasize):
    print(sname)
    res = []
    times = []
    for i in range(100):
        PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
        filename = PATH+'LMO_errs_{}_nystr_{}.npy'.format(i,datasize)
        if os.path.exists(filename):
            tmp_res = np.load(filename,allow_pickle=True)
            if tmp_res[-1] is not None:
                res += [tmp_res[-1]]
        time_path = PATH+ '/LMO_errs_{}_nystr_{}_time.npy'.format(i,datasize)
        if os.path.exists(time_path):
            t = np.load(time_path)
            times += [t]
    res = np.array(res)
    times = np.array(times)
    res = remove_outliers(res)
    times = np.sort(times)[:80]
    print(times)
    # print(res)
    # res = [e for e in res if e < res.mean()+3*res.std() and e > res.mean()-3*res.std() ]
    # print(res)
    print('mean, std: ', np.mean(res), np.std(res))
    print('time: ', np.mean(times),np.std(times))

if __name__ == '__main__':
    snames = ['step','sin','abs','linear']
    if len(sys.argv)>=2:
        ind = int(sys.argv[1])
        datasize = int(sys.argv[2])
        sid, seed = divmod(ind,100)
        test_LMO_err(snames[sid],seed,datasize,False if datasize <1000 else True)
        # for sname in snames:
        #    print(sname)
        #    for seed in range(100):
        #        print(seed)
        #        test_LMO_err(sname,seed,datasize,False) 
    elif len(sys.argv)==1:
        for sname in snames:
            plot_res_2d(sname,2000)
    else:
        inds = [int(e) for e in sys.argv[1:]]
        alids,blids = [],[]
        for ind in inds:
            alid,blid = divmod(ind,32)
            alids += [alid]
            blids += [blid]
        Parallel(n_jobs = min(len(inds),20))(delayed(test_LMO_err)(alids[i],blids[i]) for i in range(len(alids))) 
    
    
    
    
    
    
    
    
    
    
    
