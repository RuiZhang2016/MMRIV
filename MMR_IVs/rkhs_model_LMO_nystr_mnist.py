import os
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH,jitchol,_sqdist, \
data_generate,nystrom_decomp,remove_outliers,nystrom_decomp, chol_inv
from scipy.sparse import csr_matrix
import random
import time

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

def experiment(sname, seed, nystr=True):
    
    def LMO_err(params,M=2,verbal=False):
        global Nfeval
        params = np.exp(params)
        al,bl = params[:-1],params[-1] # params[:int(n_params/2)], params[int(n_params/2):] #  [np.exp(e) for e in params]
        if train.x.shape[1]<5:
            train_L = bl**2*np.exp(-train_L0/al**2/2)+1e-4*EYEN
        else:
            train_L,dev_L = 0,0
            for i in range(len(al)):
                train_L += train_L0[i]/al[i]**2
            train_L = bl*bl*np.exp(-train_L/2)+1e-4*EYEN


        tmp_mat = train_L@eig_vec_K
        C = train_L-tmp_mat@np.linalg.inv(eig_vec_K.T@tmp_mat/N2+inv_eig_val)@tmp_mat.T/N2
        c = C @ W_nystr_Y*N2
        c_y = c-train.y
        lmo_err = 0
        N = 0
        for ii in range(1):
            permutation = np.random.permutation(train.x.shape[0])
            for i in range(0,train.x.shape[0],M):
                indices = permutation[i:i + M]
                K_i = train_W[np.ix_(indices,indices)]*N2
                C_i = C[np.ix_(indices,indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(np.eye(M)-C_i@K_i)@c_y_i
                lmo_err += b_y.T@K_i@b_y
                N += 1
        return lmo_err[0,0]/M**2
    
    def callback0(params):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval %1 == 0:
            params = np.exp(params)
            print('params:',params)
            al,bl = params[:-1],params[-1]
            
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
                LWL_inv = chol_inv(train_L@train_W@train_L+train_L/N2+JITTER*EYEN)
                alpha = LWL_inv@train_L@train_W@train.y
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

    train, dev, test = load_data(ROOT_PATH+'/data/'+sname+'/main.npz')
    del dev

    # avoid same indices when run on the cluster
    for _ in range(seed+1):
        random_indices = np.sort(np.random.choice(range(train.x.shape[0]),nystr_M,replace=False))

    EYEN = np.eye(train.x.shape[0])
    N2 = train.x.shape[0]**2

    # precompute to save time on parallized computation
    if train.z.shape[1] < 5:
        ak = get_median_inter_mnist(train.z)
    else:
        ak = np.load(ROOT_PATH+'/mnist_precomp/{}_ak.npy'.format(sname))
    train_W = np.load(ROOT_PATH+'/mnist_precomp/{}_train_K0.npy'.format(sname))
    train_W = (np.exp(-train_W/ak/ak/2)+np.exp(-train_W/ak/ak/200)+np.exp(-train_W/ak/ak*50))/3/N2
    if train.x.shape[1]<5:
        train_L0 = _sqdist(train.x,None)
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
    bounds = None
    eig_val_K,eig_vec_K = nystrom_decomp(train_W*N2, random_indices)
    W_nystr_Y = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T@train.y/N2
    inv_eig_val = np.diag(1/eig_val_K/N2)
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5000,'disp':True,'ftol':0},callback=callback0)
    PATH = ROOT_PATH + "/MMR_IVs/results/"+ sname + "/"
    np.save(PATH+'LMO_errs_{}_nystr.npy'.format(seed),[opt_params,prev_norm,opt_test_err])

if __name__ == '__main__':
    snames = ['mnist_z','mnist_x','mnist_xz']
    for sname in snames:
        for seed in range(100):
            experiment(sname,seed)

        PATH = ROOT_PATH + "/MMR_IVs/results/"+ sname + "/"
        ress = []
        for seed in range(100):
            filename = PATH+'LMO_errs_{}_nystr.npy'.format(seed)
            if os.path.exists(filename):
                res = np.load(filename,allow_pickle=True)
                if res[-1] is not None:
                    ress += [res[-1]]
        ress = np.array(ress)
        ress = remove_outliers(ress)
        print(np.nanmean(ress),np.nanstd(ress))
    
    
    
    
    
    
    
    
    
    
    

