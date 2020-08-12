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
from util import get_median_inter,get_median_inter_mnist, Kernel, load_data, ROOT_PATH,jitchol
from joblib import Parallel,delayed
from early_stopping import EarlyStopping
from autograd import primitive
from scipy.sparse import csr_matrix
import autograd.scipy.stats.multivariate_normal as mvn

Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)
__sparse_fmt = csr_matrix
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

def test_LMO_err(sname, seed, nystr=True):
    
    def LMO_err(params,M=2,verbal=False):
        n_params = len(params)
        # params = np.exp(params)
        al,bl = params # params[:int(n_params/2)], params[int(n_params/2):] #  [np.exp(e) for e in params]
        L = l(X,None,al,bl)+1e-6*EYEN # l(X,None,al,bl,cl)
        # L_inv = chol_inv(L)
        if nystr:
            C = L-L@eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
        else:
            LWL_inv = chol_inv(L@W@L+L/N2 +JITTER*EYEN)# chol_inv(W*N2+L_inv) # chol_inv(L@W@L+L/N2 +JITTER*EYEN)
            C = L@LWL_inv@L/N2
            # LWL_inv = chol_inv(L@W@L*N2+L)
        c = C@W_nystr@Y*N2
        c_y = c-Y
        lmo_err = 0
        N = 0
        for ii in range(4):
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
    
    def callback0(params):
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
            test_err = ((pred_mean-test_G)**2).mean() # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha
            # plt.plot(test_X, test_G, '.',label='true')
            # plt.plot(test_X, pred_mean, '.',label='true')
            # plt.legend()
            #plt.savefig('tmp_step_{}.pdf'.format(Nfeval))
            #plt.close('all')
        Nfeval += 1
        if prev_norm is not None:
            if norm[0,0]/prev_norm >=3:
                if opt_params is None:
                    prev_norm = norm[0,0]
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
    n_train, n_test = 4000,2000
    Z,test_Z = np.random.uniform(-3,3,size=(n_train,2)),np.random.uniform(-3,3,size=(n_test,2))# np.random.normal(0,np.sqrt(2),size=(n_train,1))# np.random.uniform(-3,3,size=(n_train+n_test,2))
    cofounder,test_cofounder = np.random.normal(0,1,size=(n_train,1)),np.random.normal(0,1,size=(n_test,1))
    gamma,test_gamma = np.random.normal(0,np.sqrt(0.1),size=(n_train,1)),np.random.normal(0,np.sqrt(0.1),size=(n_test,1))
    delta, test_delta = np.random.normal(0,np.sqrt(0.1),size=(n_train,1)),np.random.normal(0,np.sqrt(0.1),size=(n_test,1))
    X = Z[:,[0]]+cofounder+gamma
    test_X = test_Z[:,[0]] + test_cofounder+test_gamma# np.linspace(np.percentile(X,5),np.percentile(X,95),100)[:,None]# X# X[n_train:,]# np.sort(X[n_train:,])
    EYEN = np.eye(X.shape[0])
    ak = get_median_inter_mnist(Z)
    N2 = X.shape[0]**2
    W = (k(Z,None,ak,1)+k(Z,None,ak*10,1)+k(Z,None,ak/10,1))/3
    W /= N2
     
    func = funcs[sname]
    Y =func(X) + cofounder+delta# func(X)+cofounder[:n_train,]+delta# L0@np.ones((n_train,1))/n_train + cofounder[:n_train,]+delta
    test_G = func(test_X)# l(test_X,X,1,1)@np.ones((n_train,1))/n_train # func(test_X)
    test_G = (test_G - Y.mean())/Y.std()
    Y = (Y-Y.mean())/Y.std()
    # np.savez('data_{}'.format(sname),X=X,Z=Z,Y=Y,test_X=test_X,test_G=test_G)
    # np.savez('data_random_{}'.format(i),X=X,Z=Z,Y=Y,test_X=test_X,test_G=test_G)
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    # test_err,norm = callback0([al,bl])
    params0 = [1.1,0.9]
    bounds =  [[0.3,10],[0.1,5]]
    for _ in range(seed+1):
        random_indices = np.sort(np.random.choice(range(W.shape[0]),nystr_M,replace=False))
    eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices)
    W_nystr = eig_vec_K @ np.diag(eig_val_K)@eig_vec_K.T/N2
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    try:
        res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5000,'disp':True},callback=callback0) 
    except Exception as e:
        # print('Finish optimization: ', opt_params, opt_test_err)
        print(e)
    if opt_test_err is None:
        callback0(params0)
    PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
    np.save(PATH+'LMO_errs_{}_nystr.npy'.format(seed),[opt_params,prev_norm,opt_test_err])

def plot_res_2d(sname):
    print(sname)
    res = []
    for i in range(100):
        PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
        filename = PATH+'LMO_errs_{}_nystr.npy'.format(i)
        if os.path.exists(filename):
            tmp_res = np.load(filename,allow_pickle=True)
            print(i, ' ' ,tmp_res)
            if tmp_res[-1] is not None:
                res += [tmp_res[-1]]
    res = np.array(res)
    # print(res)
    # res = [e for e in res if e < res.mean()+3*res.std() and e > res.mean()-3*res.std() ]
    # print(res)
    print('mean, std: ', np.mean(res), np.std(res))
    


if __name__ == '__main__':
    snames = ['step','sin','abs','linear']
    if len(sys.argv)==2:
        ind = int(sys.argv[1])
        sid, seed = divmod(ind,100)
        test_LMO_err(snames[sid],seed) 
    elif len(sys.argv)==1:
        for sname in snames:
            plot_res_2d(sname)
    else:
        inds = [int(e) for e in sys.argv[1:]]
        alids,blids = [],[]
        for ind in inds:
            alid,blid = divmod(ind,32)
            alids += [alid]
            blids += [blid]
        Parallel(n_jobs = min(len(inds),20))(delayed(test_LMO_err)(alids[i],blids[i]) for i in range(len(alids))) 
    
    
    
    
    
    
    
    
    
    
    

