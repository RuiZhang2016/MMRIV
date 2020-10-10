import add_path,os
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH,jitchol,_sqdist, remove_outliers,nystrom_decomp, chol_inv
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

def experiment(sname, seed,nystr=False):
    
    def LMO_err(params,M=2):
        al,bl = np.exp(params)
        L = bl*bl*np.exp(-L0/al/al/2) +1e-6*EYEN
        if nystr:
            tmp_mat = L@eig_vec_K
            C = L-tmp_mat@np.linalg.inv(eig_vec_K.T@tmp_mat/N2+inv_eig_val_K)@tmp_mat.T/N2
            c = C@W_nystr_Y*N2
        else:
            LWL_inv = chol_inv(L@W@L+L/N2 +JITTER*EYEN)
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
            al,bl = np.exp(params)
            L = bl*bl*np.exp(-L0/al/al/2) +1e-6*EYEN
            if nystr:
                tmp_mat = eig_vec_K.T@L
                alpha = EYEN-eig_vec_K@np.linalg.inv(tmp_mat@eig_vec_K/N2+inv_eig_val_K)@tmp_mat/N2
                alpha = alpha@W_nystr_Y*N2
            else:
                LWL_inv = chol_inv(L@W@L+L/N2+JITTER*EYEN)
                alpha = LWL_inv@L@W@Y
            test_L = bl*bl*np.exp(-test_L0/al/al/2)# l(test_X,X,al,bl)
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
                print(True,opt_params, opt_test_err,prev_norm, norm[0,0])
                raise Exception

        if prev_norm is None or norm[0,0]<= prev_norm:
            prev_norm = norm[0,0]
        opt_test_err = test_err
        opt_params = params
        print('params,test_err, norm: ',opt_params, opt_test_err, prev_norm, norm[0,0])
    
    folder = ROOT_PATH+"/MMR_IVs/results/mendelian/" + sname + "/"
    train, dev, test = load_data(ROOT_PATH+"/data/mendelian/"+sname+'.npz',Torch=False)
    os.makedirs(folder, exist_ok=True)

    X = train.x
    Y = train.y
    Z = train.z
    test_X = test.x
    test_G = test.g
    
    t0 = time.time()
    EYEN = np.eye(X.shape[0])
    N2 = X.shape[0]**2
    W = np.load(ROOT_PATH+'/mendelian_precomp/{}_train_K.npy'.format(sname))/N2
    L0, test_L0 = _sqdist(X,None), _sqdist(test_X,X)

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
    PATH = ROOT_PATH + "/MMR_IVs/results/mendelian/" + sname + "/"
    np.save(PATH+'LMO_errs_{}_nystr_{}.npy'.format(seed,train.x.shape[0]),[opt_params,prev_norm,opt_test_err])

def summarize_res(sname,datasize):
    print(sname)
    res = []
    times = []
    for i in range(100):
        PATH = ROOT_PATH + "/our_methods/results/mendelian/" + sname + "/"
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
    res = remove_outliers(res)
    print('mean, std: ', np.mean(res), np.std(res))

if __name__ == '__main__':
    scenarios = ["mendelian_{}_{}_{}".format(s, i, j) for s in [8, 16, 32] for i, j in [[1, 1]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[1, 0.5], [1, 2]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[0.5, 1], [2, 1]]]

    for sname in scenarios:
        for seed in range(100):
            experiment(sname,seed,True)
        summarize_res(sname,10000)
    
    
    
    
    
    
    
    
    
    
    

