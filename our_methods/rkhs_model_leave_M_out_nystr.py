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
import scipy.ndimage as spni
Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-5
nystr_M = 512
EYE_nystr = np.eye(nystr_M)
__sparse_fmt = csr_matrix

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
        print(e)
        return False

def test_LMO_err(i,j, seed=527, nystr=True):
    
    def LMO_err(params,M=2,verbal=False):
        n_params = len(params)
        # params = np.exp(params)
        c_y = c-Y
        
        lmo_err = 0
        N = 0
        EYEM = np.eye(M)
        for ii in range(4):
            permutation = np.random.permutation(Z.shape[0])
            for i in range(0, X.shape[0], M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices,indices)]*N2
                C_i = C[np.ix_(indices,indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(EYEM-C_i@K_i)@c_y_i
                # print(I_CW_inv.shape,c_y_i.shape)
                lmo_err += b_y.T@K_i@b_y
                # lmo_err += b_y.T@K_i@c_y_i
                # (_,slogdet) = np.linalg.slogdet(EYEM-C_i@K_i)
                # lmo_err -= slogdet

                N += 1
        return lmo_err[0,0]/N/M**2

    def callback0(params):
        global Nfeval
        if Nfeval % 1 == 0:
            #  print('Nfeval: ', Nfeval, ' al, bl, mean: ', params, ' nlm: ', LMO_err(params,M))
            if nystr:
                                # L_W_inv = L-L@eig_vec_K@chol_inv(eig_vec_K.T@L@eig_vec_K/N2+1/eig_val_K/N2)@eig_vec_K.T@L/N2
                alpha = EYEN-eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
                alpha = alpha@eig_vec_K@np.diag(eig_val_K)@eig_vec_K.T@Y
            else:
                LWL_inv = chol_inv(L@W@L+L/N2+JITTER*EYEN)
                alpha = LWL_inv@L@W@Y
            pred_mean = test_L@alpha
            # pred_cov =1/N2/bl[0]*l(test_X,None,al,bl)-1/N2/bl[0]*test_L@L_inv@test_L.T + test_L@LWL_inv@test_L.T/N2# +test_L@LWL_inv@test_L.T #l(test_X,None,al,bl)-test_L@L_inv@test_L.T+test_L@L_inv@C@L_inv@test_L.T
            test_err = ((pred_mean-test_G)**2).mean()
            norm = alpha.T @ L @ alpha
        Nfeval += 1
        return test_err,norm[0,0]

    funcs = {'sin':lambda x: np.sin(x),
            'step':lambda x: 0 * (x<0) +1* (x>=0),
            'abs':lambda x: np.abs(x),
            'linear':lambda x:x}
    
    k,l = Kernel('rbf'), Kernel('rbf')
    snames = ['step','sin','abs','linear']
    M = 2
    als = np.logspace(-2,1.1,32)
    bls = np.logspace(-2,0.5,32)
    al, bl = als[i], bls[j]
    # Z = np.random.uniform(-3,3,size=(6000,2))
    # cofounder = np.random.normal(0,1,size=(6000,1))
    # gamma = np.random.normal(0,0.1,size=(6000,1))
    
    # delta = np.random.normal(0,0.1,size=(4000,1))
    # X = Z[:,[0]]+cofounder+gamma
    # test_X = np.sort(X[4000:,])
    # X = X[:4000,]
    # Z = Z[:4000,]
    data = np.load(ROOT_PATH + '/our_methods/data_sin.npz')
    X,Z,test_X = data['X'], data['Z'], data['test_X']
    ak = get_median_inter_mnist(Z)
    W = (k(Z,None,ak,1)+k(Z,None,ak*10,1)+k(Z,None,ak/10,1))/Z.shape[0]**2/3
    EYEN = np.eye(Z.shape[0])
    N2 = Z.shape[0]**2
    L = l(X,None,al,bl)
    JITTER = 1e-6
    L_inv = chol_inv(L+JITTER*EYEN)
    for _ in range(seed+1):
        random_indices = np.sort(np.random.choice(range(W.shape[0]),nystr_M,replace=False))
    if nystr:
        eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices)
        C = L-L@eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
    else:
        LWL_inv = chol_inv(L@W@L+L/N2 +JITTER*EYEN)# chol_inv(W*N2+L_inv) # chol_inv(L@W@L+L/N2 +JITTER*EYEN)
        C = L@LWL_inv@L/N2
    test_L = l(test_X,X,al,bl)

    for sid in range(len(snames)-2):
        sname = snames[sid]
        data = np.load(ROOT_PATH + '/our_methods/data_{}.npz'.format(sname))
        # train, dev, test = load_data(snames[0])
        # func = funcs[sname]
        PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
        os.makedirs(PATH,exist_ok=True)
        filename = PATH+'LMO_errs_{}_{}_nystr_{}.npy'.format(i,j,seed)
        # if os.path.exists(filename):
        #     continue
        Y, test_G =  data['Y'],  data['test_G']
        permutation = np.random.permutation(X.shape[0])
        c = C@W@Y*N2
        res = LMO_err([al,bl],2)
        test_err,norm = callback0([al,bl])
        np.save(filename,[res,norm,test_err])


def plot_res_2d(sname,seed):       
    als = np.logspace(-2,1.1,32)
    bls = np.logspace(-2,0.5,32)
    PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
    # [test_LMO_err(sname,i,j) for i in range(len(als)) for j in range(len(bls)) if not os.path.exists(PATH+'LMO_errs_{}_{}.npy'.format(i,j))]
    res = np.array([np.load(PATH+'LMO_errs_{}_{}_nystr_{}.npy'.format(i,j,seed)) for i in range(len(als)) for j in range(len(bls)) if os.path.exists(PATH+'LMO_errs_{}_{}_nystr_{}.npy'.format(i,j,seed))])
    res = np.vstack(res)
    # res = res[16*32:,]
    # als = als[16:]
    output_str = ''
#    for i in range(res.shape[1]-1):
#        res[:,i] = spni.median_filter(res[:,i].reshape(len(als),len(bls)),size=3,mode='nearest').flatten()
    for i in range(res.shape[1]-2):
        res[:,i] = np.argsort(np.argsort(res[:,i]))# +np.argsort(np.argsort(res[:,-2]))
        res[:,i] = np.argsort(np.argsort(res[:,i]))                
        min_res = res[np.argsort(res[:, i])][0]
        print(np.argsort(res[:, i])[:20])
        output_str += 'star: {} \n '.format(min_res)
    min_ids = np.where(min_res==res)[0]
    # min_ids[-1] =np.argmin(res[:,-1]) 
    output_str += 'min: {} {} \n '.format(np.min(res,axis=0),np.argmin(res,axis=0))
    print(sname, output_str)
    for i in range(res.shape[1]-2):
        cv_err = res[:,i].reshape((len(als),len(bls)))# res[:,i].reshape((len(als),len(bls)))
        test_err = np.argsort(np.argsort(res[:,-1])).reshape((len(als),len(bls)))# np.argsort(np.argsort(res[:,-1])).reshape((len(als),len(bls)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
        CS = ax1.contourf(numpy.tile(bls, [len(als), 1]), numpy.tile(als, [len(bls), 1]).T, cv_err)
        ax1.contour(CS, colors='k')
        fig.colorbar(CS, ax=ax1)
        ax1.set_xlim(bls[0],bls[-1])
        ax1.set_ylim(als[0], als[-1])
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_title('cv error')
        min_al,min_bl = divmod(min_ids[i],32)
        al,bl = als[min_al],bls[min_bl]
        ax1.plot(bl,al,marker='o',color='r')
        CS = ax2.contourf(numpy.tile(bls, [len(als), 1]), numpy.tile(als, [len(bls), 1]).T, test_err)
        ax2.contour(CS, colors='k')
        fig.colorbar(CS, ax=ax2)
        ax2.set_xlim(bls[0], bls[-1])
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_title('test error')
        min_al,min_bl = divmod(min_ids[-1],32)
        al,bl = als[min_al],bls[min_bl]
        ax2.plot(bl,al,marker='o',color='r')
        plt.savefig('leave_M_{}_{}_nystr_{}.pdf'.format(sname,i,seed), bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    if len(sys.argv)==2:
        ind = int(sys.argv[1])
        seed, ind = divmod(ind,1024)
        alid, blid = divmod(ind,32)
        test_LMO_err(alid,blid,seed) 
    elif len(sys.argv)==1:
        for sname in ['step','sin']:# ['random_{}'.format(i) for i in range(4)]:# ['step','sin','abs','linear'] :#['random_{}'.format(i) for i in range(4)]:# ['step','sin','abs','linear']:
            [plot_res_2d(sname,seed) for seed in range(4)]
    else:
        inds = [int(e) for e in sys.argv[1:]]
        alids,blids = [],[]
        for ind in inds:
            alid,blid = divmod(ind,32)
            alids += [alid]
            blids += [blid]
        Parallel(n_jobs = min(len(inds),20))(delayed(test_LMO_err)(alids[i],blids[i]) for i in range(len(alids))) 
    
    
    
    
    
    
    
    
    
    
    

