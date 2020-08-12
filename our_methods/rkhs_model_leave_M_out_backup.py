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
JITTER = 1e-6
# M = 512
# EYEM = np.eye(M)
__sparse_fmt = csr_matrix


def nystrom(G,ind):
    Gnm = G[:,ind]
    sub_G = (Gnm)[ind,:]

    eig_val, eig_vec = numpy.linalg.eigh(sub_G+JITTER*EYEM)
    eig_vec = np.sqrt(len(ind) / G.shape[0]) * np.matmul(Gnm, eig_vec)/eig_val
    eig_val /= len(ind) / G.shape[0]
    return eig_val, eig_vec

def memodict(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


@primitive
def matmul_ad(a, b, c):
    """ for packing and unpacking parameters"""
    return b @ a

def make_grad_matmul_ad(ans, a, b, c):
    def gradient_product(g):
        return c @ g
    return gradient_product

matmul_ad.defgrad(make_grad_matmul_ad, 0)

__sparse = True

if __sparse:
    matmul = matmul_ad
else:
    matmul = lambda a, b, c: b @ a

@memodict
def lin2triltfm(m):
    """ tfm and tfm_t is a matrix recording the positions of elements below the main diagonal"""
    n = int(m * (m+1) / 2)
    tfm = np.zeros((m*m, n))
    i = np.where(np.tril(np.ones((m,m))).flatten())[0]
    tfm[i, range(len(i))] = 1
    tfm_t = tfm.T
    if __sparse:
        tfm = __sparse_fmt(tfm)
        tfm_t = __sparse_fmt(tfm_t)
    return tfm, tfm_t

def lin2tril(x):
    """ a horizontal vector to a triangular matrix"""
    """ for unpacking parameters"""
    n = len(x.flatten())
    m = int(1/2 * (-1 + np.sqrt(1 + 8 * n)))
    assert m * (m+1) / 2 == n, (m, n, x.shape)
    tfm, tfm_t = lin2triltfm(m)
    return (matmul(x, tfm, tfm_t)).reshape(m,m)


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

def test_LMO_err(i,j, seed=527, nystr=True):
    
    def LMO_err(params,M=2,verbal=False):
        n_params = len(params)
        # params = np.exp(params)
        al,bl = params[:int(n_params/2)], params[int(n_params/2):] #  [np.exp(e) for e in params]
        permutation = np.random.permutation(X.shape[0])
        c_y = c-Y
        
        lmo_err = 0
        N = 0
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
        global Nfeval
        if Nfeval % 1 == 0:
            #  print('Nfeval: ', Nfeval, ' al, bl, mean: ', params, ' nlm: ', LMO_err(params,M))
            n_params = len(params)
            al,bl = params[:int(n_params/2)], params[int(n_params/2):]
            alpha = LWL_inv@L@W@Y 
            pred_mean = test_L@alpha
            pred_cov =1/N2/bl[0]*l(test_X,None,al,bl)-1/N2/bl[0]*test_L@L_inv@test_L.T + test_L@LWL_inv@test_L.T/N2# +test_L@LWL_inv@test_L.T #l(test_X,None,al,bl)-test_L@L_inv@test_L.T+test_L@L_inv@C@L_inv@test_L.T
            test_err = ((pred_mean-test_G)**2).mean()
            norm = alpha.T @ L @ alpha
            print('params ',params,' ',test_err)
            fig = plt.figure(figsize=(12,8), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            ax.plot(test_X,test_G,'.',label='true')
            ax.plot(test_X,pred_mean,'.',label='pred')
            marg_std = np.sqrt(np.diag(pred_cov))
            ax.fill(np.concatenate([test_X, test_X[::-1]]),np.concatenate([pred_mean - 1.96 * marg_std,(pred_mean + 1.96 * marg_std)[::-1]]),alpha=.15, fc='Blue', ec='None')
            plt.legend()
            plt.savefig(ROOT_PATH + '/our_methods/leave_{}_m_err_{}_{}.pdf'.format(sname,i,j))
            plt.close('all')
        Nfeval += 1
        return test_err,norm[0,0]

    funcs = {'sin':lambda x: np.sin(x),
            'step':lambda x: 0 * (x<0) +1* (x>=0),
            'abs':lambda x: np.abs(x),
            'linear': lambda x: x}
    k,l = Kernel('rbf'), Kernel('rbf')
    snames = ['step','sin','abs','linear']
    M = 2
    als =np.logspace(-2,1,32)
    bls = np.logspace(-10,-4,32)
    al, bl = als[i], bls[j]
    Z = np.random.uniform(-3,3,size=(6000,2))
    cofounder = np.random.normal(0,1,size=(6000,1))
    gamma = np.random.normal(0,0.1,size=(6000,1))
    
    delta = np.random.normal(0,0.1,size=(4000,1))
    X = 0.5*Z[:,[0]]+0.5*cofounder+gamma
    test_X = np.sort(X[4000:,])
    X = X[:4000,]
    EYEN = np.eye(X.shape[0])
    Z = Z[:4000,]
    ak = get_median_inter_mnist(Z)
    W = k(Z,None,ak,1)/Z.shape[0]**2
    N2 = X.shape[0]**2
    permutation = np.random.permutation(X.shape[0])
    L = l(X,None,al,1)
    L_inv = chol_inv(L+JITTER*EYEN)
    LWL_inv = chol_inv(L@W@L+bl*L+JITTER*EYEN)
    C =1/N2*L@LWL_inv@L
    test_L = l(test_X,X,al,1)

    for sid in range(len(snames)):
        sname = snames[sid]
        
        func = funcs[sname]
        Y = func(X) + cofounder[:4000,]+delta
        test_G = func(test_X)
        c = C@W@Y*N2
        res = LMO_err([al,bl],2)
        test_err,norm = callback0([al,bl])
        PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
        np.save(PATH+'LMO_errs_{}_{}.npy'.format(i,j),[res,norm,test_err])


def plot_res_2d(sname):       
    als =np.logspace(-2,1,32)# np.logspace(-3,0,8)
    bls = np.logspace(-10,-4,32)
    PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
    # [test_LMO_err(sname,i,j) for i in range(len(als)) for j in range(len(bls)) if not os.path.exists(PATH+'LMO_errs_{}_{}.npy'.format(i,j))]
    res = np.array([np.load(PATH+'LMO_errs_{}_{}.npy'.format(i,j)) for i in range(len(als)) for j in range(len(bls)) if os.path.exists(PATH+'LMO_errs_{}_{}.npy'.format(i,j))])
    res = np.vstack(res)
    res = res[16*32:,]
    als = als[16:]
    output_str = ''
    for i in range(res.shape[1]-2):
        res[:,i] = np.argsort(np.argsort(res[:,i]))+np.argsort(np.argsort(res[:,-2]))
        res[:,i] = np.argsort(np.argsort(res[:,i]))                
        min_res = res[np.argsort(res[:, i])][0]
        output_str += 'star: {} \n '.format(min_res)
    min_ids = np.argmin(res,axis=0)
    output_str += 'min: {} {} \n '.format(np.min(res,axis=0), min_ids)
    print(output_str)

    for i in range(res.shape[1]-2):
        cv_err = res[:,i].reshape((-1,len(bls)))# res[:,i].reshape((len(als),len(bls)))
        test_err = np.argsort(np.argsort(res[:,-1])).reshape((-1,len(bls)))# np.argsort(np.argsort(res[:,-1])).reshape((len(als),len(bls)))
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
        plt.savefig('leave_M_{}_{}.pdf'.format(sname,i), bbox_inches='tight')
        plt.close('all')





if __name__ == '__main__':
    if len(sys.argv)==2:
        ind = int(sys.argv[1])
        alid, blid = divmod(ind,32)
        test_LMO_err(alid,blid) 
    elif len(sys.argv)==1:
        for sname in ['step','sin','abs','linear']:
            plot_res_2d(sname)
    else:
        inds = [int(e) for e in sys.argv[1:]]
        alids,blids = [],[]
        for ind in inds:
            alid,blid = divmod(ind,32)
            alids += [alid]
            blids += [blid]
        Parallel(n_jobs = min(len(inds),20))(delayed(test_LMO_err)(alids[i],blids[i]) for i in range(len(alids))) 
    
    
    
    
    
    
    
    
    
    
    

