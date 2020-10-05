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
torch.manual_seed(seed)
np.random.seed(seed)
JITTER = 1e-6
M = 512
EYEM = np.eye(M)
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



def train_cv_loss(params, l,k, train,test, nystr=False):
    Lambda, al, ak = params
    Lambda = Lambda
    X_train, Y_train, Z_train = train
    
    L_train = l(X_train,X_train,al,1)
    W_train = k(Z_train,Z_train,ak,1)/Z_train.shape[0]**2
    WY_train = np.matmul(W_train,Y_train)
    EYEN = np.eye(W_train.shape[0])
    if nystr:
        LWL = np.matmul(L_train,np.matmul(W_train,L_train))
        eig_val, eig_vec = nystrom(LWL+Lambda*L_train,np.sort(np.random.choice(range(X_train.shape[0]),M,replace=False)))
        #tmp_val, tmp_vec = np.linalg.eigh(LWL+Lambda*L_train)
        alpha = EYE - eig_vec@np.linalg.inv(JITTER*EYEM+np.diag(eig_val)@eig_vec.T@eig_vec)@np.diag(eig_val)@eig_vec.T
        # tmp_alpha = EYE - tmp_vec@np.linalg.inv(JITTER*np.eye(M) +tmp_vec.T@tmp_vec*tmp_val)@np.diag(tmp_val)@tmp_vec.T
        alpha = alpha@L_train@WY_train/JITTER
        # tmp_alpha = tmp_alpha@L_train@WY_train/JITTER
        # print(alpha)
        # print(tmp_alpha)
    else:
        alpha = np.linalg.inv(np.matmul(W_train,L_train) + Lambda * EYEN)
        alpha = np.matmul(alpha,WY_train)
    
    diff_train = Y_train - L_train @ alpha
    train_err = diff_train.T @ W_train @diff_train+Lambda*alpha.T@L_train@alpha
    if test is None:
        return train_err[0,0]

    X_test, Y_test, Z_test = test
    W_test = k(Z_test,Z_test,ak,1)/Z_test.shape[0]**2
    # W_test -= np.diag(np.diag(W_test))
    L_test = l(X_test,X_train,al,1)
    Y_pred = np.matmul(L_test,alpha)
    diff_test = Y_test - Y_pred
    test_err = np.matmul(np.matmul(diff_test.T,W_test),diff_test)
    return test_err[0,0], train_err[0,0]

def plot_bayes_zoo(seed=527, nystr=True):
    
    def neg_log_marginal(params,verbal=False):
        al,ak,bl,bk,mean = params
        al,ak,bl,bk = [np.exp(e) for e in [al,ak,bl,bk]]
        # vec = params[2:]
        # tri_W = lin2tril(vec)
        # W = tri_W@tri_W.T
        # W = (W + W.T)/2
        EYEN = np.eye(Z.shape[0])
        W = bk**2*k(Z,None,ak)
        tri_W = np.linalg.cholesky(W+1e-4*EYEN)
        tri_W_inv = splg.solve_triangular(tri_W,EYEN,lower=True)
        W = np.matmul(tri_W_inv.T,tri_W_inv)
        logdetW = 2*np.sum(np.log(np.diag(tri_W)))
        L = (bl**2+1e-4)*l(X,None,al)
        K = L+W
        # jitter = np.diag(K).mean()*JITTER
        tri_K = np.linalg.cholesky(L+W+1e-4*EYEN)
        logdetK = 2*np.sum(np.log(np.diag(tri_K)))
        tri_inv = splg.solve_triangular(tri_K,EYEN,lower=True)
        K_inv = np.matmul(tri_inv.T,tri_inv)
        alpha_sce = np.matmul(K_inv,Y-mean)
        nlm = (logdetK+np.matmul(Y.T-mean, alpha_sce)+np.log(2*np.pi)*len(Y))[0,0]/2
        # nlm = -mvn.logpdf(Y.flatten(), mean*np.ones(len(Y)), K)
        if verbal:
            # res = mvn.logpdf(Y.flatten(), np.zeros(Y.shape[0]), K)
            # print(res.shape)
            # print('verbal: logdetK np.dot(Y.T, alpha_sce)',logdetK,np.dot(Y.T, alpha_sce))
            test_X = test.x
            test_Y = test.g
            test_L = bl**2*l(X,test_X,al)
            pred_g = mean+test_L.T@alpha_sce
            # pred_g = mean +   np.dot(np.linalg.solve(K, test_L).T, Y - mean)
            plt.plot(test_X, pred_g, 'o',label='pred')
            plt.plot(test_X, test_Y, '.',label='true')
            plt.plot(train.x[:300], train.y[:300], '.',label='train')
            plt.legend()
            global Nfeval
            plt.savefig('tmp_{}.pdf'.format(Nfeval))
            plt.close('all')
            test_err = ((pred_g-test_Y)**2).mean()
            print('testerr, mean_alpha',test_err,np) 
        return nlm

    def callback0(params):
        global Nfeval
        if Nfeval % 1 == 0:
            print('Nfeval: ', Nfeval, ' al, ak, bl, bk, mean: ', params, ' nlm: ', neg_log_marginal(params,True))
            
        Nfeval += 1

    k,l = Kernel('rbf'), Kernel('rbf')
    train, dev, test = load_data('sin')
    
    # Z,X,Y = train.z[:N_train], train.x[:N_train], train.y[:N_train]
    # W = k(Z,Z,3)+JITTER*EYEN
    # tri_W = np.linalg.cholesky(W)
    # tri_indices = np.tril_indices(W.shape[0])
    # bounds = [[e/10,e*10] if e>0 else [e*10,e/10] for e in tri_W[tri_indices]]
    # nlm_grad = value_and_grad(lambda al,bl,mean: neg_log_marginal([al,ak,bl,bk,mean],verbal=False))
    # params0 = np.append([1,1],tri_W[tri_indices])
    rs = np.random.RandomState(0)
    params0 = 0.2* rs.randn(5)
    bounds = None
    batch_size = 300
    n_data = len(train.x)
    #for epoch in range(100):
    #permutation = np.random.permutation(n_data)
#        for i in range(0, n_data, batch_size):
#            indices = permutation[i:i + batch_size]
    X, Y, Z = train.x[:300], train.y[:300], train.z[:300]
    ak = get_median_inter_mnist(Z)
    W = k(Z,None,ak)
    res1 = (Y-train.g[:300]).T@W@(Y-train.g[:300])
    res2 = (Y).T@(W-np.diag(np.diag(W)))@(Y)
    print(res1,res2)
    return
#                nlm_grad = value_and_grad(lambda params: neg_log_marginal([params[0],params1[0],params[1],params1[1],params[2]],verbal=False))
    nlm_grad = value_and_grad(lambda params: neg_log_marginal(params,verbal=False))
    result = minimize(nlm_grad, x0=params0, method='L-BFGS-B',bounds=bounds,jac=True,options={'maxiter':500},callback=callback0)
    params0 = result.x
    return

def optimize_U_stat():
    k,l = Kernel('rbf'), Kernel('rbf')
    train, dev, test = load_data('sin')
    n_train = 100
    X, Y, Z,G = train.x[:n_train], train.y[:n_train], train.z[:n_train],train.g[:n_train]
    ak = get_median_inter_mnist(Z)

    W = k(Z,None,ak)*20
    def obj_fun(params,W):
        al,bl = params[0:2]
        alpha = params[2:]
        alpha = alpha[:,None]
        L = l(X,None,al)
        pred = L@alpha
        obj = (Y-pred).T@(W-np.diag(np.diag(W)))@(Y-pred)#+alpha.T@L@alpha
        obj *= obj
        obj += alpha.T@L@alpha
        return obj
    
    rs = np.random.RandomState(0)
    params0 = 0.2* rs.randn(n_train+2)
    obj_grad = value_and_grad(lambda params: obj_fun(params,W))
    result = minimize(obj_grad, x0=params0, method='L-BFGS-B',jac=True,options={'maxiter':5000,'disp':True})
    al,bl = result.x[0:2]
    alpha = result.x[2:]
    alpha = alpha[:,None]
    L = l(X,None,al)
    pred = L@alpha
    print(((G-pred)**2).mean())
    print(((Y-pred)**2).mean())
    plt.plot(X,G,'.',label='True')
    plt.plot(X,pred,'.',label='Pred')
    plt.legend()
    plt.savefig('tmp_opt.pdf')
    plt.close('all')
    

Nfeval = 1


if __name__ == '__main__':
    optimize_U_stat()
