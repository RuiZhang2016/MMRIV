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
from scipy.special import gamma as Gamma
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

def test_LMO_err(sname, seed=527, nystr=False):
    
    def LMO_err(params,M=2,verbal=False):
        n_params = len(params)
        # params = np.exp(params)
        al,bl,cl = params # params[:int(n_params/2)], params[int(n_params/2):] #  [np.exp(e) for e in params]
        print(al._value,bl._value,cl._value)
        L = l(X,None,al,bl)+JITTER*EYEN # l(X,None,al,bl,cl)
        # L_inv = chol_inv(L)
        if nystr:
            L_W_inv = L-L@eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
        else:
            LWL_inv = chol_inv(L@W@L+L/N2 +JITTER*EYEN)# chol_inv(W*N2+L_inv) # chol_inv(L@W@L+L/N2 +JITTER*EYEN)
            # LWL_inv = chol_inv(L@W@L*N2+L)
        C = L@LWL_inv@L/N2 # L_W_inv # L@LWL_inv@L # L_W_inv
        c = C@W@Y*N2
        c_y = c-Y
        
        lmo_err = 0
        N = 0
        EYEM = np.eye(M)
        for ii in range(4):
            permutation = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices,indices)]*N2
                C_i = C[np.ix_(indices,indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(EYEM-C_i@K_i)@c_y_i
                # print(I_CW_inv.shape,c_y_i.shape)
                lmo_err += b_y.T@K_i@c_y_i
                (_,slogdet) = np.linalg.slogdet(EYEM-C_i@K_i)
                lmo_err -= slogdet
                N += 1
        return lmo_err[0,0]/N/M**2

    def callback0(params):
        global Nfeval
        if Nfeval % 1 == 0:
            n_params = len(params)
            al,bl,cl = params # params[:int(n_params/2)], params[int(n_params/2):]
            # print('Nfeval: ', Nfeval, 'params: ', [al._value,bl._value, JITTER._value])
            L = l(X,None,al,bl)+1e-4*EYEN
            # L_inv = chol_inv(L+JITTER*EYEN)
            if nystr:
                # L_W_inv = L-L@eig_vec_K@chol_inv(eig_vec_K.T@L@eig_vec_K/N2+1/eig_val_K/N2)@eig_vec_K.T@L/N2
                pass
            else:
                LWL_inv = chol_inv(L@W@L+L/N2+JITTER*EYEN)
                # L_W_inv = chol_inv(W*N2+L_inv)
            test_L = l(test_X,X,al,bl)
            alpha = LWL_inv@L@W@Y # EYEN-eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2# LWL_inv@L@W@Y# L_inv@L_W_inv@W@Y*N2 # LWL_inv@L@W@Y
            # alpha = alpha@W@Y 
            pred_mean = test_L@alpha
            # pred_cov =1/N2/bl[0]*l(test_X,None,al,bl)-1/N2/bl[0]*test_L@L_inv@test_L.T + test_L@LWL_inv@test_L.T/N2# +test_L@LWL_inv@test_L.T #l(test_X,None,al,bl)-test_L@L_inv@test_L.T+test_L@L_inv@C@L_inv@test_L.T
            # pred_cov = l(test_X,None,al,bl)-test_L@L_inv@test_L.T+test_L@LWL_inv@test_L.T/N2
            test_err = ((pred_mean-test_G)**2).mean() # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha
            plt.plot(test_X, test_G, '.',label='true')
            plt.plot(test_X, pred_mean, '.',label='true')
            plt.legend()
            plt.savefig('tmp_step_{}.pdf'.format(Nfeval))
            plt.close('all')
        Nfeval += 1
        print(test_err,norm[0,0], (L@alpha-Y).T@W@(L@alpha-Y))
        return test_err, norm[0,0]

    funcs = {'sin':lambda x: np.sin(x),
            'step':lambda x: 0* (x<0) +1* (x>=0),
            'abs':lambda x: np.abs(x),
            'linear': lambda x: x}
    k,l = Kernel('rbf'), Kernel('rbf')#exp_sin_squared')# Kernel('rbf')
    snames = ['step','sin','abs','linear']
    M = 2
    n_train, n_test = 800,800
    Z,test_Z = np.random.uniform(-3,3,size=(n_train,2)),np.random.uniform(-3,3,size=(n_test,2))# np.random.normal(0,np.sqrt(2),size=(n_train,1))# np.random.uniform(-3,3,size=(n_train+n_test,2))
    cofounder,test_cofounder = np.random.normal(0,1,size=(n_train,1)),np.random.normal(0,1,size=(n_test,1))
    gamma,test_gamma = np.random.normal(0,np.sqrt(0.1),size=(n_train,1)),np.random.normal(0,np.sqrt(0.1),size=(n_test,1))
    delta, test_delta = np.random.normal(0,np.sqrt(0.1),size=(n_train,1)),np.random.normal(0,np.sqrt(0.1),size=(n_test,1))
    X = Z[:,[0]]+cofounder+gamma
    test_X = test_Z[:,[0]] + test_cofounder+test_gamma# np.linspace(np.percentile(X,5),np.percentile(X,95),100)[:,None]# X# X[n_train:,]# np.sort(X[n_train:,])
    EYEN = np.eye(X.shape[0])
    ak = get_median_inter_mnist(Z)
    N2 = X.shape[0]**2
    beta = (1+np.sqrt(1+4*ak*ak))/2
    alpha = beta**2/ak**2
    sqd = Kernel('sqdist')
    W = (l(Z,None,ak,1)+l(Z,None,ak/10,1)+l(Z,None,ak*10,1))/3/Z.shape[0]**2# np.sqrt(sqd(Z,None))# l(Z,None,ak,1)+l(Z,None,ak/10,1)+l(Z,None,ak*10,1) # 1/(sqd(Z,None)/2+beta)**alpha
    # Gamma((2*alpha+1)/2)/Gamma(alpha)/(2*np.pi)**0.5/(beta/lamb)**0.5*(1+sqd(Z,None)*(lamb/2/beta))**(-(2*alpha+1)/2)
    # random_indices = np.sort(np.random.choice(range(W.shape[0]),nystr_M,replace=False))
    # eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices)
    # W = eig_vec_K@np.diag(eig_val_K)@eig_vec_K.T
    permutation = np.random.permutation(X.shape[0])
    # L_inv = chol_inv(L+JITTER*EYEN)
    for i in range(len(snames)):
        sname = snames[i]
        func = funcs[sname]
    # L0 = l(X,None,1,1)
    # alpha0 = (L0@W@L0+L0)
#        coef = (np.random.rand(n_train,1)+1e-8)/n_train*2
#        func = lambda x:l(x,X,0.5,1)@coef
        Y =func(X) + cofounder+delta# func(X)+cofounder[:n_train,]+delta# L0@np.ones((n_train,1))/n_train + cofounder[:n_train,]+delta
        test_G = func(test_X)# l(test_X,X,1,1)@np.ones((n_train,1))/n_train # func(test_X)
        test_G = (test_G - Y.mean())/Y.std()
        # G = (func(X)-Y.mean())/Y.std()
        Y = (Y-Y.mean())/Y.std()
        # np.savez('data_{}'.format(sname),X=X,Z=Z,Y=Y,test_X=test_X,test_G=test_G)
        # np.savez('data_random_{}'.format(i),X=X,Z=Z,Y=Y,test_X=test_X,test_G=test_G)
        obj_grad = value_and_grad(lambda params: LMO_err(params))
    # test_err,norm = callback0([al,bl])
        params0 = [1.1,1.1,0.01]
        bounds =  [[1e-2,5],[1e-2,5],[1e-3,0.1]]
        print(sname)
        for _ in range(1):
            obj_grad = value_and_grad(lambda params: LMO_err(params))
            permutation = np.random.permutation(X.shape[0])
            res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5000,'disp':True},callback=callback0) 
            print(res)
            params0 = res.x
        # return
    # PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
    # np.save(PATH+'LMO_errs_{}_{}_nystr.npy'.format(i,j),[res,norm,test_err])


def plot_res_2d(sname):       
    als = np.logspace(-2,1,32)# np.logspace(-3,0,8)
    bls = np.logspace(-10,-4,32)
    PATH = ROOT_PATH + "/our_methods/results/zoo/" + sname + "/"
    # [test_LMO_err(sname,i,j) for i in range(len(als)) for j in range(len(bls)) if not os.path.exists(PATH+'LMO_errs_{}_{}.npy'.format(i,j))]
    res = np.array([np.load(PATH+'LMO_errs_{}_{}_nystr.npy'.format(i,j)) for i in range(len(als)) for j in range(len(bls)) if os.path.exists(PATH+'LMO_errs_{}_{}_nystr.npy'.format(i,j))])
    res = np.vstack(res)
    # res = res[16*32:,]
    # als = als[16:]
    output_str = ''
    for i in range(res.shape[1]-2):
        res[:,i] = np.argsort(np.argsort(res[:,i]))# +np.argsort(np.argsort(res[:,-2]))
        # res[:,i] = np.argsort(np.argsort(res[:,i]))                
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
        plt.savefig('leave_M_{}_{}_nystr.pdf'.format(sname,i), bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    test_LMO_err('step')
    raise Exception()
    
    if len(sys.argv)==2:
        ind = int(sys.argv[1])
        alid, blid = divmod(ind,32)
        test_LMO_err(alid,blid) 
    elif len(sys.argv)==1:
        for sname in ['step','sin','abs','linear']:
            # plot_res_2d('sin')
            # test_LMO_err(sname)
            pass
    else:
        inds = [int(e) for e in sys.argv[1:]]
        alids,blids = [],[]
        for ind in inds:
            alid,blid = divmod(ind,32)
            alids += [alid]
            blids += [blid]
        Parallel(n_jobs = min(len(inds),20))(delayed(test_LMO_err)(alids[i],blids[i]) for i in range(len(alids))) 
    
    
    
    
    
    
    
    
    
    
    

