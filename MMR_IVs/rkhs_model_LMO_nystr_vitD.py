import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH,jitchol,_sqdist,remove_outliers, nystrom_decomp, chol_inv
import time
import rpy2.robjects as robjects
import matplotlib.pyplot as plt

Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None


def experiment(nystr=True,IV=True):
    
    def LMO_err(params,M=2):
        params = np.exp(params)
        al,bl = params[:-1], params[-1]
        L = bl*bl*np.exp(-L0[0]/al[0]/al[0]/2)+bl*bl*np.exp(-L0[1]/al[1]/al[1]/2) +1e-6*EYEN # l(X,None,al,bl)# +1e-6*EYEN
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
                b_y = np.linalg.inv(np.eye(C_i.shape[0])-C_i@K_i)@c_y_i
                # print(I_CW_inv.shape,c_y_i.shape)
                lmo_err += b_y.T@K_i@b_y
                N += 1
        return lmo_err[0,0]/N/M**2

    def callback0(params, timer=None):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval % 1 == 0:
            params = np.exp(params)
            al,bl = params[:-1], params[-1]
            L = bl*bl*np.exp(-L0[0]/al[0]/al[0]/2)+bl*bl*np.exp(-L0[1]/al[1]/al[1]/2) +1e-6*EYEN
            if nystr:
                alpha = EYEN-eig_vec_K@np.linalg.inv(eig_vec_K.T@L@eig_vec_K/N2+np.diag(1/eig_val_K/N2))@eig_vec_K.T@L/N2
                alpha = alpha@W_nystr@Y*N2
            else:
                LWL_inv = chol_inv(L@W@L+L/N2+JITTER*EYEN)
                alpha = LWL_inv@L@W@Y
            pred_mean = L@alpha
            if timer:
                return
            norm = alpha.T @ L @ alpha

        Nfeval += 1
        if prev_norm is not None:
            if norm[0,0]/prev_norm >=3:
                if opt_params is None:
                    opt_params = params
                    opt_test_err = ((pred_mean-Y)**2).mean()
                print(True,opt_params, opt_test_err,prev_norm)
                raise Exception

        if prev_norm is None or norm[0,0]<= prev_norm:
            prev_norm = norm[0,0]
        opt_params = params
        opt_test_err = ((pred_mean-Y)**2).mean()
        print('params,test_err, norm:',opt_params, opt_test_err, prev_norm)
       
        ages =np.linspace(min(X[:,0])-abs(min(X[:,0]))*0.05,max(X[:,0])+abs(max(X[:,0]))*0.05,32)
        vitd = np.linspace(min(X[:,1])-abs(min(X[:,1]))*0.05,max(X[:,1])+abs(max(X[:,1]))*0.05,64)

        X_mesh, Y_mesh = np.meshgrid(ages, vitd)
        table = bl**2*np.hstack([np.exp(-_sqdist(X_mesh[:,[i]],X[:,[0]])/al[0]**2/2-_sqdist(Y_mesh[:,[i]],X[:,[1]])/al[1]**2/2)@alpha for i in range(X_mesh.shape[1])])
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Generate a contour plot
        cpf = ax.contourf(X_mesh,Y_mesh, table)
        # cp = ax.contour(X_mesh, Y_mesh, table)
        plt.colorbar(cpf,ax=ax)
        plt.xlabel('Age',fontsize=12)
        plt.ylabel('Vitamin D',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if IV:
            plt.savefig('VitD_IV.pdf',bbox_inches='tight')
        else:
            plt.savefig('VitD.pdf',bbox_inches='tight')
        plt.close('all')

    robjects.r['load'](ROOT_PATH+"/data/VitD.RData")
    data = np.array(robjects.r['VitD']).T
    
    # plot data
    fig = plt.figure()
    plt.scatter((data[:,0])[data[:,4]>0],(data[:,2])[data[:,4]>0],marker='s',s=3,c='r',label='dead')
    plt.scatter((data[:,0])[data[:,4]==0],(data[:,2])[data[:,4]==0],marker='o',s=1,c='b',label='alive')
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    plt.xlabel('Age',fontsize=12)
    plt.ylabel('Vitamin D',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('VitD_data.pdf',bbox_inches='tight')
    plt.close('all')

    for i in range(data.shape[1]):
        data[:,i] = (data[:,i]-data[:,i].mean())/data[:,i].std()
    Y = data[:,[4]]
    X = data[:,[0,2]]
    Z = data[:,[0,1]]
    t0 = time.time()
    EYEN = np.eye(X.shape[0])
    N2 = X.shape[0]**2
    if IV:
        ak = get_median_inter_mnist(Z)
        W0 = _sqdist(Z,None)
        W =  (np.exp(-W0/ak/ak/2)+np.exp(-W0/ak/ak/200)+np.exp(-W0/ak/ak*50))/3/N2
        del W0
    else:
        W = EYEN/N2
    L0 = np.array([_sqdist(X[:,[i]],None) for i in range(X.shape[1])])
    params0 =np.random.randn(3)/10
    bounds =  None # [[0.01,10],[0.01,5]]
    if nystr:
        for _ in range(seed+1):
            random_indices = np.sort(np.random.choice(range(W.shape[0]),nystr_M,replace=False))
        eig_val_K,eig_vec_K = nystrom_decomp(W*N2, random_indices)
        inv_eig_val_K = np.diag(1/eig_val_K/N2)
        W_nystr = eig_vec_K @ np.diag(eig_val_K)@eig_vec_K.T/N2
        W_nystr_Y = W_nystr@Y
    
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    res = minimize(obj_grad, x0=params0,bounds=bounds, method='L-BFGS-B',jac=True,options={'maxiter':5000}, callback=callback0)    


if __name__ == '__main__':
    experiment(IV=True)
    experiment(IV=False)
    
    
    
    
    
    
    
    
    
    
    

