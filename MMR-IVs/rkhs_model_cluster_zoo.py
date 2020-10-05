import add_path,os,sys
import numpy,scipy
import torch
import torch.optim as optim
import autograd.numpy as np
from scenarios.abstract_scenario import AbstractScenario
import matplotlib.pyplot as plt
from autograd import value_and_grad
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from util import get_median_inter,get_median_inter_mnist, Kernel, load_data, ROOT_PATH
from joblib import Parallel,delayed
from early_stopping import EarlyStopping
Nfeval = 1
seed = 527
torch.manual_seed(seed)
np.random.seed(seed)
JITTER = 1e-6

def nystrom(G,ind):
    Gnm = G[:,ind]
    sub_G = (Gnm)[ind,:]
    M = len(ind)
    EYEM = np.eye(M)
    eig_val, eig_vec = numpy.linalg.eigh(sub_G+JITTER*EYEM)
    eig_vec = np.sqrt(M / G.shape[0]) * np.matmul(Gnm, eig_vec)/eig_val
    eig_val /= M / G.shape[0]
    return eig_val, eig_vec


def compute_alpha(params,l,k,X,Z,indices, nystr=False):
    Lambda, al, ak = params
    L = l(X,X,al)
    W = k(Z,Z,ak)/Z.shape[0]**2
    EYEN = np.eye(W.shape[0])
    EYEM = np.eye(len(indices))
    if nystr:
        LWL = np.matmul(L,np.matmul(W,L))
        eig_val, eig_vec = nystrom(LWL+Lambda*L,indices)
        alpha = np.diag(eig_val)@eig_vec.T
        alpha = EYEN - eig_vec@np.linalg.inv(JITTER*EYEM+alpha@eig_vec)@alpha
        alpha = np.matmul(alpha,np.matmul(L,W))/JITTER
    else:
        alpha = np.linalg.inv(np.matmul(W,L) + Lambda * EYEN)
        alpha = np.matmul(alpha,W)
    return alpha,L,W

def R_risk(alpha,Y,L,W,Lambda=0,train=True, nystr=False):
    diff =  Y - L@alpha
    R = diff.T@W@diff
    if train:
        R +=  Lambda*alpha.T@L@alpha
    return R[0,0]

def run_experiment_rkhs_2(i,j, M,nystr=True):
    # load data
    scenario_list = ["step", "sin", "abs", "linear"]
    X, Ys, Z,Gs =  None,[],None,[]

    for s in scenario_list:
        train, dev, test = load_data(s)
        if X is None:
            X = np.vstack((train.x,dev.x))
            Z = np.vstack((train.z,dev.z))
        Ys += [np.vstack((train.y,dev.y))]
        Gs += [test.g]
    folder = ROOT_PATH + "/our_methods/results/zoo/joint/"
    os.makedirs(folder, exist_ok=True)
    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    ak = get_median_inter_mnist(Z)

    al0 = get_median_inter_mnist(X)
    als = np.linspace(al0/50,al0*3,16)
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)
    
    Lambdas = np.logspace(-10,-6,16)
    params = [Lambdas[i], als[j], ak]
    test_L = l(test.x,X,als[j])
    cv_errs= []
    for rep in range(20):
        rs = np.zeros(0)
        terr = np.zeros(0)
        for train_index, test_index in kf.split(X):
            indices = np.sort(np.random.choice(range(len(train_index)),M,replace=False))
            alpha,L_train,W_train = compute_alpha(params,l,k,X[train_index],Z[train_index],indices, nystr=True)
            L_test = l(test.x,X[test_index],als[j])
            W_test = k(Z[test_index],None,ak)/len(test_index)**2
            # r1s = [R_risk(alpha@Y[train_index],Y[train_index],L_train,W_train,Lambda=Lambda,train=True, nystr=False) for Y in Ys]
            rs += np.array([R_risk(alpha@Y[test_index],Y[test_index],L_test,W_test,Lambda=0,train=False, nystr=False) for Y in Ys])
        cv_errs += [rs]
    np.save(folder+'cv_err_{}_{}_{}.npy'.format(i,j,M),cv_errs)



def cv_errs_merge(sce_id,Mi,nystr=False):
    sce = scenario_list[sce_id]
    train, dev, test = load_data(sce)
    folder = ROOT_PATH + "/our_methods/results/zoo/joint/"
    k, l = Kernel('rbf'), Kernel('rbf')
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    train_tensor = [train.x,train.y,train.z]
    dev_tensor = [dev.x,dev.y,dev.z]
    ak = get_median_inter_mnist(Z)

    al0 = get_median_inter(X)
    als = np.linspace(al0/50,al0*3,16)
    Lambdas = np.logspace(-10,-6,16)
    Ms= [32,64,128,256,512,1024,2024]
    M = Ms[Mi]
    cv_errs = np.array([np.load(folder+'cv_err_{}_{}_{}.npy'.format(i,j,M))[:,sce_id] for i in range(len(Lambdas)) for j in range(len(als))])
    W = k(Z, Z, ak)/Z.shape[0]**2
    WY = np.matmul(W, Y)
    EYEN = np.eye(W.shape[0])
    EYEM = np.eye(M)
    errs = []
    def loop(col):
        cv_err = cv_errs[:,col]
        optimid = np.argmin(cv_err)
        i,j = divmod(optimid,len(als))
        Lambda = Lambdas[i]
        al = als[j]
        test_L = l(test.x, X, al)
        L = l(X,X,al)
        LWL = np.matmul(L,np.matmul(W,L))
        if nystr:
            eig_val, eig_vec = nystrom(LWL+Lambda*L,np.sort(np.random.choice(range(X.shape[0]),M,replace=False)))
            mat = np.diag(eig_val)@eig_vec.T
            alpha = EYEN - eig_vec@np.linalg.inv(JITTER*EYEM +mat@eig_vec)@mat
            alpha = alpha@L@WY/JITTER
        else:
            alpha = np.linalg.inv(np.matmul(W, L) + Lambda * EYEN)
            alpha = np.matmul(alpha,WY)
    
        g_pred = test_L @ alpha
        err = ((g_pred - test.g) ** 2).mean()
        return err
    errs = Parallel(n_jobs=10)(delayed(loop)(col) for col in range(10))
    print(sce, ' ', M, ' test_err ', np.mean(errs),' ',np.std(errs))

def plot_cv(scenario_name,seed=527):
    # load data
    train, dev, test = load_data(scenario_name)

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    ak = get_median_inter(Z)
    al0 = get_median_inter(X)
    # al = get_median_inter(X)
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)
    W = k(Z, Z, ak, 1)/Z.shape[0]**2

    def loss(params):
        Lambda, al = params
        Lambda = np.exp(Lambda)
        error_test = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            Z_train, Z_test = Z[train_index], Z[test_index]

            L_train = l(X_train, X_train, al)
            W_train = k(Z_train, Z_train, ak)/Z_train.shape[0]**2
            WY_train = np.matmul(W_train,Y_train)
            alpha = np.linalg.inv(np.matmul(W_train,L_train) + Lambda * np.eye(W_train.shape[0]))
            alpha = np.matmul(alpha,WY_train)

            W_test = k(Z_test, Z_test, ak)/Z_test.shape[0]**2
            W_test -= np.diag(np.diag(W_test))
            L_test = l(X_test, X_train, al)
            Y_pred = np.matmul(L_test, alpha)
            diff_test = Y_test - Y_pred
            error_test += np.matmul(np.matmul(diff_test.T,W_test),diff_test)
        return error_test

    Lambda_list = numpy.linspace(-10,-5,32)
    al_list = np.linspace(al0[0,0]/3,al0[0,0]*3,10)
    WY = np.matmul(W, Y)
    EYE = np.eye(W.shape[0])

    def loop(i):
        al = al_list[i]
        L = l(X, X, al)
        WL = np.matmul(W, L)
        test_L = l(test.x, X, al)
        length =len(Lambda_list) 
        err_list = numpy.zeros(length)
        loss_list = numpy.zeros(length)
        for j in range(length):
            Lambda = Lambda_list[j]
            alpha = np.matmul(np.linalg.inv(WL+ np.exp(Lambda) *EYE),WY)

            # predictions
            g_pred = np.matmul(test_L, alpha)
            err = ((g_pred - test.g) ** 2).mean()
            err_list[j] = err
            loss_list[j] = loss([Lambda,al])
        return err_list, loss_list
    
    result = Parallel(n_jobs=30)(delayed(loop)(i) for i in range(len(al_list)))
    err_list = np.array([e[0] for e in result])
    loss_list = np.array([e[1] for e in result])
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,8),sharey=True)
    CS = ax1.contourf(numpy.tile(Lambda_list,[len(al_list),1]),numpy.tile(al_list,[len(Lambda_list),1]).T,err_list)
    fig.colorbar(CS,ax=ax1)
    ax1.set_xlim(Lambda_list[0],Lambda_list[-1])
    ax1.set_ylim(al_list[0], al_list[-1])
    # ax1.set_yscale('log')

    CS = ax2.contourf(numpy.tile(Lambda_list,[len(al_list),1]),numpy.tile(al_list,[len(Lambda_list),1]).T,loss_list)
    fig.colorbar(CS, ax=ax2)
    ax2.set_xlim(Lambda_list[0], Lambda_list[-1])
    plt.savefig('{}_cvu_{}.pdf'.format(scenario_name,ak),bbox_inches='tight')
    plt.close('all')
    np.save('err_{}_cvu_{}.npy'.format(scenario_name,ak),err_list)
    np.save('loss_{}_cvu_{}.npy'.format(scenario_name,ak),loss_list)

Lambda_list = numpy.logspace(-11, -7, 16) # [-7 -4] last 2 scenario

def plot_bayes_zoo(i,j,ii, seed=527, nystr=True):
    if 1==0:
        k, l = Kernel('rbf'), Kernel('rbf')
        train, dev, test = load_data("sin")
        Z,X = np.vstack((train.z,dev.z)), np.vstack((train.x,dev.x))
        ak0 = get_median_inter_mnist(Z)
        ak_list = [ak0] #numpy.logspace(np.log10(ak0/1e2),np.log10(ak0*10),16)
        al0 = get_median_inter_mnist(X)
        al_list = numpy.logspace(np.log10(al0/100),np.log10(10),16)
        def mat_i(i):
            ak = ak_list[i]
            # W = (k(Z,Z,ak)+JITTER*np.eye(X.shape[0]))/Z.shape[0]**2
            W = k(Z,Z,ak,1)/Z.shape[0]**2
            (_,logdetW) = np.linalg.slogdet(W)
            np.savez('../zoo_precomp/W_{}'.format(i), W=W, logdetW=logdetW)
            al = al_list[i]
            # L = l(X,X,al)+JITTER*np.eye(X.shape[0])
            L = l(X,X,al,1)
            (_,logdetL) = np.linalg.slogdet(L)
            print('logdetW',logdetW,'logdetL', logdetL)
            np.savez('../zoo_precomp/L_{}'.format(i), L=L,logdetL=logdetL)
            test_L = l(test.x, X, al,1)
            np.savez('../zoo_precomp/test_L_{}'.format(i), test_L=test_L)
        Parallel(n_jobs=16)(delayed(mat_i)(i) for i in range(len(ak_list)))
        print('done')
        assert 1 ==0 
    def neg_log_marginal(logdetLWL,Lambda,alpha,Y):
        lm1 = (-np.log(Lambda)*W.shape[0]-logdetL-logdetW+logdetLWL +Lambda*np.matmul(Y.T, alpha))[0,0]
        lm2 = lm1+logdetW
        return lm1, lm2
    
    done_flag = True
    scenario_list = ["step", "sin", "abs", "linear"]
    for sce in scenario_list:
        # output destination
        folder = ROOT_PATH + "/our_methods/results/zoo/" + sce + "/"
        path = folder+'bayes_{}_{}_{}_m.npy'.format(i,j,ii)
        if os.path.exists(path):
            pass
        else:
            done_flag = False
            break

    if done_flag:
        return

    # loading pre-compute constants
    W_info = np.load(ROOT_PATH+'/zoo_precomp/W_{}.npz'.format(ii))
    W = W_info['W']
    logdetW = W_info['logdetW']
    del W_info
    EYEN = np.eye(W.shape[0])

    test_L = np.load(ROOT_PATH+'/zoo_precomp/test_L_{}.npz'.format(i))['test_L']
    L_info = np.load(ROOT_PATH+'/zoo_precomp/L_{}.npz'.format(i))
    L, logdetL = L_info['L'],L_info['logdetL']
    del L_info
    Lambda = Lambda_list[j]
    LWL = np.matmul(L,np.matmul(W, L)+Lambda*EYEN)
    tri_mat, lower = scipy.linalg.cho_factor(LWL+JITTER*EYEN,lower=True)
    LWL_inv = scipy.linalg.cho_solve((tri_mat,lower),EYEN)
    del EYEN
    alpha = np.matmul(LWL_inv,np.matmul(L, W))
    del LWL_inv
    logdetLWL = 2*np.sum(np.log(np.diag(tri_mat)))
    del LWL,tri_mat
    # logdetLW = logdetL-logdetLWL
    for sce in scenario_list:
        # output destination
        folder = ROOT_PATH + "/our_methods/results/zoo/" + sce + "/"
        path = folder+'bayes_{}_{}_{}.npy'.format(i,j,ii)
        if os.path.exists(path):
            continue
        os.makedirs(folder, exist_ok=True)
        
        # load data
        print("\nLoading " + sce + "...")
        train, dev, test = load_data(sce)
        Y = np.vstack((train.y, dev.y))
        alpha_sce = np.matmul(alpha,Y)
        loss1, loss2 = neg_log_marginal(logdetLWL,Lambda,alpha_sce,Y)
        g_pred = np.matmul(test_L, alpha_sce)
        err_ij = ((g_pred - test.g) ** 2).mean()
        err2_ij = ((g_pred - test.y) ** 2).mean()
        np.save(path,[err_ij,loss1,loss2,err2_ij])
    return

    def loop(i,j,M):
        print("\nLoading " + scenario_name + "...")
        Lambda = Lambda_list[j]
        test_L = np.load('../tmp/test_L_{}.npz'.format(i))['arr_0']
        L = np.load('../tmp/L_{}.npz'.format(i))['arr_0']
        if nystr:
            EYEM = np.eye(M)
            random_indices = np.sort(np.random.choice(range(W.shape[0]),M,replace=False))
            LWL = np.matmul(L,np.matmul(W,L))
            LWY  = np.matmul(L,WY)/JITTER
            eig_val, eig_vec = nystrom(LWL+Lambda*L,random_indices)
            tmp = np.matmul(np.diag(eig_val),eig_vec.T)
            tmp = np.matmul(np.linalg.inv(JITTER*EYEM +np.matmul(tmp,eig_vec)),tmp)
            alpha = EYEN - np.matmul(eig_vec,tmp)
            alpha = np.matmul(alpha,LWY)
        else:
            WL = np.matmul(W, L) + Lambda*EYEN
            alpha = np.linalg.inv(WL)
            alpha = np.matmul(alpha,WY)
            loss1, loss2 = neg_log_marginal(WL,Lambda,alpha)
        g_pred = np.matmul(test_L, alpha)
        err_ij = ((g_pred - test.g) ** 2).mean()
        return err_ij, loss1, loss2
    res = loop(j)
    np.save(folder+'bayes_{}_{}_{}.npy'.format(i,j,ii),res)



def plot_bayes_merge(scenario_name,jj):
    folder = ROOT_PATH + "/our_methods/results/zoo/" + scenario_name + "/"
    files = os.listdir(folder) 
    train, dev, test = load_data(scenario_name)
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    ak0 = get_median_inter_mnist(Z)
    ak_list = [ak0]#numpy.logspace(np.log10(ak0/1e2),np.log10(ak0*10),16)
    al0 = get_median_inter_mnist(X)
    al_list = numpy.logspace(np.log10(al0/1e2),np.log10(al0*10),16)
    def loop(ii):
        [plot_bayes_zoo(i,j,ii, nystr=False) for i in range(len(al_list)) for j in range(len(Lambda_list)) if not os.path.exists(folder+'bayes_{}_{}_{}.npy'.format(i,j,ii))]
        res = np.array([np.load(folder+'bayes_{}_{}_{}.npy'.format(i,j,ii)) for i in range(len(al_list)) for j in range(len(Lambda_list))])
        min_ids = np.argmin(res,axis=0)
        output_str = 'min: {} {} \n '.format(np.min(res,axis=0), min_ids)
        min_res = res[np.argsort(res[:, jj])][0]
        output_str += 'star: {} \n '.format(min_res)
        res[:,0] = np.argsort(np.argsort(res[:, 0]))
        errs = (res[:,0]).reshape((len(al_list),len(Lambda_list)))
        res[:,jj] = np.argsort(np.argsort(res[:, jj])) # (res[:,jj] -np.min(res[:,jj],axis=0))/(np.max(res[:,jj],axis=0)-np.min(res[:,jj],axis=0))
        loss = (res[:,jj]).reshape((len(al_list),len(Lambda_list)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
        CS = ax1.contourf(numpy.tile(Lambda_list, [len(al_list), 1]), numpy.tile(al_list, [len(Lambda_list), 1]).T,
                                  errs)
        ax1.contour(CS, colors='k')
        fig.colorbar(CS, ax=ax1)
        ax1.set_xlim(Lambda_list[0], Lambda_list[-1])
        ax1.set_ylim(al_list[0], al_list[-1])
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_title('test error')
    
        CS = ax2.contourf(numpy.tile(Lambda_list, [len(al_list), 1]), numpy.tile(al_list, [len(Lambda_list), 1]).T,
                                                              loss)
        ax2.contour(CS, colors='k')
        fig.colorbar(CS, ax=ax2)
        ax2.set_xlim(Lambda_list[0], Lambda_list[-1])
        ax2.set_xscale('log')
        ax2.set_title('negative log marginal likelihood')
        plt.savefig('{}_bayes_{}_{}.pdf'.format(scenario_name,ii,jj), bbox_inches='tight')
        plt.close('all')
        return output_str, min_res
    outputs = loop(0)# Parallel(n_jobs=16)(delayed(loop)(ii) for ii in range(0))
    out_strs = outputs[0] #''.join([e[0] for e in outputs])
    print(out_strs)
    return
    plt.plot(ak_list, [e[1][0] for e in outputs])
    plt.savefig('{}_min_{}.pdf'.format(scenario_name,jj),bbox_inches='tight')
    plt.close('all')


scenario_list = ["step", "sin", "abs", "linear"]

if __name__ == '__main__':
    # Ms = [32,64,128,256,512,1024,2024]
    # for sid in range(4):
    #    for Mi in range(6):
    #        cv_errs_merge(sid,Mi,True)# Parallel(n_jobs=20)(delayed(cv_errs_merge)(sce_id,Mi,True) for sce_id in range(4) for Mi in range(6))
    # assert 1 == 0
    #ind = int(sys.argv[1])
    #Mi, ind = divmod(ind,256)
    #i,j= divmod(ind,16)
    #run_experiment_rkhs_2(i,j,Ms[Mi],nystr=True)
    #assert 1 == 0
    # scenario_list = ["mnist_x","mnist_z","mnist_xz"]
    # scenario_id = int(sys.argv[1])
    # scenario = scenario_list[scenario_id]
    # Parallel(n_jobs=20)(delayed(run_experiment_rkhs_2)(s,rep) for s in scenario_list for rep in range(1))
    # plot_bayes_tmp("step",0,0)
    # a = 0/0
    if len(sys.argv)==1:
        for s in scenario_list[:1]:
            print(s)
            try:
                plot_bayes_merge(s,1)
            except Exception as e:
                print(e)
                pass
    elif len(sys.argv) == 2:
        # strs = '107 245 15 535 196 4 91 192 273 105 230 42 12 98 18 732 702 270 272 108 558 560 278 20 276 99 511 95 557 290 275 178 8 7 532 17 23 289 556 22 549 545 103 14 197 2 537 13 106 280 104 500 277 184 279 194 161 193 274 559 195 24 191 271 187 29 564 0 16 28 109'
        # strs = [int(e) for e in strs.split()]
        ind = int(sys.argv[1])
        # ind = strs[ind]
        #akid, ind = divmod(ind,256)
        alid, Lambdaid = divmod(ind,16)
        plot_bayes_zoo(alid,Lambdaid,0, nystr=False)
        #for i in range(akid*4,(akid+1)*4):
        #    plot_bayes_zoo(alid,Lambdaid,i, nystr=False)
    else:
        inds = [int(e) for e in sys.argv[1:]]
        akids = []
        alids = []
        Lambdaids = []
        for ind in inds:
            akid,r = divmod(ind,256)
            akids += [akid]
            alid, Lambdaid = divmod(r,16)
            alids += [alid]
            Lambdaids += [Lambdaid]

        Parallel(n_jobs = 20)(delayed(plot_bayes_zoo)(alids[i],Lambdaids[i],akid, nystr=False) for i in range(len(inds)) for akid in range(akids[i]*4,(akids[i]+1)*4))
