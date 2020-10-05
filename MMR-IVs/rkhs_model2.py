import add_path,os,sys
import torch,numpy
import torch.optim as optim
import autograd.numpy as np
from scenarios.abstract_scenario import AbstractScenario
import matplotlib.pyplot as plt
from autograd import value_and_grad
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from nn_model import get_median_inter
from joblib import Parallel,delayed
from early_stopping import EarlyStopping

Nfeval = 1
seed = 527
torch.manual_seed(seed)
np.random.seed(seed)
JITTER = 1e-6
M = 512
def Kernel(name):
    def poly(x,c,d):
        return (np.matmul(x,x.T)+c*c)**d

    def rbf(x,y,a,b):
        x,y = x/a, y/a
        x2,y2 = np.sum(x*x,axis=1,keepdims=True),np.sum(y*y,axis=1,keepdims=True)
        sqdist = x2+y2.T-2*np.matmul(x,y.T)
        out = b*b*np.exp(-sqdist)
        return out

    def laplace(x,a):
        return 0

    def quad(x,y,a,b):
        x, y = x/a, y/a
        x2, y2 = np.sum(x * x, axis=1, keepdims=True), np.sum(y * y, axis=1, keepdims=True)
        sqdist = x2 + y2.T - 2 * np.matmul(x,y.T)
        out = (sqdist+1)**(-b)
        return out

    assert isinstance(name,str), 'name should be a string'
    kernel_dict = {'rbf':rbf,'poly':poly,'quad':quad}
    return kernel_dict[name]

def load_data(scenario_name):
    # load data
    # print("\nLoading " + scenario_name + "...")
    if 'mnist' in scenario_name:
        scenario_path = "../data/" + scenario_name + "/main.npz"
    else:
        scenario_path = "../data/zoo/" + scenario_name + ".npz"
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
        # scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    return train, dev, test

def nystrom(G,ind):
    Gnm = G[:,ind]
    sub_G = (Gnm)[ind,:]

    eig_val, eig_vec = numpy.linalg.eigh(sub_G+JITTER*np.eye(sub_G.shape[0]))
    eig_vec = np.sqrt(len(ind) / G.shape[0]) * np.matmul(Gnm, eig_vec)/eig_val
    eig_val /= len(ind) / G.shape[0]
    return eig_val, eig_vec


def train_cv_loss(params, l,k, train,test, nystr=False):
    Lambda, al, ak = params
    Lambda = Lambda
    X_train, Y_train, Z_train = train
    
    L_train = l(X_train,X_train,al,1)
    W_train = k(Z_train,Z_train,ak,1)/Z_train.shape[0]**2
    WY_train = np.matmul(W_train,Y_train)
    EYE = np.eye(W_train.shape[0])
    if nystr:
        LWL = np.matmul(L_train,np.matmul(W_train,L_train))
        eig_val, eig_vec = nystrom(LWL+Lambda*L_train,np.sort(np.random.choice(range(X_train.shape[0]),M,replace=False)))
        #tmp_val, tmp_vec = np.linalg.eigh(LWL+Lambda*L_train)
        alpha = EYE - eig_vec@np.linalg.inv(JITTER*np.eye(M)+np.diag(eig_val)@eig_vec.T@eig_vec)@np.diag(eig_val)@eig_vec.T
        # tmp_alpha = EYE - tmp_vec@np.linalg.inv(JITTER*np.eye(M) +tmp_vec.T@tmp_vec*tmp_val)@np.diag(tmp_val)@tmp_vec.T
        alpha = alpha@L_train@WY_train/JITTER
        # tmp_alpha = tmp_alpha@L_train@WY_train/JITTER
        # print(alpha)
        # print(tmp_alpha)
    else:
        # alpha = np.linalg.inv(np.matmul(W_train,L_train) + Lambda * np.eye(W_train.shape[0]))
        # alpha = np.matmul(alpha,WY_train)
        alpha = np.linalg.inv(Y_train @ Y_train.T / (Y_train.T@np.linalg.inv(L_train)@Y_train-1)+JITTER*EYE)@Y_train
    
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

def run_experiment_rkhs(scenario_name, seed=527):
    # load data
    train, dev, test = load_data(scenario_name)
    
    # training settings
    nreps = 1
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    ak = get_median_inter(Z)
    #al = get_median_inter(X)
    kf = KFold(n_splits=2,random_state=527)
    kf.get_n_splits(X)
    W = k(Z, Z, ak, 1)
    WY = np.matmul(W,Y)
    def loss(params):
        Lambda,al = params
        error_test = 0
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        L_train = l(X_train,X_train,al,1)
        W_train = W[train_index[:, None], train_index]
        WY_train = np.matmul(W_train,Y_train)
        alpha = np.linalg.inv(np.matmul(W_train,L_train) + Lambda * np.eye(W_train.shape[0]))
        alpha = np.matmul(alpha,WY_train)

        W_test = W[test_index[:, None], test_index]
        W_test -= np.diag(np.diag(W_test))
        L_test = l(X_test,X_train,al,1)
        Y_pred = np.matmul(L_test,alpha)
        diff_test = Y_test - Y_pred
        error_test += np.matmul(np.matmul(diff_test.T,W_test),diff_test)
        return error_test

    def callback(xk):
        global Nfeval
        print(str(Nfeval)+' - {} - {} '.format(xk,loss(xk)))
        Nfeval += 1

    for rep in range(nreps):

        Lambda0,al0 = 7,get_median_inter(X)
        save_path = os.path.join(folder, 'our_method_rkhs_%d.npz' % rep)
        loss_grad = value_and_grad(loss)
        result = minimize(loss_grad, x0=np.array([Lambda0,al0]), method='L-BFGS-B', bounds=[[3,10]]+[[0.01,10]]*X.shape[1],jac=True,options={'maxiter':50,'disp':True})
        Lambda,al = result.x
        L = l(X,X,al,1)
        test_L = l(test.x, X, al, 1)
        alpha = np.linalg.inv(np.matmul(W, L) + np.exp(Lambda) * np.eye(W.shape[0]))
        alpha = np.matmul(alpha,WY)

        # display predictions
        g_pred = np.matmul(test_L,alpha)
        # err = ((g_pred - test.g) ** 2).mean()
        np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=g_pred, al=al, Lambda=Lambda)
        # print(str(rep) + " done, test loss " + str(err))

        # plt.plot(test.x, g_pred, '.', label='pred')
        # plt.plot(test.x, test.g, '.', label='true')
        # plt.legend()
        # plt.savefig(str(ite) + '.pdf')
        # plt.close('all')
        # plt.show()


def run_experiment_rkhs_2(scenario_name,rep, nystr=True):
    # load data
    train, dev, test = load_data(scenario_name)

    # training settings
    folder = "results/zoo/" + scenario_name + "/"

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    train_tensor = [train.x,train.y,train.z]
    dev_tensor = [dev.x,dev.y,dev.z]
    ak = get_median_inter(Z)
    al0 = get_median_inter(X)
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)
    als = np.linspace(al0/10,al0*3,8)

    def loop(Li,al):
        cv_err = 0
        # al0 = np.random.randn()/10+np.sqrt(2) / 2
        # als = np.logspace(-2,1,32)
        for train_index, test_index in kf.split(X):
            train_tensor = [X[train_index],Y[train_index],Z[train_index]]
            dev_tensor = [X[test_index],Y[test_index],Z[test_index]]
            # loss_grad = value_and_grad(lambda al: train_cv_loss([Li,al,ak],l,k,train_tensor,None))
            # result = minimize(loss_grad,x0=np.array([al0]),method='L-BFGS-B',jac=True,bounds=[[0.01,5]],options={'maxiter':500})
            # result = np.array([train_cv_loss([Li,al,ak],l,k,train_tensor,None) for al in als])
            # al1 = result.x
            # al1 = als[np.argmin(result)]
            cv_err += train_cv_loss([Li,al,ak],l,k,train_tensor,dev_tensor,True)[0]
        # loss_grad = value_and_grad(lambda al: train_cv_loss([Li,al,ak],l,k,dev_tensor,None))
        # result = minimize(loss_grad, x0=np.array([al0]), method='L-BFGS-B', jac=True,bounds=[[0.01,5]], options={'maxiter':500})
        # al2 = result.x
        # cv_err += train_cv_loss([Li,al2,ak],l,k,dev_tensor,train_tensor)
        return cv_err

    Lambdas = np.logspace(-10,-6,8)
    save_path = os.path.join(folder, 'our_method_rkhs_%d.npz' % rep)

    # cross validation
    cv_errs = Parallel(n_jobs=20)(delayed(loop)(Li,al) for Li in Lambdas for al in als)
    optimid = np.argmin(cv_errs)
    i,j = divmod(optimid,len(als))
    Lambda = Lambdas[i]
    al = als[j]
    # loss_grad = value_and_grad(lambda al: train_cv_loss([Lambda, al, ak],l,k,X,None))
    # result = minimize(loss_grad, x0=np.array([al0]), method='L-BFGS-B', jac=True, bounds=[[0.01, 10]],options={'maxiter': 500})
    # results = Parallel(n_jobs=10)(delayed(train_cv_loss)([Lambda, al, ak], l, k, [X,Y,Z], None) for al in als)
    # al = als[np.argmin(results)]

    # test
    # res = [train_cv_loss([Lambda,al,ak],l,k,[X,Y,Z],None) for al in als]
    # al = als[np.argmin(res)]
    # print('al, Lambda ',al, Lambda)
   #  for al in als:
    test_L = l(test.x, X, al, 1)
    L = l(X,X,al,1)# l(X, X, al, 1)
    # print(Y.shape)
    # a,b=np.linalg.eigh(Y@ Y.T -L)
    # print(a)
    # return
    W = k(Z, Z, ak, 1)/Z.shape[0]**2
    WY = np.matmul(W, Y)
    EYE = np.eye(W.shape[0])
    if nystr:
        LWL = np.matmul(L,np.matmul(W,L))
        eig_val, eig_vec = nystrom(LWL+Lambda*L,np.sort(np.random.choice(range(X.shape[0]),M,replace=False)))
        alpha = EYE - eig_vec@np.linalg.inv(JITTER*np.eye(M) +np.diag(eig_val)@eig_vec.T@eig_vec)@np.diag(eig_val)@eig_vec.T
        alpha = alpha@L@WY/JITTER
    else:
        alpha = np.linalg.inv(np.matmul(W, L) + Lambda * np.eye(W.shape[0]))
        alpha = np.matmul(alpha,WY)

    g_pred = test_L @ alpha
    err = ((g_pred - test.g) ** 2).mean()
    np.savez('tmp_{}_{}_nystr{}.npz'.format(scenario_name,rep,nystr),cv_errs=cv_errs, Lambda=Lambda, al = al,g_pred = g_pred)
    print(scenario_name, ' ', rep, ' test_err ', err)
    ak = get_median_inter(Z)
    #np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=g_pred)
    #print(str(rep) + " done, test loss " + str(err) + ' ' + str(Lambda) + ' ' + str(al))


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

            L_train = l(X_train, X_train, al, 1)
            W_train = k(Z_train, Z_train, ak, 1)/Z_train.shape[0]**2
            WY_train = np.matmul(W_train,Y_train)
            alpha = np.linalg.inv(np.matmul(W_train,L_train) + Lambda * np.eye(W_train.shape[0]))
            alpha = np.matmul(alpha,WY_train)

            W_test = k(Z_test, Z_test, ak, 1)/Z_test.shape[0]**2
            W_test -= np.diag(np.diag(W_test))
            L_test = l(X_test, X_train, al, 1)
            Y_pred = np.matmul(L_test, alpha)
            diff_test = Y_test - Y_pred
            error_test += np.matmul(np.matmul(diff_test.T,W_test),diff_test)
        return error_test

    Lambda_list = numpy.linspace(-10,-5,32)
    al_list = np.linspace(al0[0,0]/3,al0[0,0]*3,10)
    WY = np.matmul(W, Y)
    eyes = np.eye(W.shape[0])

    def loop(i):
        al = al_list[i]
        L = l(X, X, al, 1)
        WL = np.matmul(W, L)
        test_L = l(test.x, X, al, 1)
        length =len(Lambda_list) 
        err_list = numpy.zeros(length)
        loss_list = numpy.zeros(length)
        for j in range(length):
            Lambda = Lambda_list[j]
            alpha = np.matmul(np.linalg.inv(WL+ np.exp(Lambda) *eyes),WY)

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

def plot_bayes(scenario_name, seed=527):
    # load data
    scenario_path = "../data/zoo/" + scenario_name + ".npz"
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    # training settings
    nreps = 1
    folder = "results/zoo/" + scenario_name + "/"

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    # X, Y, Z = train.x, train.y, train.z
    ak = get_median_inter(Z)
    W = k(Z, Z, ak, 1)/Z.shape[0]**2
    WY = np.matmul(W, Y)
    eyes = np.eye(X.shape[0])

    def log_marginal(params):
        ak, al, Lambda = params
        L = l(X,X,al,1)*Lambda
        inv_A = np.linalg.inv(eyes +np.matmul( W,L))
        inv_A = np.matmul(inv_A,W)
        (sign,logdet) = np.linalg.slogdet(inv_A)
        return (-logdet+np.matmul(np.matmul(Y.T, inv_A),Y))[0,0]
    
    Lambda_list = numpy.logspace(-10, 0, 10)
    al_list = numpy.logspace(-2,1,10)
    def loop(i,j):
        al = al_list[i]
        Lambda = Lambda_list[j]
        L = Lambda*l(X, X, al, 1)
        WL = np.matmul(W, L)
        test_L = Lambda*l(test.x, X, al, 1)
        alpha = np.matmul(np.linalg.inv(WL+ eyes), WY)
        g_pred = np.matmul(test_L, alpha)
        err_ij = ((g_pred - test.g) ** 2).mean()
        loss_ij = log_marginal([ak,al,Lambda])
        return err_ij, loss_ij
    result = Parallel(n_jobs=20)(delayed(loop)(i,j) for i in range(len(al_list)) for j in range(len(Lambda_list)))
    err_list = np.array([e[0] for e in result]).reshape((len(al_list),len(Lambda_list)))
    loss_list = np.array([e[1] for e in result]).reshape((len(al_list),len(Lambda_list)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    CS = ax1.contourf(numpy.tile(Lambda_list, [len(al_list), 1]), numpy.tile(al_list, [len(Lambda_list), 1]).T,
                      err_list)
    fig.colorbar(CS, ax=ax1)
    ax1.set_xlim(Lambda_list[0], Lambda_list[-1])
    ax1.set_ylim(al_list[0], al_list[-1])
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    plt.title('test error')

    CS = ax2.contourf(numpy.tile(Lambda_list, [len(al_list), 1]), numpy.tile(al_list, [len(Lambda_list), 1]).T,
                      loss_list)
    fig.colorbar(CS, ax=ax2)
    ax2.set_xlim(Lambda_list[0], Lambda_list[-1])
    ax2.set_xscale('log')
    plt.title('negative log marginal likelihood')

    plt.savefig('{}_bayes_{}.pdf'.format(scenario_name,ak), bbox_inches='tight')
    plt.close('all')
    # plt.show()
    np.save('err_{}_bayes_{}.npy'.format(scenario_name,ak),err_list)
    np.save('loss_{}_bayes_{}.npy'.format(scenario_name,ak),loss_list)


if __name__ == '__main__':
    scenario_list = ["step", "sin", "abs", "linear"]
    # scenario_list = ["mnist_x","mnist_z","mnist_xz"]
    #scenario_id = int(sys.argv[1])
    #scenario = scenario_list[scenario_id]
    # Parallel(n_jobs=20)(delayed(run_experiment_rkhs_2)(s,rep) for s in scenario_list for rep in range(1))
    for s in scenario_list:
        # plot_bayes(s)
        # train, dev, test = load_data(s)
        # res = np.load('tmp_{}_{}_nystr{}.npz'.format(s,0,'True'))
        # print(res['Lambda'],res['al'])
        # print(((res['g_pred']-test.g)**2).mean())
        for i in range(3):
            run_experiment_rkhs_2(s,i,nystr = True)
    # [plot_cv(s) for s in scenario_list]
    # run_experiment_rkhs(scenario)
    # a = np.load('tmp_step_0_nystrTrue.npz')
    # print(a['g_pred'])
