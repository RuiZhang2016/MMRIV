import torch
import torch.optim as optim
import numpy
import autograd.numpy as np
from scenarios.abstract_scenario import AbstractScenario
import matplotlib.pyplot as plt
from autograd import value_and_grad
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from nn_model import get_median_inter
import scipy, time
import os
Nfeval = 1


def Kernel(name):
    def poly(x,c,d):
        return (x @ x.T+c*c)**d

    def rbf(x,y,a,b):
        if y is None:
            y = x
        x,y = x/a, y/a
        x2,y2 = np.sum(x*x,axis=1,keepdims=True),np.sum(y*y,axis=1,keepdims=True)
        sqdist = x2+y2.T-2*x@y.T
        if y is None:
            sqdist = (sqdist +abs(sqdist).T)/2
        out = b*b*np.exp(-sqdist)
        return out

    def laplace(x,a):
        return 0

    def quad(x,y,a,b):
        x, y = x/a, y/a
        x2, y2 = np.sum(x * x, axis=1, keepdims=True), np.sum(y * y, axis=1, keepdims=True)
        sqdist = x2 + y2.T - 2 * x @ y.T
        out = (sqdist+1)**(-b)
        return out

    assert isinstance(name,str), 'name should be a string'
    kernel_dict = {'rbf':rbf,'poly':poly,'quad':quad}
    return kernel_dict[name]



def run_experiment_rkhs(scenario_name, seed=527):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if 'mnist' in scenario_name:
        scenario_path = "../data/" + scenario_name + "/main.npz"
        folder = "/Users/ruizhang/{}/".format(scenario_name)
    else:
        scenario_path = "../data/zoo/" + scenario_name + ".npz"
        folder = "/Users/ruizhang/our_methods/results/zoo/" + scenario_name + "/"

    os.makedirs(folder, exist_ok=True)

    # load data
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")


    # tmp and may use later
    # scipy.io.savemat(scenario_name+'_data.mat',mdict={'X_train':train.x,'Y_train':train.y,'Z_train':train.z,
    #                                           'X_dev': dev.x, 'Y_dev': dev.y, 'Z_dev': dev.z,
    #                                           'X_test': test.x, 'g_test': test.g})

    # training settings
    nreps = 1

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    ak, al = 1, 1
    # kf = KFold(n_splits=2)
    # kf.get_n_splits(X)
    # W = k(Z, None, ak, 1)+ 1e-6*np.eye(Z.shape[0])
    # t0 = time.time()
    # u = numpy.linalg.cholesky(W)
    # print(scenario_name)
    # assert numpy.allclose(u@u.T,W)
    # assert all(numpy.diag(u)>0)
    # # u_inv = scipy.linalg.solve_triangular(u, np.eye(W.shape[0]), lower=False, overwrite_b=True)
    # u_inv = numpy.linalg.solve(u, np.eye(W.shape[0]))
    # assert numpy.allclose(u@u_inv, np.eye(W.shape[0])), u@u_inv
    # print(u_inv.T@u_inv@W-np.eye(W.shape[0]))
    # t1 = time.time()
    # print(t1-t0)
    # W_inv = numpy.linalg.inv(W)
    # print(W_inv @ W- np.eye(W.shape[0]))
    # print(time.time()-t1)
    def comp_kernel(x,y):
        xx = (x[:,Z.shape[1]:]).reshape((-1,X.shape[1]))
        zz = x[:,:Z.shape[1]]
        if y is not None:
            xx2 = y[:, Z.shape[1]:].reshape((-1,X.shape[1]))
            zz2 = y[:, :Z.shape[1]]
            L = l(xx,xx2, al, 1)
            K = k(zz,zz2, ak, 1)
        else:
            L = l(xx, None, al, 1)
            K = k(zz, None, ak, 1)
        return L.T @ K @ L + L
    eig_val,eig_vec = nystroem(np.hstack((Z,X)),numpy.random.choice(range(X.shape[0]),100,replace=False),lambda x,y : comp_kernel(x,y))
    print(eig_val)
    print(eig_vec)

    return

    def cv_loss(params, train_index, test_index = None):
        Lambda, al = params
        Lambda = np.exp(Lambda)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Lambda,al = params
        Lambda = np.exp(Lambda)

        L_train = l(X_train,X_train,al,1)
        W_train = W[train_index[:, None], train_index]
        alpha = np.linalg.inv(W_train @ L_train + Lambda * np.eye(W_train.shape[0])) @ W_train@ Y_train
        diff_train = Y_train - L_train @ alpha
        train_err = diff_train.T @ W_train @ diff_train+Lambda*alpha.T@L_train@alpha

        if test_index is None:
            return train_err[0,0]

        W_test = W[test_index[:, None], test_index]
        L_test = l(X_test,X_train,al,1)
        Y_pred = L_test @ alpha
        diff_test = Y_test - Y_pred
        test_err = diff_test.T @ W_test @ diff_test
        return test_err[0,0]

    Lambdas = np.linspace(3, 10, 5)
    for rep in range(nreps):
        al0 = np.random.randn()/10+np.sqrt(2) / 2
        save_path = os.path.join(folder, 'our_method_rkhs_%d.npz' % rep)

        # cross validation
        cv_errs = []

        for Li in Lambdas:
            cv_err = 0
            for train_index, test_index in kf.split(X):
                loss_grad = value_and_grad(lambda al: cv_loss([Li,al],train_index))
                result = minimize(loss_grad, x0=np.array([al0]), method='L-BFGS-B', jac=True,bounds=[[0.08,3]],  # [[3,20],[0.1,3]]
                                   options={'maxiter':500})
                cv_err += cv_loss([Li,result.x],train_index,test_index)
            print("Lambda, CV_err :",Li,cv_err)
            cv_errs += [cv_err]
        Lambda = Lambdas[np.argmin(cv_errs)]
        loss_grad = value_and_grad(lambda al: cv_loss([Lambda, al], train_index))
        result = minimize(loss_grad, x0=np.array([al0]), method='L-BFGS-B', jac=True, bounds=[[0.08, 3]],
                          # [[3,20],[0.1,3]]
                          options={'maxiter': 500, 'disp': True})

        # test
        al = result.x
        test_L = l(test.x, X, al, 1)
        L = l(X,X,al,1)
        alpha = np.linalg.inv(W @ L + np.exp(Lambda) * np.eye(W.shape[0])) @ W@Y

        # display predictions
        g_pred = test_L @ alpha
        err = ((g_pred - test.g) ** 2).mean()
        np.savez('tmp_{}_{}.npz'.format(scenario_name,rep),cv_errs=cv_errs, Lambda=Lambda, al = al,g_pred = g_pred)
        print(err)
        #np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=g_pred)
        #print(str(rep) + " done, test loss " + str(err) + ' ' + str(Lambda) + ' ' + str(al))


def plot_bayes(scenario_name, seed=527):
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    # X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    X, Y, Z = train.x, train.y, train.z
    ak = get_median_inter(Z)
    W = k(Z, None, ak, 1)
    WY = W @ Y
    eyes = np.eye(X.shape[0])

    def log_marginal(params):
        ak, al, Lambda = params
        Lambda = np.exp(Lambda)
        L = l(X,X,al,1)/Lambda
        inv_A = np.linalg.inv(eyes + W@L)@W
        (sign,logdet) = np.linalg.slogdet(inv_A)
        return (-logdet+Y.T@ inv_A @Y)[0,0]

    Lambda_list = numpy.linspace(5, 10, 5)
    al_list = numpy.logspace(-2,1,5)
    loss_list = np.array([[log_marginal([ak,al,Lambda]) for Lambda in Lambda_list] for al in al_list])
    err_list = np.zeros((len(al_list),len(Lambda_list)))
    for i in range(len(al_list)):
        al = al_list[i]
        L = l(X, X, al, 1)
        WL = W @ L
        test_L = l(test.x, X, al, 1)
        for j in range(len(Lambda_list)):
            Lambda = Lambda_list[j]
            alpha = np.linalg.inv(WL+ np.exp(Lambda) * eyes) @ WY
            g_pred = test_L @ alpha
            err = ((g_pred - test.g) ** 2).mean()
            err_list[i,j]=err

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    CS = ax1.contourf(numpy.tile(Lambda_list, [len(al_list), 1]), numpy.tile(al_list, [len(Lambda_list), 1]).T,
                      err_list)
    fig.colorbar(CS, ax=ax1)
    ax1.set_xlim(Lambda_list[0], Lambda_list[-1])
    ax1.set_ylim(al_list[0], al_list[-1])
    ax1.set_yscale('log')

    CS = ax2.contourf(numpy.tile(Lambda_list, [len(al_list), 1]), numpy.tile(al_list, [len(Lambda_list), 1]).T,
                      loss_list)
    fig.colorbar(CS, ax=ax2)
    ax2.set_xlim(Lambda_list[0], Lambda_list[-1])
    plt.savefig('tmp_{}_bayes.pdf'.format(scenario_name), bbox_inches='tight')
    plt.close('all')
    # plt.show()
    np.save('tmp_err_{}_bayes.npy'.format(scenario_name),err_list)
    np.save('tmp_loss_{}_bayes.npy'.format(scenario_name),loss_list)


def plot_cv(scenario_name,seed=527):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load data
    scenario_path = "../data/zoo/" + scenario_name + ".npz"
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    X, Y, Z = [np.vstack(e) for e in [(train.x, dev.x), (train.y, dev.y), (train.z, dev.z)]]
    ak = get_median_inter(Z)
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    W = k(Z, None, ak, 1)
    WY = W @ Y

    def loss(params):
        Lambda, al = params
        Lambda = np.exp(Lambda)
        error_test = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            L_train = l(X_train, X_train, al, 1)
            W_train = W[train_index[:, None], train_index]
            alpha = np.linalg.inv(W_train @ L_train + Lambda * np.eye(W_train.shape[0])) @ W_train @ Y_train

            W_test = W[test_index[:, None], test_index]
            L_test = l(X_test, X_train, al, 1)
            Y_pred = L_test @ alpha
            diff_test = Y_test - Y_pred
            error_test += diff_test.T @ W_test @ diff_test
        return error_test

    # Lambda_list = numpy.linspace(0,10,4)
    Lambda_list = numpy.linspace(5,10,5)
    al_list = numpy.logspace(-2,1,5)
    err_list = numpy.zeros((len(al_list),len(Lambda_list)))
    loss_list = numpy.zeros((len(al_list),len(Lambda_list)))
    eyes = np.eye(W.shape[0])

    for i in range(len(al_list)):
        al = al_list[i]
        L = l(X, X, al, 1)
        test_L = l(test.x, X, al, 1)
        for j in range(len(Lambda_list)):
            Lambda = Lambda_list[j]
            alpha = np.linalg.inv(W @ L+ np.exp(Lambda) *eyes)@ WY
            # predictions
            g_pred = test_L @ alpha
            err = ((g_pred - test.g) ** 2).mean()
            err_list[i,j] = err
            loss_list[i,j] = loss([Lambda,al])

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,8),sharey=True)
    CS = ax1.contourf(numpy.tile(Lambda_list,[len(al_list),1]),numpy.tile(al_list,[len(Lambda_list),1]).T,err_list)
    fig.colorbar(CS,ax=ax1)
    ax1.set_xlim(Lambda_list[0],Lambda_list[-1])
    ax1.set_ylim(al_list[0], al_list[-1])
    ax1.set_yscale('log')

    CS = ax2.contourf(numpy.tile(Lambda_list,[len(al_list),1]),numpy.tile(al_list,[len(Lambda_list),1]).T,loss_list)
    fig.colorbar(CS, ax=ax2)
    ax2.set_xlim(Lambda_list[0], Lambda_list[-1])
    plt.savefig('tmp_{}_cv.pdf'.format(scenario_name),bbox_inches='tight')
    plt.close('all')


def nystroem(X,x_ind,l):
    L = l(X, X)
    print('done', L.shape)
    Lnm = L[:,x_ind]
    sub_L = (Lnm)[x_ind,:]

    eig_val_L, eig_vec_L = numpy.linalg.eigh(sub_L)
    eig_vec_L = np.sqrt(len(x_ind) / X.shape[0]) * (Lnm @ eig_vec_L)/eig_val_L
    eig_val_L /= len(x_ind) / X.shape[0]
    return eig_val_L, eig_vec_L


if __name__ == '__main__':
    # for scenario in ['sin']:# ["linear", "sin","abs","step"]:
    #     run_experiment_rkhs(scenario)
    #     continue
    # folder = "/Users/ruizhang/our_methods/results/zoo/"
    # files = os.listdir(folder)
    # files = [f for f in files if 'cv' in f and 'loss' in f]
    # files = sorted(files)
    # for scenario in ["step", "sin", "abs", "linear"]:
    #     print('------------------')
    #     subsets = []
    #     for f in files:
    #         if scenario in f:
    #             res = np.load(folder+'/'+f)
    #             ids = np.argmin(res)
    #             res = np.load(folder+'/'+f.replace('loss','err'))
    #             print(f, (res.flatten())[ids])

    #
    # nrep = 10
    for scenario in ["step", "sin", "abs", "linear"]:
    #     optim_test_err = []
    #     for rep in range(nrep):
    #         res_list = []
    #         params_list = []
    #         for lr_id in range(3):
    #             for dw_id in range(4):
    #                 file = "/Users/ruizhang/results/zoo/"+scenario+"/our_method_nn_{}_{}_{}.npz".format(rep,lr_id,dw_id)
    #                 res = np.load(file)
    #                 res_list += [res['err'].astype(float)]
    #                 params_list += [[res['lr'].astype(float),res['dw'].astype(float),res['test_err']]]
    #
    #         optim_id = np.argmin(res_list)
    #         optim_test_err += [params_list[optim_id][2]]
    #         print(scenario,rep,params_list[optim_id],np,res_list[optim_id])
    #     print(numpy.mean(optim_test_err),numpy.std(optim_test_err))

        # res = np.load('/Users/ruizhang/tmp_{}_0_nystrTrue.npz'.format(scenario))
        # print(res['g_pred'])
        print(np.std([0.036,0.034,0.04]))

