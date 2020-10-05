import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scenarios.abstract_scenario import AbstractScenario
from our_methods.early_stopping import EarlyStopping
import time, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy

def Kernel(name):
    def poly(x,c,d):
        return (x @ x.T+c*c)**d

    def rbf(x,y,a,b):
        x,y = x/a, y/a
        x2,y2 = torch.sum(x*x,dim=1,keepdim=True),torch.sum(y*y,dim=1,keepdim=True)
        sqdist = x2+y2.T-2*x@y.T
        out = b*b*torch.exp(-sqdist)
        return out

    def laplace(x,a):
        return 0

    def quad(x,y,a,b):
        x, y = x /a, y /a
        x2, y2 = torch.sum(x * x, dim=1, keepdim=True), torch.sum(y * y, dim=1, keepdim=True)
        sqdist = x2 + y2.T - 2 * x @ y.T
        out = (sqdist+1)**(-b)
        return out



    assert isinstance(name,str), 'name should be a string'
    kernel_dict = {'rbf':rbf,'poly':poly,'quad':quad}
    return kernel_dict[name]

class Net(nn.Module):

    def __init__(self,input_size):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 100)  # 6*6 from image dimension
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):

    def __init__(self,input_size):
        super(CNN, self).__init__()
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def get_median_inter(x):
    n,m = x.shape
    vx = np.zeros((1,m))
    for i in range(m):
        A = np.tile(x[:,[i]],[1,n])
        B = A.T
        dist = abs(A - B)
        dist = dist.reshape(-1,1)
        vx[0,i] = np.median(dist,axis=0)
    return vx

def run_experiment_nn(scenario_name,seed=527):
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

    x,z,y = torch.from_numpy(train.x).float(),torch.from_numpy(train.z).float(),torch.from_numpy(train.y).float()
    dev_x, dev_z, dev_y, test_x = [torch.from_numpy(e).float() for e in [dev.x, dev.z,dev.y, test.x]]

    # training settings
    n_epochs = 2000
    batch_size = 1024
    nreps = 10

    # kernel
    kernel = Kernel('rbf')
    a = torch.from_numpy(get_median_inter(np.vstack((train.z,dev.z)))).float()
    # train_K = kernel(z,z,a,1)
    # dev_K = kernel(dev_z,dev_z,a,1)

    # training loop
    lrs = [0.1,0.01,0.001,0.0001,0.00001]
    folder = "results/zoo/" + scenario_name + "/"
    os.makedirs(folder, exist_ok=True)

    def my_loss(output, target, indices, K):
        d = output - target
        if indices is None:
            W = K
        else:
            W = K[indices[:, None], indices]
        loss = d.T @ W @ d / (d.shape[0]) ** 2
        return loss[0, 0]

    def fit(x,y,z,dev_x,dev_y,dev_z,a,lr,n_epochs=n_epochs):
        train_K = kernel(z, z, a, 1)
        if dev_z is not None:
            dev_K = kernel(dev_z,dev_z,a,1)
        n_data = x.shape[0]

        net = Net(x.shape[1])
        es = EarlyStopping(patience=10)
        optimizer = optim.Adam(list(net.parameters()), lr=lr)

        for epoch in range(n_epochs):
            permutation = torch.randperm(n_data)

            for i in range(0, n_data, batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x[indices], y[indices]

                # training loop
                def closure():
                    optimizer.zero_grad()
                    pred_y = net(batch_x)
                    loss = my_loss(pred_y, batch_y, indices, train_K)
                    loss.backward()
                    return loss

                optimizer.step(closure)  # Does the update
            if epoch % 5 == 0 and epoch > 0 and dev_x is not None:
                dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
                if es.step(dev_err):
                    break
        return es.best, epoch, net


    for rep in range(nreps):

        min_err = float("inf")
        optim_lr = None
        save_path = os.path.join(folder, 'our_method_nn_%d.npz' % rep)

        for lr in lrs:
            err1,epoch1,_ = fit(x,y,z,dev_x,dev_y,dev_z,a,lr)
            err2,epoch2,_ = fit(dev_x, dev_y, dev_z, x, y, z,a,lr)
            # print(err1.item(),err2.item(),lr)
            if err1+err2 < min_err:
                min_err = err1+err2
                optim_lr = lr

        _,_, net = fit(torch.cat([x,dev_x],dim=0),torch.cat([y,dev_y],dim=0),torch.cat([z,dev_z],dim=0),None,None,None,a,optim_lr,min(epoch1,epoch2))
        g_pred = net(test_x).detach().numpy()
        # test_err = ((g_pred - test.g) ** 2).mean()
        np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=g_pred)
        # print(str(rep)+" done, test loss "+str(test_err) + " optim lr "+str(optim_lr))

        # plt.plot(train.x, train.g, '*')
        # plt.plot(test.x, optim_g_pred, '.', label='pred')
        # plt.plot(test.x, test.g, '.', label='true')
        # plt.legend()
        # plt.show()





def run_experiment_rkhs(scenario_name,seed=527):
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

    x,z,y = torch.from_numpy(train.x).float(),torch.from_numpy(train.z).float(),torch.from_numpy(train.y).float()
    dev_x, dev_z, dev_y, test_x = [torch.from_numpy(e).float() for e in [dev.x, dev.z, dev.y, test.x]]

    # training settings
    nreps = 10

    folder = "results/zoo/" + scenario_name + "/"
    os.makedirs(folder, exist_ok=True)

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    W, dev_W = k(z, z, 1, 1),k(dev_z, dev_z, 1, 1)
    Wy = W @ y

    def step_decay(epoch):
        initial_lrate = 8
        drop = 0.95
        epochs_drop = 5.0
        lrate = initial_lrate * drop**((1 + epoch) / epochs_drop)
        return lrate

    for rep in range(nreps):
        optim_c = None

        t0 = time.time()
        save_path = os.path.join(folder, 'our_method_rkhs_%d.npz' % rep)
        la, lb = ag.Variable(torch.tensor(1.0), requires_grad=True),1
        Lambda = ag.Variable(torch.tensor(1.0),requires_grad=True)
        optimizer = optim.Adam([Lambda,la], lr=1)
        lrate = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=step_decay)
        for ite in range(150):
            optimizer.zero_grad()
            L, dev_L, test_L = l(x, x, la, lb),l(dev_x, x, la, lb),l(test_x, x, la, lb)
            alpha = torch.inverse(L@W@L+Lambda*L+torch.eye(L.shape[0]))@L@ Wy
            a = torch.inverse(L@W@L+Lambda*L+torch.eye(L.shape[0]))
            b = L@W@L+Lambda*L+torch.eye(L.shape[0])
            print(a@b)
            g_pred = dev_L @ alpha

            dev_d = dev_y - g_pred
            dev_err = dev_d.T @ dev_W @ dev_d

            dev_err.backward()
            optimizer.step()
            lrate.step()
            if ite % 1 == 0 and ite > 0:
                print(ite, dev_err.item(),Lambda.item(),lrate.get_lr(),la,lb)

        # if dev_err <= min_dev_err:
        #     optim_g_pred = l(test_x,x,1,1).numpy()@alpha
        #     min_dev_err = dev_err
        #     optim_c = c

        # display predictions
        g_pred = (test_L @ alpha).detach().numpy()
        test_err = ((g_pred - test.g) ** 2).mean()
        # np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=optim_g_pred)
        print(str(rep)+" done, test loss "+str(test_err) + ' optim ' + str(optim_c))

        plt.plot(train.x,train.g,'*')
        plt.plot(test.x,g_pred,'.',label='pred')
        plt.plot(test.x,test.g,'.',label='true')
        plt.legend()
        plt.show()



def run_experiment_rkhs2(scenario_name,seed=527):
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

    x,z,y = torch.from_numpy(train.x).float(),torch.from_numpy(train.z).float(),torch.from_numpy(train.y).float()
    dev_x, dev_z, dev_y, test_x = [torch.from_numpy(e).float() for e in [dev.x, dev.z, dev.y, test.x]]
    n_data = x.shape[0]

    # training settings
    nreps = 10

    folder = "results/zoo/" + scenario_name + "/"
    os.makedirs(folder, exist_ok=True)

    # pre-compute constants
    k, l = Kernel('rbf'), Kernel('rbf')
    la, lb = 1, 1
    W, dev_W, L, dev_L,test_L = k(z, z, 1, 1),k(dev_z, dev_z, 1, 1), l(x,x,la,lb),l(dev_x,x,la,lb),l(test_x,x,la,lb)
    Wy = W @ y
    u = torch.cholesky(L@W@L+L+1e-3*torch.eye(L.shape[0]))
    u_inv = u.inverse()
    print(u_inv.T@u_inv-L@W@L+L)


    def step_decay(epoch):
        initial_lrate = 8
        drop = 0.95
        epochs_drop = 5.0
        lrate = initial_lrate * drop**((1 + epoch) / epochs_drop)
        return lrate

    for rep in range(nreps):
        optim_g_pred = None
        min_dev_err = float("inf")
        optim_c = None

        Lambda = ag.Variable(torch.tensor(1.0),requires_grad=True)
        optimizer = optim.Adam([Lambda], lr=1)
        lrate = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=step_decay)
        for ite in range(150):
            optimizer.zero_grad()
            alpha = torch.inverse(L@W@L+Lambda*L+torch.eye(L.shape[0]))@L@ Wy
            a = torch.inverse(L@W@L+Lambda*L+torch.eye(L.shape[0]))
            b = L@W@L+Lambda*L+torch.eye(L.shape[0])
            print(a@b)
            g_pred = dev_L @ alpha

            dev_d = dev_y - g_pred
            dev_err = dev_d.T @ dev_W @ dev_d

            dev_err.backward()
            optimizer.step()
            lrate.step()
            if ite % 1 == 0 and ite > 0:
                print(ite, dev_err.item(),Lambda.item(),lrate.get_lr(),la,lb)

        # if dev_err <= min_dev_err:
        #     optim_g_pred = l(test_x,x,1,1).numpy()@alpha
        #     min_dev_err = dev_err
        #     optim_c = c

        # display predictions
        g_pred = (test_L @ alpha).detach().numpy()
        test_err = ((g_pred - test.g) ** 2).mean()
        # np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=optim_g_pred)
        print(str(rep)+" done, test loss "+str(test_err) + ' optim ' + str(optim_c))

        plt.plot(train.x,train.g,'*')
        plt.plot(test.x,g_pred,'.',label='pred')
        plt.plot(test.x,test.g,'.',label='true')
        plt.legend()
        plt.show()


if __name__ == '__main__':


    # Lambdas = np.linspace(-12,-5,32)
    #
    for scenario_name in ["mnist_xz"]:
    #     # run_experiment_nn(scenario)
    #     # continue
    #
    #     print(scenario_name)
    #     # folder = "../results/zoo/" + scenario + "/"
    #     # nreps = 10
    #     # test_errs = []
    #     # ts = []
    #     # for rep in range(nreps):
    #     #     load_path = os.path.join(folder, 'Ours_%d.npz' % rep)
    #     #     res = np.load(load_path)
    #     #     test_err = ((res['g_true']-res['g_hat'])**2).mean()
    #     #     test_errs += [test_err]
    #     #     ts += [res['t']]
    #     # print("## DeepGMM test error mean " + str(np.mean(test_errs)) + " std "+str(np.std(test_errs))+' time '
    #     #       +str(np.mean(ts))+' - '+str(np.std(ts)))
    #
    #     folder = "results/zoo/" + scenario_name + "/"
    #     # nreps = 10
    #     # test_errs = []
    #     # for rep in range(nreps):
    #     #     load_path = os.path.join(folder, 'our_method_nn_%d.npz' % rep)
    #     #     res = np.load(load_path)
    #     #     test_err = ((res['g_true']-res['g_hat'])**2).mean()
    #     #     test_errs += [test_err]
    #     #     # ts += [res['t']]
    #     # print("## our method (NN) test error mean " + str(np.mean(test_errs)) + " std "+str(np.std(test_errs)))
    #
    #
    #     # nreps = 1
    #     # test_errs = []
    #     # for rep in range(nreps):
    #     #     load_path = os.path.join(folder, 'our_method_rkhs_%d.npz' % rep)
    #     #     if os.path.exists(load_path):
    #     #         res = np.load(load_path)
    #     #         test_err = ((res['g_true']-res['g_hat'])**2).mean()
    #     #         test_errs += [test_err]
    #     #     else:
    #     #         pass
    #     # print("## our method (RKHS) test error mean " + str(np.mean(test_errs)) + " std "+str(np.std(test_errs)))
    #     try:
    #         scenario_path = "../data/zoo/" + scenario_name + ".npz"
    #         scenario = AbstractScenario(filename=scenario_path)
    #         scenario.to_2d()
    #
    #         train = scenario.get_dataset("train")
    #         dev = scenario.get_dataset("dev")
    #         test = scenario.get_dataset("test")
    #         als = np.logspace(-2, 1, 10)
    #
    #         err = np.load('/Users/ruizhang/err_{}_cvu_[[1.74563831 1.77592396]].npy'.format(scenario_name))
    #         loss = np.load('/Users/ruizhang/loss_{}_cvu_[[1.74563831 1.77592396]].npy'.format(scenario_name))
    #
    #         X = np.vstack((train.x, dev.x))
    #         al0 = get_median_inter(X)
    #         print(al0)
    #         als = np.linspace(al0[0,0]/3,al0[0,0]*3,10)
    #
    #         r,c = err.shape
    #         optim_r, optim_c = divmod(np.argmin(err),c)
    #         print('err optim id ({},{}):'.format(r,c), optim_r, optim_c,'optim al and Lambda: ', als[optim_r],Lambdas[optim_c])
    #         optim_r, optim_c = divmod(np.argmin(loss), c)
    #         print('-> loss optim id ({},{}):'.format(r,c), optim_r, optim_c, 'optim al and Lambda: ', als[optim_r],Lambdas[optim_c])
    #         optimid = np.argmin(loss)
    #         print('choose err: ', err.flatten()[optimid],' ideal :', np.min(err.flatten()))
    #
    #         # res = np.load('/Users/ruizhang/tmp_{}_0.npz'.format(scenario_name))
    #         # plt.scatter(test.x,res['g_pred'])
    #         # plt.show()
    #
    #
    #
    #         # print(((res['g_pred']-test.g)**2).mean())
    #     except Exception as e:
    #         print(e)
        if 'mnist' in scenario_name:
            scenario_path = "../data/" + scenario_name + "/main.npz"
            folder = "/Users/ruizhang/{}/".format(scenario_name)
        else:
            scenario_path = "../data/zoo/" + scenario_name + ".npz"
            folder = "/Users/ruizhang/our_methods/results/zoo/" + scenario_name + "/"


        err_mean = []
        for rep in range(10):
            print(rep)
            train_err = []
            test_err = []
            for lr_id in range(3):
                for dw_id in range(5):
                    save_path = os.path.join(folder, 'our_method_nn_{}_{}_{}.npz'.format(rep, lr_id, dw_id))
                    try:
                        res = np.load(save_path)
                        train_err += [res['err']]
                        test_err += [res['test_err']]
                    except Exception as e:
                        print(e)

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
            train_err = np.reshape(train_err,(3,5))
            # ax1.contourf(train_err)
            ind = np.argmin(train_err[1])
            # r,c = divmod(ind, 5)
            test_err = np.reshape(test_err, (3, 5))
            # ax2.contourf(test_err)
            print(ind,train_err[1,ind],test_err[1,ind])
            err_mean += [test_err[1,ind]]
            # plt.show()
        print(np.mean(err_mean),np.std(err_mean))



