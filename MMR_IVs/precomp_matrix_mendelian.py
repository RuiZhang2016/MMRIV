import numpy as np
from util import load_data, ROOT_PATH,_sqdist,get_median_inter_mnist
import os

def precomp(sname,seed=527):
    np.random.seed(seed)
    train, dev, test = load_data(ROOT_PATH+'/data/mendelian/'+sname+'.npz',Torch=False)

    M = int(train.z.shape[0]/400)
    train_K, dev_K = 0, 0
    for i in range(train.z.shape[1]):
        mat = _sqdist(train.z[:,[i]],None)
        train_K += mat
        mat = _sqdist(dev.z[:,[i]],None)
        dev_K += mat
    ak = np.median((np.sqrt(train_K)).flatten())
    train_K = (np.exp(-train_K/ak**2/2)+np.exp(-train_K/ak**2/200)+np.exp(-train_K/ak**2*50))/3
    dev_K = (np.exp(-dev_K/ak**2/2)+np.exp(-dev_K/ak**2/200)+np.exp(-dev_K/ak**2*50))/3
    folder = ROOT_PATH+'/mendelian_precomp/'
    os.makedirs(folder, exist_ok=True)
    np.save(folder+'{}_train_K.npy'.format(sname),train_K)
    np.save(folder+'{}_dev_K.npy'.format(sname),dev_K)

if __name__ == '__main__':
    scenarios = ["mendelian_{}_{}_{}".format(s, i, j) for s in [8, 16, 32] for i, j in [[1, 1]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[1, 0.5], [1, 2]]]
    scenarios += ["mendelian_{}_{}_{}".format(16, i, j) for i, j in [[0.5, 1], [2, 1]]]

    for s in scenarios:
        precomp(s)

