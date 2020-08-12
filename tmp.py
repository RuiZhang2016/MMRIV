import numpy as np

x = np.load('tmp_x.npy')
print(x)

x2 = np.sum(x*x,axis=1,keepdims=True)
tmp_x2 = np.load('tmp_x2.npy')
print(x2-tmp_x2)

sqdist = x2+x2.T-2*x@x.T
tmp_sqdist = np.load('tmp_sqdist.npy')
tmp_sq =tmp_sqdist- np.diag(np.diag(tmp_sqdist))
print(np.all(tmp_sq>=0))
