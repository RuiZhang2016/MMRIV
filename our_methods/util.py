def get_median_inter(x):
    n,m = x.shape
    def loop(a):
        A = a[:,None]# np.tile(x[:,[i]],[1,n])
        B = A.T
        dist = abs(A - B)
        dist = dist.flatten()
        med = np.median(dist)
        print(med)
        return med
    mat = np.array([loop(x[:,i]) for i in range(m)])
    return mat.reshape((1,-1))


def get_median_inter_mnist(x):
    x2 = np.sum(x*x,axis=1,keepdims=True)
    sqdist = x2+x2.T-2*x@x.T
    dist = np.sqrt((sqdist+abs(sqdist).T)/2)
    return np.median(dist.flatten())

def load_data(scenario_name,verbal=False):
    # load data
    # print("\nLoading " + scenario_name + "...")
    if 'mnist' in scenario_name:
        scenario_path = "../data/" + scenario_name + "/main.npz"
    else:
        scenario_path = "../data/zoo/" + scenario_name + ".npz"
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    if verbal:
        scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    return train, dev, test


def Kernel(name, torch=False):
    def poly(x,c,d):
        return (x @ x.T+c*c)**d

    def rbf(x,y,a,b,torch=torch):
        if y is None:
            y = x
        x,y = x/a, y/a
        if torch:
            x2,y2 = torch.sum(x*x,dim=1,keepdim=True),torch.sum(y*y,dim=1,keepdim=True)
        else:
            x2,y2 = np.sum(x*x,axis=1,keepdims=True),np.sum(y*y,axis=1,keepdims=True)
        sqdist = x2+y2.T-2*np.matmul(x,y.T)
        if y is None:
            sqdist = (sqdist+torch.abs(sqdist).T)/2
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
    
    # return the kernel function
    assert isinstance(name,str), 'name should be a string'
    kernel_dict = {'rbf':rbf,'poly':poly,'quad':quad}
    return kernel_dict[name]


