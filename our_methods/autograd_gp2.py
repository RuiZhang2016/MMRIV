from __future__ import absolute_import
from __future__ import print_function
import add_path
import matplotlib.pyplot as plt
from scenarios.abstract_scenario import AbstractScenario
import autograd.numpy as np
from autograd.numpy.linalg import solve
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from scipy.optimize import minimize
from torchvision import datasets, transforms
import random
from collections import defaultdict
import torch
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
ROOT_PATH = os.path.split(os.getcwd())[0]


def make_gp_funs(cov_func, num_cov_params):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_kernel_params(params):
        mean = params[0]
        cov_params = params[2:]
        noise_scale = np.exp(params[1]) + 0.0001
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x, xstar)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        pred_mean = mean + np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def log_marginal_likelihood(params, x, y, Ks):
        mean, cov_params, noise_scale = unpack_kernel_params(params)
    
        output_scale = np.exp(cov_params[0])
        lengthscales = np.exp(cov_params[1:])
        cov_y_y = 0
        for i in range(len(Ks)):
            cov_y_y += Ks[i]/lengthscales[i]
        cov_y_y = np.exp(-cov_y_y/2)*output_scale**2 + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y)

    return num_cov_params + 2, predict, log_marginal_likelihood


# Define an example covariance function.
def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x / lengthscales, 1) \
            - np.expand_dims(xp / lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))


digit_dict = None

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(ROOT_PATH+"/datasets", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])), batch_size=60000)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(ROOT_PATH+"/datasets", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])), batch_size=10000)
train_data, test_data = list(train_loader), list(test_loader)
images_list = [train_data[0][0].numpy(), test_data[0][0].numpy()]
labels_list = [train_data[0][1].numpy(), test_data[0][1].numpy()]
del train_data, test_data

images = np.concatenate(images_list, axis=0)
labels = np.concatenate(labels_list, axis=0)
idx = list(range(images.shape[0]))
random.shuffle(idx)
images = images[idx]
labels = labels[idx]
digit_dict = defaultdict(list)
for label, image in zip(labels, images):
    digit_dict[int(label)].append(image)
del images, labels, idx



def build_toy_dataset(D=1, n_data=2000, noise_std=0.1):
    rs = npr.RandomState(0)
    # inputs = np.concatenate([np.linspace(0, 3, num=int(n_data / 2)),
    #                         np.linspace(6, 8, num=int(n_data / 2))])
    inputs = np.random.uniform(-3,3,size=(n_data,1))
    targets = np.abs(inputs) + rs.randn(n_data,1) * noise_std
     
    # inputs = (inputs - 4.0) / 2.0
    # inputs = inputs.reshape((len(inputs), D))
    X_digits = np.clip(inputs*1.5+5, 0, 9).round()
    targets = np.abs((X_digits-5)/1.5).flatten()
    inputs = np.stack([random.choice(digit_dict[int(d)]).flatten() for d in X_digits.flatten()], axis=0)
    pca = PCA(n_components=16)
    pca.fit(inputs)
    inputs = pca.transform(inputs)
    return inputs, targets
    scaler = StandardScaler()
    scaler.fit(inputs)
    inputs = scaler.transform(inputs)
    return inputs, targets


if __name__ == '__main__':
    D = 16

    # Build model and objective function.
    num_params, predict, log_marginal_likelihood = \
        make_gp_funs(rbf_covariance, num_cov_params=D + 1)

    # X, y = build_toy_dataset(D=D)
    # print(X,y)
    # test_X = X[-400:]
    # test_y = y[-400:]
    # X = X[:-400]
    # y = y[:-400]
    # objective = lambda params: -log_marginal_likelihood(params, X, y)

    scenario = AbstractScenario(filename=ROOT_PATH+'/data/mnist_x/main.npz')
    scenario.to_2d()
    
    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    
    X = np.vstack((train.x,dev.x))[:1000]
    y = np.vstack((train.g,dev.g))[:1000]
    test_X = np.vstack((train.x,dev.x))[1000:2000]
    test_y = np.vstack((train.g,dev.g))[1000:2000]
    
    y = y.flatten()
    test_y = test_y.flatten()

    pca = PCA(n_components=16)
    pca.fit(X)
    X = pca.transform(X)
    test_X = X
    
    Ks = [rbf_covariance([1,1], X[:,[i]],X[:,[i]]) for i in range(X.shape[1])]
    objective = lambda params: -log_marginal_likelihood(params, X, y,Ks)

    def callback(params):
        print("Log likelihood {}".format(-objective(params)))
        plt.cla()

        # Show posterior marginals.
        plot_xs = test_X # np.reshape(np.linspace(-7, 7, 300), (300, 1))
        pred_mean, pred_cov = predict(params, X, y, plot_xs)
        print(((pred_mean-test_y)**2).mean(), params)
        return
    


    # Initialize covariance parameters
    rs = npr.RandomState(0)
    init_params = abs(rs.randn(num_params))*0.1

    print("Optimizing covariance parameters...")
    cov_params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='L-BFGS-B', callback=callback)
    plt.pause(10.0)
