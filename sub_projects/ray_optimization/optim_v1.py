from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import numpy as np
from scipy.optimize import minimize


# objective function - loss function from david needed (sinkhorn)
def objective(x):
    loss = np.inf
    return loss


# surrogate - test a range of samples in order to select best candidates
def surrogate(model, samples):
    # returns mean and stdev of samples
    return model.predict(samples, return_std=True)


# return the probability of improvement of the acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = min(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs


# optimize the acquisition function
# X = current training set
# Xsamples = new candidate samples
# model = Gaussian Process model
def opt_acquisition(X, y, model):
    # cleverly select samples to optimize acquisition function (using BFGS?)
    Xsamples = bayes_sample()
    # Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the best scores
    ix = np.argmin(scores)
    return Xsamples[ix, 0]

# function to select samples to optimize acquisition
def bayes_sample(num_restarts=25):
    # here should be some code
    return best_sample


### optimisation loop ###

epochs = 100
X = random(100)  # samples
y = asarray([objective(x) for x in X])
# reshape into rows and cols
# X = X.reshape(len(X), 1)
# y = y.reshape(len(y), 1)

# define the model
model = GaussianProcessRegressor(kernel=None)  # RBF radial basis function kernel
# fit the model
model.fit(X, y)

# perform the optimization process
for i in range(epochs):
    # select the next point to sample
    x = opt_acquisition(X, y, model)
    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = surrogate(model, [[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # add the data to the dataset
    X = vstack((X, [[x]]))
    y = vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)

# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
