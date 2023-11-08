import numpy as np

from binning import binning

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0) dx1

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def grad_app(val1,val2):
    # get the mean value of the actual gradient between two points
    approximation = (1+np.exp(val1))**-1 + (1+np.exp(val2))**-1
    return approximation/2
def quantlogistic(w,xTr,yTr,num_bins):

    y_pred = w.T @ xTr
    #keeping same loss function as for normal log loss?
    loss = np.mean(np.log(1 + np.exp(-yTr * y_pred)))

    # implement a better approximation of the gradient
    values = -yTr*y_pred
    bins = binning(values, num_bins)
    # get the integer bin numbers from digitize
    alpha = np.digitize(-yTr * y_pred, bins)[0]
    # map them to more appropriate values based on the real loss function

    beta = np.zeros_like(alpha)
    N = len(bins)
    i = 0
    for a in alpha:
        if a == 0:
            beta[i] = 0
        elif a == N:
            beta[i] = 1
        else:
            beta[i] = grad_app(bins[a-1],bins[a])
        i += 1

    gradient = -np.mean(yTr * xTr * beta, axis = 1).reshape(-1, 1)
    return loss, gradient