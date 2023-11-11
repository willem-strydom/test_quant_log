import numpy as np
from quantize import quantize

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

def quantlogistic(w,xTr,yTr,num_bins):

    y_pred = w.T @ xTr
    vals = yTr * y_pred
    #keeping same loss function as for normal log loss?
    loss = np.mean(np.log(1 + np.exp(-vals)))

    # get approximation of the gradient

    func = lambda x: 1/(1+np.exp(x))
    beta = quantize(vals, num_bins, func)
    gradient = -np.mean(yTr * xTr * beta, axis = 1).reshape(-1, 1)
    return loss, gradient