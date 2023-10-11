import numpy as np

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
def quantlogistic(w,xTr,yTr):

    y_pred = w.T@xTr
    #keeping same loss function as for normal log loss?
    loss = np.sum(np.log(1 + np.exp(-yTr * (y_pred)))) / xTr.shape[1]
    gradient = -np.sum(yTr*xTr*np.sign(yTr * (y_pred)), axis=1).reshape(-1, 1)
    return loss,gradient