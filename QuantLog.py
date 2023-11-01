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
def test_loss(w,X,y):
    #calculates test loss, I think we wanted to change the loss func at some point
    log_odds = np.dot(w.T, X)
    probs = 1 / (1 + np.exp(-log_odds))
    preds = (probs > 0.5).astype(int)
    preds = np.where(preds == 0, -1, preds)
    test_loss = np.sum(preds != y) / len(y)

    return test_loss
def quantlogistic(w,xTr,yTr):

    y_pred = w.T @ xTr
    #keeping same loss function as for normal log loss?
    loss = np.mean(np.log(1 + np.exp(-yTr * y_pred)))

    # implement more binnings
    bins = [-0.5, 0.5]
    alpha = np.digitize(-yTr * y_pred, bins)
    alpha = alpha/(len(bins))
    gradient = np.mean(yTr * xTr * alpha, axis = 1).reshape(-1, 1)
    return loss, gradient