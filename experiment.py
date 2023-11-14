import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GrdDscnt import grdescentnormal
from GrdDscntQuant import grdescentquant
from NormalLog import normallogistic
from QuantLog import quantlogistic
from sklearn.model_selection import train_test_split


def test_loss(w,X,y):
    #calculates test loss
    log_odds = np.dot(w.T, X)
    probs = 1 / (1 + np.exp(-log_odds))
    preds = (probs > 0.5).astype(int)
    preds = np.where(preds == 0, -1, preds)
    test_loss = np.sum(preds != y) / len(y)

    return test_loss

def experiment(X, y, gbins: list, wbins: list):

    """
    generates plots to compare performance of quantized gradient with normal gradient accross different number of bins
    :param X: features
    :param y: labels
    :param num_trials:
    :param gbins: array of number of bins to try
    :return: normal_iters, quant_iters w_quant, w
    """

    normal_iters = []
    quant_iters = []
    normal_loss = []
    quant_loss = []
    w_quants = []

    # split randomly into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # have to transpose data to get it to work with the function implementation where features are along the rows
    X_train = X_train.T
    X_test = X_test.T
    y_test = y_test.T
    y_train = y_train.T
    w0 = np.random.uniform(-1, 1, (X_train.shape[0], 1))
    w,iters = grdescentnormal(normallogistic, w0, 0.1, 50000, X_train, y_train)
    loss = test_loss(w, X_test, y_test)

    # store the results
    normal_iters = iters
    normal_loss = loss

    #do the same for quantized version
    num_gbin = gbins[0]
    for num_wbin in wbins:
        w_quant, iters = grdescentquant(quantlogistic, w0, 0.1, 50000, X_train, y_train, num_gbin, num_wbin)

        loss = test_loss(w_quant,X_test,y_test)

        quant_iters.append((num_wbin, iters))
        quant_loss.append((num_wbin, loss))
    iters_dict = dict(quant_iters)
    loss_dict = dict(quant_loss)
    iters_dict = {key: np.mean(values) for key, values in iters_dict.items()}
    loss_dict = {key: np.mean(values) for key, values in loss_dict.items()}
    loss_dict[0] = normal_loss
    iters_dict[0] = normal_iters

    plt.bar(loss_dict.keys(), loss_dict.values())
    plt.xlabel("log bins")
    plt.ylabel("mean test loss")
    plt.show()

    plt.bar(iters_dict.keys(), iters_dict.values())
    plt.xlabel("log bins")
    plt.ylabel("mean iterations until convergence")
    plt.show()


    return normal_iters, quant_iters,  w_quant, w