import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GrdDscnt import grdescentnormal
from GrdDscntQuant import grdescentquant
from NormalLog import normallogistic
from QuantLog import quantlogistic
from sklearn.preprocessing import MinMaxScaler


def test_loss(w,X,y):
    #calculates test loss
    log_odds = np.dot(w.T, X)
    probs = 1 / (1 + np.exp(-log_odds))
    preds = (probs > 0.5).astype(int)
    preds = np.where(preds == 0, -1, preds)
    test_loss = np.sum(preds != y) / len(y)

    return test_loss

def experiment(X,y,num_trials):
    # performs gradient descent on a dataset
    # for both the normal and quantized versions
    # inputs: X , y :self-explanatory
    # inputs: num_trials :the number of times to perform gradient descent with a new random w0
    # outputs: num_iters_normal :number of iterations until stop criteria met for normal gradient descent
    # outputs: num_iters_quant :--//-- for quantized gradient descent
    # outputs: test_loss_normal :the 1-0 accuracy of the w_normal on the test set (E_out)
    # outputs: test_loss_quant :--//-- for w_quant


    # split randomly into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # have to transpose data to get it to work with the function implementation where features are along the rows
    X_train = X_train.T
    X_test = X_test.T
    y_test = y_test.T
    y_train = y_train.T

    normal_iters = []
    quant_iters = []
    normal_loss = []
    quant_loss = []
    for i in range(num_trials):
        w0 = np.random.uniform(-1, 1, (X_test.shape[0], 1))
        w,iters = grdescentnormal(normallogistic, w0, 0.1, 50000, X_train, y_train)
        loss = test_loss(w,X_test,y_test)

        # store the results
        normal_iters.append(iters)
        normal_loss.append(loss)

        #do the same for quantized version
        w_quant = np.sign(w0)
        w_quant, iters = grdescentquant(quantlogistic, w_quant, 0.1, 10000, X_train, y_train)
        loss = test_loss(w_quant,X_test,y_test)

        quant_iters.append(iters)
        quant_loss.append(loss)

    return normal_iters, quant_iters, normal_loss, quant_loss