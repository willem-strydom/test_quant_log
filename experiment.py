import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GrdDscnt import grdescentnormal
from GrdDscntQuant import grdescentquant
from NormalLog import normallogistic
from QuantLog import quantlogistic
from sklearn.model_selection import train_test_split
import time



def test_loss(w,X,y):
    #calculates test loss
    log_odds = np.dot(w.T, X)
    probs = 1 / (1 + np.exp(-log_odds))
    preds = (probs > 0.5).astype(int)
    preds = np.where(preds == 0, -1, preds)
    test_loss = np.sum(preds != y) / len(y)

    return test_loss

def experiment(X,y):
    # remember to transpose data to have shape dxn

    func = quantlogistic

    scales = [np.sqrt(2),1/2,"sqrd"]
    levels_w = [5,6,7,8]
    levels_q = [1,2,3,4]

    loss_grid = np.zeros((len(levels_w),len(levels_q)))
    for _ in range(10):
        w0 = np.random.uniform(-1, 1, (X.shape[0], 1))
        for i, level_w in enumerate(levels_w):
            for j, level_q in enumerate(levels_q):
                start = time.time()
                w, iters = grdescentquant(func, w0, 0.1, 10000, X, y, level_w, level_q, 'unif', 'unif', 1, tolerance=1e-02)
                end = time.time()
                #print(f'time: {end-start},iterations: {iters}')
                loss = test_loss(w,X,y)
                loss_grid[i,j] += loss
    loss_grid = loss_grid/10
    xlabel = "gradient lvl"
    ylabel = "w lvl"
    plt.pcolormesh(loss_grid)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.yticks([1,2,3,4], labels = [f"{w}" for w in levels_w])
    plt.ylabel(ylabel)
    plt.title(f"unif unif quant for 10 runs diff w0 same data")
    plt.show()
    print(loss_grid)
    w0 = np.random.uniform(-1, 1, (X.shape[0], 1))
    w, iters = grdescentnormal(normallogistic, w0, 0.1, 10000, X, y, tolerance=1e-02)
    loss = test_loss(w, X, y)
    print(loss, iters)




