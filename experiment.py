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
    quantizers = ['unif', 'gauss']
    levels = [1,2,3,4]
    ulevels = [1,2,3,4,5,6,7,8]
    func = quantlogistic
    w0 = np.random.uniform(-1, 1, (X.shape[0], 1))
    scales = [np.sqrt(2),1/2,"sqrd"]
    for scale in scales:
        for type_w in quantizers:
            for type_q in quantizers:
                if type_w == 'unif':
                    levels_w = ulevels
                if type_w == 'gauss':
                    levels_w = levels
                if type_q == 'unif':
                    levels_q = ulevels
                if type_q == 'gauss':
                    levels_q = levels

                loss_grid = np.zeros((len(levels_w),len(levels_q)))
                for i, level_w in enumerate(levels_w):
                    for j, level_q in enumerate(levels_q):
                        start = time.time()
                        w, iters = grdescentquant(func, w0, 0.1, 10000, X, y, level_w, level_q, type_w, type_q, scale, tolerance=1e-02)
                        end = time.time()
                        #print(f'time: {end-start},iterations: {iters}')
                        loss = test_loss(w,X,y)
                        loss_grid[i,j] = loss
                xlabel = "gradient lvl"
                ylabel = "w lvl"
                plt.pcolormesh(loss_grid)
                plt.colorbar()
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(f"w quantizer: {type_w}, gradient quantizer: {type_q}, scale: {scale}")
                plt.show()
                print(loss_grid)



