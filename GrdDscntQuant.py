import numpy as np

def grdescentquant(func, w0, maxiter, xTr, yTr, p):
    # INPUT:
    # func function to minimize
    # w0 = initial weight vector
    # p = nummber of bits to flip
    #
    # OUTPUTS:
    #
    # w = final weight vector
    eps = 2.2204e-14  # minimum step size for gradient descent

    num_iter = 0
    w = w0

    gradient = 0

    prior_gradient = np.zeros(w.shape)
    prior_w = np.zeros(w.shape)

    prior_loss = 1e06

    while num_iter < maxiter:
        loss, gradient = func(w, xTr, yTr)
        w1 = np.abs((w - gradient))
        # print(w1)
        w_grad_diff = np.argsort(w1, axis=0)
    # print(w_grad_diff)
        flipped = 0
        i =0
        while i < len(gradient) and flipped < p:
            index = w_grad_diff[i]

            if np.sign(gradient[index]) == w[index]:
    # print(num_iter)
                flipped += 1
                w[index] = -w[index]
            i +=1
        num_iter += 1
        prior_w = w
        if flipped == 0:
            break
    return w, num_iter