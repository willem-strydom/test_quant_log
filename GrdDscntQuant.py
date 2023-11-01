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
    # num_iter = number of times the gradient was calculated

    num_iter = 0
    w = w0
    w_prev = None
    w_prev2 = None
    while num_iter < maxiter:
        loss, gradient = func(w, xTr, yTr)

        flipped = 0
        expected_flips = np.sum(np.sign(gradient) == w)

        for index in range(len(w)):

            if np.sign(gradient[index]) == w[index]:
                flipped += 1
                w[index] = -w[index]

        num_iter += 1

        assert expected_flips == flipped


        if np.all(w_prev2 == w):
            print('in loop')
            break

        if flipped == 0:
            print('no flips')
            break

        w_prev2 = w_prev
        w_prev = w
    return w, w_prev, w_prev2, num_iter