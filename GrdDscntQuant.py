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


    while num_iter < maxiter:
        loss, gradient = func(w, xTr, yTr)

        flipped = 0
        print(np.sign(gradient), w)
        expected_flips = np.sum(np.sign(gradient) == w)
        for index in range(len(w)):

            if np.sign(gradient[index]) == w[index]:
                flipped += 1
                w[index] = -w[index]

        num_iter += 1

        assert expected_flips == flipped
        if flipped == 0:
            break
    return w, num_iter