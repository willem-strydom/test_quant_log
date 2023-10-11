import numpy as np



def grdescentquant(func,w0,stepsize,maxiter,xTr,yTr,tolerance=1e-03):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    num_iter = 0
    w = w0
    gradient = 0
    prior_gradient = np.zeros(w.shape)
#why init to 0? should be large number or else bad no?
    prior_loss = 1e06
        # Increase the stepsize by a factor of 1.01 each iteration where the loss goes down,
        # and decrease it by a factor 0.5 if the loss went up. ...
        # also undo the last update in that case to make sure
        # the loss decreases every iteration
    while num_iter <maxiter:
        loss, gradient = func(w,xTr,yTr)
        # undo previous update if the loss got worse
        if loss > prior_loss:
            # undo the previous update
            w = np.sign(w + stepsize * prior_gradient)
            # decrease step-size
            stepsize = (stepsize / 1.01) * 0.5
            # take a smaller step
            w = np.sign(w - stepsize * prior_gradient)
        # to speed up convergence for the first few steps
        else:
            if num_iter < 10:
                stepsize = stepsize * 1.1
                w = np.sign(w - stepsize * gradient)
            else:
                stepsize = stepsize * 1.01
                w = np.sign(w - stepsize * gradient)
        if stepsize < eps:
            break
        if np.linalg.norm(gradient)<tolerance:
            print('gradient too small')
            break
        if np.array_equal(gradient,prior_gradient):
            print("gradient is unchanged")
            break
        prior_loss = loss
        prior_gradient = gradient
        num_iter += 1

    return w, num_iter