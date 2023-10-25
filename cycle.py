# Find the size of the cycle that the gradient descent alg is stuck in
# Start from end, go until the final weight vector has appeared twice, assuming that one cycle haas already completed

def cycle(W):
    w_init = W[-1]
    i = 1
    for w in reversed(W[:-1]):
        if w == w_init:
            return i
        i+=1
