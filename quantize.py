import numpy as np
from binning import binning

def quantize(vals,num_bins,func):
    """
    :param vals: numpy nx1 array of func arguments that will be quantized
    :param num_bins: int: log2 number of bins/quantization levels
    :param func: function of one variable over vals which will be approximated
    :return: nx1 numpy array of quantized values
    """
    partitions = binning(vals,num_bins)
    # alpha is a list of which bin each val belongs to
    alpha = np.digitize(vals, partitions).flatten()
    # map them to appropriate values based on the mean of func evaluation of the respective bin edges

    beta = np.zeros(alpha.shape)
    N = len(partitions)
    i = 0
    for a in alpha:

        # edge cases: there is not a partition edge for the tails, so just set them to the edge partition value
        if a == 0:
            beta[i] = func(partitions[0])
        elif a == N:
            beta[i] = func(partitions[-1])
        # in general
        else:
            beta[i] = (func(partitions[a - 1]) + func(partitions[a]))/2
        i += 1

    return beta
