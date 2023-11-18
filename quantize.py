import numpy as np
from binning import binning

def quantize(vals,num_bins):
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
    step = partitions[1] - partitions[1]
    for a in alpha:

        # edge cases: there is not a partition edge for the tails,
        # but digitize will not work correctly if we add the tail bins before calling it,
        # so just calculate what the bin edge would be for the values which fall in the tail
        if a == 0:
            beta[i] = ((partitions[0]) + (partitions[0] - step))/2
        elif a == N:
            beta[i] = ((partitions[-1]) + (partitions[-1] + step))/2
        # in general
        else:
            beta[i] = ((partitions[a - 1]) + (partitions[a]))/2
        i += 1

    return beta
