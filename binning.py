import numpy as np
from scipy.stats import zscore


def binning(exps, num_bins: int):

    """
    :param values: numpy array of values to be binned
    :param num_bins: log_2 of the number of bins i.e. bins = 2 -> 4 bins
    :return: bins: a sort of partitioning scheme which will be used with np.digitized
    :note: will remove outliers defined as any point with a zscore >3
    """

    exps = exps.flatten() #unpacking for some reason
    scores = np.abs(zscore(exps))
    # keep only data with -3 < zscore < 3 to create the binnings
    exps = exps[(scores < 2)]
    min = np.min(exps)
    max = np.max(exps)
    # ignore the first bit
    # this is in order to get a correct partitioning scheme
    bins = np.arange(min, max, (max - min)/2**num_bins)[1:]

    """values = [grad_app(bins[i-1],bins[i]) for i in range(1,len(bins))]
    # just say that anything that falls outside of the bins is 0 or 1 depending on which side
    values.insert(0,0)
    values.append(1)"""

    return bins


