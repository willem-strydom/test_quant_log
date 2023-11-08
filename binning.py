import numpy as np
from scipy.stats import zscore
def binning(values, num_bins: int):

    """
    :param values: numpy array of values to be binned
    :param num_bins: log_2 of the number of bins i.e. bins = 2 -> 4 bins
    :return: bins: a sort of partitioning scheme which will be used with np.digitized
    :note: will remove outliers defined as any point with a zscore >3
    """
    print(values.shape)
    print(values)
    values = values[0] #unpacking for some reason...
    scores = np.abs(zscore(values))
    # keep only data with -3 < zscore < 3 to create the binnings
    values = values[(scores < 3)]
    min = np.min(values)
    max = np.max(values)
    # ignore the first bit
    # this is in order to get a correct partitioning scheme
    binnings = np.arange(min, max, (max - min)/2**num_bins)[1:]

    return binnings


