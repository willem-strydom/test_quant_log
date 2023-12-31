import numpy as np
from scipy.stats import zscore


def binning(vals, bins: int, type:str, scale):

    """

    :param type: "unif" or "gauss" -> assumed distribution
    :param bins: log2 number of bins... must be 1,2,3,or 4 for now
    :param scale: either 2**1/2 var
    :return: (2^num_bins)+1 partitions.. need to chop the ends off though numpy 1xd array
    """
    var = np.var(vals)
    mean = np.mean(vals)
    if scale == "sqrd":
        scale = var
    if type == 'unif':
        a = mean - np.sqrt(3*var)
        b = mean + np.sqrt(3*var)
        return np.linspace(a, b, num=2**bins +1)
    elif type == 'gauss':
        if bins == 1:
            return np.arange(-1.596,1.597,1.596) * var * scale + mean
        elif bins == 2:
            return np.arange(-1.991,1.992,0.996) * var * scale + mean
        elif bins == 3:
            return np.arange(-2.344,2.345,0.586) * var * scale + mean
        elif bins == 4:
            return np.arange(-2.68,2.69,0.335) * var * scale + mean


def _binning():
    vals = np.arange(0,11,1)
    bins = [1,2,3,4]
    types = ['unif', 'gauss']
    scales = [np.sqrt(2), 1/2, np.mean(vals)]
    for bin in bins:
        for type in types:
            for scale in scales:
                print(binning(vals,bin,type,scale), f"{bin}, {type}, {scale}")

