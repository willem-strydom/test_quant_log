import numpy as np

def iterate_binary_combinations(n):
    for i in range(2**n):
        combination = []
        for j in range(n):
            combination.append((i >> j) & 1)
        combination = [-1 if x == 0 else x for x in combination]
        combination = np.array(combination)

        yield combination.reshape(-1,1)

# Example
"""n = 2
for combination in iterate_binary_combinations(n):
    print(combination)
"""

# 00, 10, 01, 11