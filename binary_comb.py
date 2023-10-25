def iterate_binary_combinations(n):
    for i in range(2**n):
        combination = []
        for j in range(n):
            combination.append((i >> j) & 1)
        yield combination

# Example usage:
n = 3
for combination in iterate_binary_combinations(n):
    print(combination)
