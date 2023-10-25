import numpy as np

continuous_values = np.linspace(0, 1, 100)  # Example: Range from 0 to 1

# Define the bin edges for the discrete mapping
bin_edges = np.array([0.2, 0.4, 0.6, 0.8])

# Use np.digitize to map the continuous range to discrete bins
discrete_values = np.digitize(continuous_values, bin_edges)

# The 'discrete_values' array contains the indices of the bins for each value
print(discrete_values)