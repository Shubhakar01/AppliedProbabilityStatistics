# Measure of Central Tenency
# Generate 10,000 random samples and compute Mean,modem median
# 1. Using Numpy (Using built in function )
# 2. Not built in and then compare

import numpy as np

def numpy_central_tendency(data):
    mean_val = np.mean(data)
    median_val = np.median(data)
    vals, counts = np.unique(data, return_counts=True)
    mode_val = vals[np.argmax(counts)]
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Mode: {mode_val}")

def manual_central_tendency(data):
    total_sum = 0
    count = 0
    for x in data:
        total_sum += x
        count += 1
    mean_val = total_sum / count

    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        median_val = sorted_data[n // 2]
    else:
        mid = n // 2
        median_val = (sorted_data[mid - 1] + sorted_data[mid]) / 2

    freq_dict = {}
    for x in data:
        if x in freq_dict:
            freq_dict[x] += 1
        else:
            freq_dict[x] = 1

    max_freq = -1
    mode_val = None
    for val, freq in freq_dict.items():
        if freq > max_freq:
            max_freq = freq
            mode_val = val

    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Mode: {mode_val}")

data = np.random.randint(1, 101, size=10000)

print("NumPy Built-in")
numpy_central_tendency(data)
print("\nManual Implementation")
manual_central_tendency(data)