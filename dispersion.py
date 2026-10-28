import numpy as np
import math

def numpy_dispersion(data):
    variance = np.var(data)
    std_dev = np.std(data)
    data_range = np.ptp(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Range: {data_range}")
    print(f"IQR: {iqr}")

def manual_dispersion(data):
    total_sum = 0
    count = 0
    for x in data:
        total_sum += x
        count += 1
    mean_val = total_sum / count

    sum_sq_diff = 0
    for x in data:
        sum_sq_diff += (x - mean_val) ** 2
    variance = sum_sq_diff / count
    std_dev = variance ** 0.5

    min_val = data[0]
    max_val = data[0]
    for x in data:
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x
    data_range = max_val - min_val

    sorted_data = sorted(data)
    def manual_percentile(s_data, p):
        k = (len(s_data) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s_data[int(k)]
        return s_data[int(f)] * (c - k) + s_data[int(c)] * (k - f)

    q1 = manual_percentile(sorted_data, 25)
    q3 = manual_percentile(sorted_data, 75)
    iqr = q3 - q1

    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Range: {data_range}")
    print(f"IQR: {iqr}")

data = np.random.randint(1, 101, size=10000)

print("NumPy Built-in")
numpy_dispersion(data)

print("\nManual Implementation")
manual_dispersion(data)