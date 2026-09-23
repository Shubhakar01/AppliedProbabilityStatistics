import numpy as np

n_transactions = 50   # Number of transactions examined/hour
p_flag = 0.04         # Probability of a transaction being flagged
n_hours = 20000       # Number of independent one-hour periods

simulated_fraud_counts = np.random.binomial(n=n_transactions, p=p_flag, size=n_hours)

print("First 10 hours of simulated flagged transactions:")
print(simulated_fraud_counts[:10])

empirical_mean = np.mean(simulated_fraud_counts)
empirical_var = np.var(simulated_fraud_counts, ddof=1) # Sample variance

print(f"Empirical Mean (Average flagged per hour): {empirical_mean:.4f}")
print(f"Empirical Variance: {empirical_var:.4f}")

