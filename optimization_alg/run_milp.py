import pandas as pd
import numpy as np
from linear_program import milp

# Load predicted prices
df = pd.read_csv('modelTesting/forecast_vectors.csv')

# Convert to numpy array
predictions = df.to_numpy()

print(predictions.shape)

# Store results
all_histories = []
all_profits = []

# Run MILP on each vector
for i in range(len(predictions)):
    history, profit = milp(predictions[i])

    all_histories.append(history)
    all_profits.append(profit)

    #print("Vector:", i + 1, "Profit:", profit)

# Create profit summary
results = pd.DataFrame({
    'Vector': range(1, len(predictions) + 1),
    'Predicted Profit': all_profits
})

# Save results
results.to_csv('milp_results.csv', index=False)

# Save trading histories
for i in range(len(all_histories)):
    all_histories[i]['Vector'] = i + 1

trading_history = pd.concat(all_histories, ignore_index=True)
trading_history.to_csv('milp_history.csv', index=False)

print(results)