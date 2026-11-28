from linear_program import milp
import pandas as pd

prices = pd.read_parquet('Data/cleaned_data/dam.parquet')
prices = prices['lmp']

history, profit = milp(prices)

print(history)

print("Total Profit:", profit)