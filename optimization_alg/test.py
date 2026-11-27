from linear_program import milp

prices = [30, 20, 40, 10, 50]

history, profit = milp(prices)

print(history)

print("Total Profit:", profit)