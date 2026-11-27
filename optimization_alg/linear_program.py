import pulp
import pandas as pd
import numpy as np


def milp(prices):
    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    # Create optimization model
    model = pulp.LpProblem('Battery_Optimization', pulp.LpMaximize)

    # Decision variables
    buy = {i: model.add_variable(f'Buy_{i}', cat='Binary') for i in range(n)}
    sell = {i: model.add_variable(f'Sell_{i}', cat='Binary') for i in range(n)}
    battery = {i: model.add_variable(f'Battery_{i}', cat='Binary') for i in range(n + 1)}

    # Objective function
    model += pulp.lpSum(prices[i] * sell[i] - prices[i] * buy[i] for i in range(n))

    # Battery starts full
    model += battery[0] == 1

    # Battery ends empty
    model += battery[n] == 0

    # Battery constraints
    for i in range(n):
        # Cannot buy and sell at the same time
        model += buy[i] + sell[i] <= 1

        # Update battery status
        model += battery[i + 1] == battery[i] + buy[i] - sell[i]

    # Solve optimization model
    stats = model.solve(pulp.COIN_CMD(msg=False))

    # Check if optimal solution was found
    if stats.status != pulp.LpSolveStatus.Optimal:
        raise RuntimeError('Optimal solution was not found')

    # Create trading history
    history = []
    total_profit = 0

    for i in range(n):
        if pulp.value(buy[i]) > 0.5:
            action = "Buy"
            profit = -prices[i]

        elif pulp.value(sell[i]) > 0.5:
            action = "Sell"
            profit = prices[i]

        else:
            action = "Hold"
            profit = 0

        total_profit += profit

        history.append({
            'Interval': i + 1,
            'Price': prices[i],
            'Action': action,
            'Battery': round(pulp.value(battery[i + 1])),
            'Profit': profit,
            'Cumulative Profit': total_profit
        })

    # Convert history to DataFrame
    history = pd.DataFrame(history)

    return history, total_profit