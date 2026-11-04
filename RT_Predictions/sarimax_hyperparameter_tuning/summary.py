import glob
import json
import pandas as pd

results = []
for filepath in glob.glob('run_2_results/result_*.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
        params = data['params']
        metrics = data['metrics']
        
        row = {
            'task_id': data['task_id'],
            'order': str(params['arima_order']),
            'fourier_k': params['fourier_k'],
            'lookback': params['lookback'],
            'exog_cols': ", ".join(params['exog_cols']) if params['exog_cols'] else "None",
            'val_mae': round(metrics['val_mae'], 2),
            'val_rmse': round(metrics['val_rmse'], 2)
        }
        results.append(row)

df_summary = pd.DataFrame(results).sort_values(by='val_rmse')
print("=== TOP 10 SARIMAX CONFIGURATIONS BY VALIDATION MAE ===")
print(df_summary.head(7).to_string(index=False))
