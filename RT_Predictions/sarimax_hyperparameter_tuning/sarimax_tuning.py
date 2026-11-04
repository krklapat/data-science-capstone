import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import sys
import json
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuration
DATE_COL = "interval_start_utc"
TARGET_COL = "lmp_rtm"

# Train / Validation / Test Cutoffs
TRAIN_END_DATE = "2025-07-01 00:00:00+00:00"
VAL_END_DATE   = "2026-07-01 00:00:00+00:00"

# Horizon and Windowing
FORECAST_HORIZON = 288       # 288 steps = 24 hours (at 5-min resolution)
STRIDE_STEPS     = 240       # Step size for rolling validation (240 steps = 20 hours)

# Outlier Handling
OUTLIER_THRESHOLD = 132


# Transformations
def apply_shifted_log_transform(df: pd.DataFrame, target_col: str):
    """
    Applies y_t = log(LMP + C) where C = |min(LMP)| + 1 to safely convert
    negative and zero prices into log domain without destroying order or rank.
    """
    min_val = df[target_col].min()
    shift_constant = abs(min_val) + 1.0 if min_val <= 0 else 0.0
    
    df['target_transformed'] = np.log(df[target_col] + shift_constant)
    return df, shift_constant

# Inverse Transformation Helper
def inverse_shifted_log(transformed_vals, shift_constant):
    """Converts log-scale predictions back to real $/MWh values."""
    return np.exp(transformed_vals) - shift_constant


# Fourier Term Generation
def generate_fourier_terms(df: pd.DataFrame, period: float = 288.0, k_terms: int = 4) -> pd.DataFrame:
    """
    Generates k Sine and Cosine harmonic pairs for a daily 288-step cycle.
    """
    fourier_df = pd.DataFrame(index=df.index)
    
    # Integer time index for continuous sine/cosine evaluation
    t = np.arange(len(df))
    
    for k in range(1, k_terms + 1):
        fourier_df[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        fourier_df[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
        
    return fourier_df

def construct_exogenous_matrix(df: pd.DataFrame, fourier_df: pd.DataFrame, exog_cols: list) -> pd.DataFrame:
    """Combines Fourier terms with external market features into a unified matrix X."""
    X = pd.concat([fourier_df, df[exog_cols]], axis=1)
    return X


# Validation Pipeline
def run_rolling_sarimax_validation(
    df: pd.DataFrame,
    X_matrix: pd.DataFrame,
    trans_target_col: str,
    orig_target_col: str,
    shift_constant: float,
    train_end: str,
    val_end: str,
    horizon: int = 288,
    stride: int = 48,
    lookback: int = 2016,
    order: tuple = (1, 0, 1)
):
    """
    Executes a rolling-window 288-step dynamic forecast evaluation across
    the validation set using SARIMAX + Fourier exogenous inputs.
    """
    # Split index boundaries
    val_mask = (df.index >= train_end) & (df.index < val_end)
    val_indices = np.where(val_mask)[0]
    
    # Ensure validation sequence has room for full horizon evaluation
    valid_start_indices = [idx for idx in val_indices[::stride] if idx + horizon <= len(df)]
    
    forecasts_real = []
    actuals_real = []
    forecast_timestamps = []
    
    #print(f"Beginning Rolling Validation across {len(valid_start_indices)} origins...")
    
    for i, t_origin in enumerate(valid_start_indices):
        # Define historical training window
        t_start = max(0, t_origin - lookback)
        
        # Extract target slices
        y_train = df[trans_target_col].iloc[t_start:t_origin]
        
        # Extract exogenous matrix slices (Train vs. Horizon Future)
        X_train = X_matrix.iloc[t_start:t_origin]
        X_forecast = X_matrix.iloc[t_origin : t_origin + horizon]
        
        # Fit SARIMAX on residual errors after Fourier + Exog fit
        model = SARIMAX(
            endog=y_train,
            exog=X_train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)
        
        # Generate 288-step vector prediction in log space
        log_pred = fitted_model.forecast(steps=horizon, exog=X_forecast)
        
        # Inverse transform back to $/MWh
        real_pred = inverse_shifted_log(log_pred.values, shift_constant)
        real_actual = df[orig_target_col].iloc[t_origin : t_origin + horizon].values
        
        # Append outputs
        forecasts_real.append(real_pred)
        actuals_real.append(real_actual)
        forecast_timestamps.append(df.index[t_origin : t_origin + horizon])
        
        if (i + 1) % 50 == 0 or (i + 1) == len(valid_start_indices):
            #print(f"Completed {i + 1}/{len(valid_start_indices)} rolling origins.")
            pass
            
    return np.array(forecasts_real), np.array(actuals_real), forecast_timestamps


# Model Evaluation Function
def evaluate_model(params, task_id):
    # Import data
    df = pd.read_parquet(r"../Data/cleaned_data/all.parquet")

    # Preprocess Data
    df_clean = df.copy()
    # 1. Datetime formatting and index setup
    df_clean[DATE_COL] = pd.to_datetime(df_clean[DATE_COL])
    df_clean = df_clean.sort_values(DATE_COL).reset_index(drop=True)
    df_clean = df_clean.set_index(DATE_COL)
    # 2. Linear Interpolation for missing target values
    if df_clean[TARGET_COL].isnull().sum() > 0:
        print(f"Interpolating {df_clean[TARGET_COL].isnull().sum()} missing values in {TARGET_COL}...")
        df_clean[TARGET_COL] = df_clean[TARGET_COL].interpolate(method='time').bfill().ffill()
    # 3. Interpolate missing values in exogenous features if present
    for col in params.get('exog_cols', []):
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].interpolate(method='time').bfill().ffill()
    # 4. Set all values over the outlier threshold to the threshold
    df_clean[TARGET_COL] = df_clean[TARGET_COL].clip(upper=OUTLIER_THRESHOLD)

    # Transformations
    if params.get('transform_target', False):
        df_clean, shift_constant = apply_shifted_log_transform(df_clean, TARGET_COL)
        trans_col = 'target_transformed'
    else:
        trans_col = TARGET_COL
        shift_constant = 0.0

    # Generate Fourier Terms
    fourier_df = generate_fourier_terms(df_clean, period=288, k_terms=params.get('fourier_k', 4))
    X_matrix = construct_exogenous_matrix(df_clean, fourier_df, params.get('exog_cols', []))

    # Validation
    forecasts, actuals, timestamps = run_rolling_sarimax_validation(
        df=df_clean,
        X_matrix=X_matrix,
        trans_target_col=trans_col,
        orig_target_col=TARGET_COL,
        shift_constant=shift_constant,
        train_end=TRAIN_END_DATE,
        val_end=VAL_END_DATE,
        horizon=FORECAST_HORIZON,
        stride=STRIDE_STEPS,
        lookback=params.get('lookback', 2016),
        order=tuple(params.get('arima_order', (1, 1, 1)))
    )

    # Evaluation
    mae_overall = mean_absolute_error(actuals.flatten(), forecasts.flatten())
    rmse_overall = np.sqrt(mean_squared_error(actuals.flatten(), forecasts.flatten()))

    # Saving Results
    results = {
        'task_id': task_id,
        'params': params,
        'metrics': {
            'val_mae': mae_overall,
            'val_rmse': rmse_overall
        }
    }
    os.makedirs('run_2_results', exist_ok=True)
    out_path = f'run_2_results/result_{task_id}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Task {task_id} Complete -> Val MAE: {mae_overall:.4f} | Val RMSE: {rmse_overall:.4f}")



if __name__ == '__main__':
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with open('v1_params.json', 'r') as f:
        params = json.load(f)
    evaluate_model(params[task_id], task_id)
