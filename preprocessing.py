
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

"""
15033 is the index selected to represent our observational data.
"""

def _resample_group(df: pd.DataFrame, column: str, Np: int = 32) -> pd.DataFrame:
    df = df.sort_values(column).reset_index(drop=True)
    if column == 'rho':
        target = np.linspace(1, 7.4, Np)
    else:
        target = np.linspace(df[column].min(), df[column].max(), Np)
    chosen = []
    df_temp = df.copy()
    for t in target:
        proposed_idx = (df_temp[column] - t).abs().idxmin()
        chosen.append(proposed_idx)
        df_temp = df_temp.drop(proposed_idx)
    resampled = df.iloc[chosen].reset_index(drop=True)
    resampled.index = range(len(resampled))
    return resampled.copy(deep=True)


def tov_load_and_preprocess(input_csv: str, output_csv: str, Np: int = 32) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Load CSVs, resample per-ID and return scaled X and Y arrays plus scalers.

    Returns:
        X_scaled: (N, Np, 1)
        Y_scaled: (N, Np, 2) where channels are [M, R]
        p_scaler, mr_scaler
    """

    input_df = pd.read_csv(input_csv)
    input_df = input_df.query("ID != 15033")

    output_df = pd.read_csv(output_csv)
    output_df = output_df.query("ID != 15033")

    input_df['rho'] = input_df['rho'] / 0.16
    if Np <= 32:
        input_df = input_df.groupby('ID', group_keys=False).apply(_resample_group, column='rho', Np=Np)
        output_df = output_df.groupby('ID', group_keys=False).apply(_resample_group, column='M', Np=Np)

    unique_ids = output_df['ID'].unique()
    N = len(unique_ids)
    X = np.zeros((N, Np, 1), dtype=np.float32)
    Y = np.zeros((N, Np, 2), dtype=np.float32)
    for i, ID in enumerate(unique_ids):
        df_in = input_df[input_df['ID'] == ID].sort_values('rho').reset_index(drop=True)
        df_out = output_df[output_df['ID'] == ID].sort_values('M').reset_index(drop=True)
        X[i, :, 0] = np.log10(np.clip(df_in['p'].values, 1e-30, None))
        Y[i, :, 0] = df_out['M'].values
        Y[i, :, 1] = df_out['R'].values

    # Global normalization across all EOSs
    p_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X.reshape(-1, 1))
    mr_scaler = MinMaxScaler(feature_range=(0, 1)).fit(Y.reshape(-1, 2))

    X_scaled = p_scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    Y_scaled = mr_scaler.fit_transform(Y.reshape(-1, 2)).reshape(Y.shape)

    # Save global scalers for inference
    joblib.dump(p_scaler, 'scalers/p_scaler.pkl')
    joblib.dump(mr_scaler, 'scalers/mr_scaler.pkl')

    print("X_scaled mean/std:", np.mean(X_scaled), np.std(X_scaled))
    print("Y_scaled M std:", np.std(Y_scaled[...,0]))
    print("Y_scaled R std:", np.std(Y_scaled[...,1]))

    print("X_scaled shape:", X_scaled.shape)
    print("Y_scaled shape:", Y_scaled.shape)

    return X_scaled, Y_scaled