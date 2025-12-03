from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
"""
19248 is the index selected to represent our observational data.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d


def _resample_group(df: pd.DataFrame, column: str, Np: int = 32) -> pd.DataFrame:
    """
    Resampling for EOS or TOV groups WITHOUT interpolation:
      • EoS (ρ–p): For each target ρ in a fixed log-spaced grid,
        pick the closest existing (ρ, p) point in the group.
      • M–R: pick Np UNIQUE points, sorted by increasing radius R.

    Output always has exactly Np rows.
    """

    ID_val = df["ID"].iloc[0]
    model_val = df["model"].iloc[0] if "model" in df.columns else "nomodel"
    out = {}

    # --------------------------------------
    # ρ–p branch
    # --------------------------------------
    if column == "rho":
        # --- Fixed 5 log-spaced segments per the paper ---
        seg_bounds = np.array([1.0, 1.4, 2.2, 3.3, 4.9, 7.4], dtype=float)
        counts = [7, 6, 6, 6, 7]  # Total = 32

        rho_grid = []
        for i, (lo, hi, n) in enumerate(zip(seg_bounds[:-1], seg_bounds[1:], counts)):
            seg = np.logspace(
                np.log10(lo),
                np.log10(hi),
                n,
                endpoint=(i == len(counts) - 1),
            )
            rho_grid.extend(seg)
        rho_grid = np.array(rho_grid)

        out["rho"] = rho_grid
        out["ID"] = ID_val
        out["model"] = model_val

        # --- Nearest-neighbor selection of p(ρ) ---
        if "p" in df.columns and np.issubdtype(df["p"].dtype, np.number):
            rho_orig = df["rho"].to_numpy()
            p_orig = df["p"].to_numpy()
            p_grid = np.empty_like(rho_grid)

            for k, rho_target in enumerate(rho_grid):
                j = np.argmin(np.abs(rho_orig - rho_target))
                p_grid[k] = p_orig[j]

            out["p"] = p_grid
        else:
            out["p"] = np.full_like(rho_grid, np.nan)

    # --------------------------------------
    # M–R branch
    # --------------------------------------
    elif column == "M":

        # Extract arrays
        M = df["M"].to_numpy(dtype=float)
        R = df["R"].to_numpy(dtype=float)

        # --- Physical cuts: ONLY on radius ---
        mask = (R <= 16.0)
        M = M[mask]
        R = R[mask]

        # Nothing left?
        n_points = len(M)
        if n_points == 0:
            raise ValueError(f"No valid points for (ID {ID_val}, model {model_val}) after R <= 16 mask.")

        # --- Ensure consistent ordering ---
        # WARNING: original TOV output may be descending or ascending!
        # Let's sort by R ascending.
        # --- Select Np evenly spaced indices (no interpolation) ---
        idxs = np.linspace(0, n_points - 1, Np).round().astype(int)
        idxs = np.unique(idxs)  # avoid repeats

        # If duplicates removed, pad to size Np
        if len(idxs) < Np:
            idxs = np.pad(idxs, (0, Np - len(idxs)), mode="edge")

        # Pick those points
        R_sel = R[idxs]
        M_sel = M[idxs]

        # --- Output ---
        out["ID"] = ID_val
        out["model"] = model_val
        out["R"] = R_sel
        out["M"] = M_sel

    else:
        raise ValueError(f"Unsupported column '{column}'. Use 'rho' or 'M'.")

    resampled = pd.DataFrame(out)
    assert len(resampled) == Np, f"{column}: expected {Np}, got {len(resampled)}"
    return resampled



def tov_load_and_preprocess(input_csv: str, output_csv: str, Np: int = 32, seed: int = 42):
    """
    Load EoS & TOV tables, interpolate to fixed grids,
    and do GLOBAL normalization:
        M_scaled = M / observed_mr["M"].max()
        R_scaled = R / 16.0
    Saves:
        X_max, M_MAX, R_MAX
    """

    # --- Load and filter data ---
    input_df = pd.read_csv(input_csv).query("ID != 19248")
    output_df = pd.read_csv(output_csv).query("ID != 19248")

    # --- Identify unique (ID, model) pairs in input ---
    input_pairs = input_df[["ID", "model"]].drop_duplicates()
    sample_pairs = input_pairs.sample(n=59997, random_state=seed)

    # --- Filter both datasets ---
    input_df = input_df.merge(sample_pairs, on=["ID", "model"])
    output_df = output_df.merge(sample_pairs, on=["ID", "model"])

    # --- Keep only pairs present in both files ---
    common_ids = (
        input_df[["ID", "model"]].drop_duplicates()
        .merge(output_df[["ID", "model"]].drop_duplicates(), on=["ID", "model"])
    )
    input_df = input_df.merge(common_ids, on=["ID", "model"])
    output_df = output_df.merge(common_ids, on=["ID", "model"])

    # --- Normalize density (ρ / ρ_sat) ---
    input_df["rho"] = input_df["rho"] / 0.16

    # --- Resample EoS and MR curves ---
    input_interp = input_df.groupby(["ID", "model"], group_keys=False).apply(
        _resample_group, column="rho", Np=Np, include_groups=True
    )
    output_interp = output_df.groupby(["ID", "model"], group_keys=False).apply(
        _resample_group, column="M", Np=Np, include_groups=True
    )

    print("Done with interpolation")

    # --- Prepare arrays ---
    unique_pairs = output_interp[["ID", "model"]].drop_duplicates().values
    N = len(unique_pairs)

    X = np.zeros((N, Np, 1), dtype=np.float32)
    Y_Scaled = np.zeros((N, Np, 2), dtype=np.float32)

    # --- GLOBAL SCALING CONSTANTS ---
    M_MAX = output_df["M"].max()
    R_MAX = 16.0
    R_MIN = output_df["R"].min()

    # --- Build dataset ---
    for i, (ID, model) in enumerate(unique_pairs):
        eos_grp = input_interp.loc[(input_interp["ID"] == ID) & (input_interp["model"] == model)]
        tov_grp = output_interp.loc[(output_interp["ID"] == ID) & (output_interp["model"] == model)]

        # log10(p)
        p_vals = eos_grp["p"].to_numpy()
        X[i, :, 0] = np.log10(np.clip(p_vals, 1e-30, None))

        # Normalized M and R
        M = tov_grp["M"].to_numpy() / M_MAX
        R = (tov_grp["R"].to_numpy() - R_MIN) / (R_MAX - R_MIN)

        Y_Scaled[i, :, 0] = M
        Y_Scaled[i, :, 1] = R

        if i % 1000 == 0:
            print(f"Processed {i}/{N}")

    # --- Scale X globally ---
    X_max = np.max(X)
    X_scaled = X / X_max

    # --- Save scalers ---
    os.makedirs("scalers", exist_ok=True)

    # --- Save dataset ---
    os.makedirs("data", exist_ok=True)
    np.save("data/X_scaled.npy", X_scaled)
    np.save("data/Y_scaled.npy", Y_Scaled)

    return X_scaled, Y_Scaled




def eos_load_and_preprocess(
    mr_csv: str,
    eos_csv: str,
    Np: int = 32,
    device=None
):
    observed_mr = pd.read_csv(mr_csv)
    observed_eos = pd.read_csv(eos_csv)

    # ---------------- M–R preprocessing (same as before) ----------------
    M_MAX = observed_mr["M"].max()
    R_MAX = 16.0
    R_MIN = observed_mr["R"].min()

    observed_mr = observed_mr.query("ID == 19248").query("model == 'RMFNL'")
    M_targets = np.linspace(1.0, min(2.25, observed_mr["M"].max()), 11)

    M_values = observed_mr["M"].values
    R_values = observed_mr["R"].values

    R_obs = np.array([R_values[np.argmin(np.abs(M_values - m))] for m in M_targets])

    M_scaled = M_targets / M_MAX
    R_scaled = (R_obs - R_MIN) / (R_MAX - R_MIN)
    dM = np.ones_like(M_scaled)
    dR = np.ones_like(R_scaled)

    # ---------------- ρ-grid from observed_eos ----------------
    eos_19248 = observed_eos.query("ID == 19248").query("model == 'RMFNL'").sort_values("rho")

    rho_phys = eos_19248["rho"].values.astype(np.float32)
    rho_min, rho_max = rho_phys.min(), rho_phys.max()

    # 32 linearly spaced physical densities between rho_min and rho_max
    rho_phys_targets = np.linspace(rho_min, rho_max, Np, dtype=np.float32)

    rho_scaled = 0.1 * (rho_phys_targets - rho_min) / (rho_max - rho_min)

    print("M_scaled:", M_scaled)
    print("R_scaled:", R_scaled)
    print("rho_scaled:", rho_scaled)

    return rho_scaled.reshape(-1, 1), M_scaled, R_scaled, dM, dR
