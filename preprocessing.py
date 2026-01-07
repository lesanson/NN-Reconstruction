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
        counts = [14, 12, 12, 12, 14]  # Total = 64

        rho_grid = []
        for i, (lo, hi, n) in enumerate(zip(seg_bounds[:-1], seg_bounds[1:], counts)):
            seg = np.logspace(
                np.log10(lo),
                np.log10(hi),
                n,
                endpoint=(i == len(counts) - 1),
            )
            rho_grid.extend(seg)
        rho_grid = np.asarray(rho_grid, dtype=float)

        out["rho"] = rho_grid
        out["ID"] = ID_val
        out["model"] = model_val

        # --- Interpolated p(ρ) onto rho_grid ---
        if "p" in df.columns and np.issubdtype(df["p"].dtype, np.number):
            rho_orig = df["rho"].to_numpy(dtype=float)
            p_orig = df["p"].to_numpy(dtype=float)

            # Remove NaNs/infs
            mask = np.isfinite(rho_orig) & np.isfinite(p_orig)
            rho_orig = rho_orig[mask]
            p_orig = p_orig[mask]

            if rho_orig.size < 2:
                # Not enough points to interpolate
                out["p"] = np.full_like(rho_grid, np.nan, dtype=float)
                return pd.DataFrame(out)

            # Interpolate in log(rho) since rho_grid is log-spaced
            # np.interp does linear interpolation and clamps outside range to endpoints.
            log_rho_orig = np.log(rho_orig)
            log_rho_grid = np.log(rho_grid)

            p_grid = np.interp(log_rho_grid, log_rho_orig, p_orig)

            out["p"] = p_grid
        else:
            out["p"] = np.full_like(rho_grid, np.nan, dtype=float)

        return pd.DataFrame(out)

    # --------------------------------------
    # M–R branch
    # --------------------------------------
    elif column == "M":
        M = df["M"].to_numpy(dtype=float)
        R = df["R"].to_numpy(dtype=float)

        mask = R <= 16.0
        M = M[mask]
        R = R[mask]

        n_points = M.size
        if n_points == 0:
            raise ValueError(f"No valid points for (ID {ID_val}, model {model_val}) after R <= 16 mask.")

        # Parameter along the ORIGINAL sequence (preserves MR relation)
        t = np.arange(n_points, dtype=float)

        if n_points >= Np:
            # no interpolation needed: take Np indices without repeats
            idxs = np.round(np.linspace(0, n_points - 1, Np)).astype(int)
            # (with n_points>=Np this is usually unique; enforce just in case)
            idxs = np.clip(idxs, 0, n_points - 1)
            # if duplicates still happen, switch to interpolation instead:
            if np.unique(idxs).size < Np:
                t_grid = np.linspace(0, n_points - 1, Np)
                M_sel = np.interp(t_grid, t, M)
                R_sel = np.interp(t_grid, t, R)
            else:
                M_sel = M[idxs]
                R_sel = R[idxs]
        else:
            # n_points < Np: interpolate LOCALLY along the sequence index
            t_grid = np.linspace(0, n_points - 1, Np)
            M_sel = np.interp(t_grid, t, M)
            R_sel = np.interp(t_grid, t, R)

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
    input_df = pd.read_csv(input_csv).query("model == 'RMFNL'")
    output_df = pd.read_csv(output_csv).query("model == 'RMFNL'")

    # --- Scaling Constants ---
    M_MAX = output_df["M"].max()
    np.save("scalers/M_MAX.npy", M_MAX)
    R_MAX = 16.0
    R_MIN = output_df["R"].min()
    np.save("scalers/R_MIN.npy", R_MIN)


    # --- Identify unique (ID, model) pairs in input ---
    input_pairs = input_df[["ID", "model"]].drop_duplicates()
    sample_pairs = input_pairs

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

    # --- Prepare arrays ---
    unique_pairs = output_interp[["ID", "model"]].drop_duplicates().values
    N = len(unique_pairs)

    X = np.zeros((N, Np, 2), dtype=np.float32)
    Y_Scaled = np.zeros((N, Np, 2), dtype=np.float32)

    # --- Build dataset ---
    for i, (ID, model) in enumerate(unique_pairs):
        eos_grp = input_interp.loc[(input_interp["ID"] == ID) & (input_interp["model"] == model)]
        tov_grp = output_interp.loc[(output_interp["ID"] == ID) & (output_interp["model"] == model)]

        # log10(p)
        p_vals = eos_grp["p"].to_numpy()
        X[i, :, 1] = np.log10(np.clip(p_vals, 1e-30, None))

        # Normalized M and R
        M = tov_grp["M"].to_numpy() / M_MAX
        R = (tov_grp["R"].to_numpy() - R_MIN) / (R_MAX - R_MIN)

        Y_Scaled[i, :, 0] = M
        Y_Scaled[i, :, 1] = R

        rho_phys = input_df["rho"].loc[(input_df["ID"] == ID) & (input_df["model"] == model)].to_numpy(dtype=np.float32)
        rho_min, rho_max = float(rho_phys.min()), float(rho_phys.max())

        rho_phys_targets = np.linspace(rho_min, rho_max, Np, dtype=np.float32)
        rho_scaled = 0.1 * (rho_phys_targets - rho_min) / (rho_max - rho_min)

        X[i, :, 0] = rho_scaled

        if i % 1000 == 0:
            print(f"Processed {i}/{N}")

    # --- Scale X globally ---
    X_max = np.max(X[:, :, 1])
    X_scaled = X.copy()
    X_scaled[:, :, 1] = X[:, :, 1] / X_max

    # --- Save scalers ---
    os.makedirs("scalers", exist_ok=True)
    np.save("scalers/X_max.npy", X_max)

    # --- Save dataset ---
    os.makedirs("data", exist_ok=True)
    np.save("data/X_scaled.npy", X_scaled)
    np.save("data/Y_scaled.npy", Y_Scaled)

    return X_scaled, Y_Scaled




def eos_load_and_preprocess():
    # ---------------- load ----------------
    X_scaled = np.load("data/X_test.npy")   # shape (N, Np, 2)
    Y_scaled = np.load("data/Y_test.npy")   # shape (N, Np, 2)

    M_MAX = np.load("scalers/M_MAX.npy")

    # Use sample 0 (you are doing inference for one EOS here)
    M_scaled_full = Y_scaled[0, :, 0].astype(np.float32)   # (Np,)
    R_scaled_full = Y_scaled[0, :, 1].astype(np.float32)   # (Np,)

    # Keep rho and P exactly as stored (full length)
    rho_scaled = X_scaled[0, :, 0].astype(np.float32)      # (Np,)
    P_scaled   = X_scaled[0, :, 1].astype(np.float32)      # (Np,)

    # ---------------- build 11-point MR curve (1.0 Msun -> M_max) ----------------
    # Convert scaled mass back to physical mass for thresholding and M_max detection
    M_phys_full = M_scaled_full * M_MAX

    # Mask for physical mass > 1.0 (Msun if your M is in Msun)
    mask = M_phys_full > 1.0
    if not np.any(mask):
        raise ValueError("No points satisfy M_phys > 1.0. Check scaling (M_MAX) and units.")

    M_m     = M_scaled_full[mask]
    R_m     = R_scaled_full[mask]
    Mphys_m = M_phys_full[mask]

    # Index closest to 1.0 Msun within the masked region
    i0 = int(np.argmin(np.abs(Mphys_m - 1.0)))

    # Index of maximum mass (peak of the MR curve) within the masked region
    imax = int(np.argmax(Mphys_m))

    # Ensure we have a valid stable branch segment from ~1.0 Msun up to M_max
    if i0 >= imax:
        raise ValueError(
            "Invalid MR ordering: the point closest to 1.0 Msun occurs after the maximum mass point. "
            "Check that your MR curve is ordered by central density and that scaling is consistent."
        )

    # Keep only the stable branch: from ~1.0 Msun up to (and including) M_max
    M_m = M_m[i0:imax + 1]
    R_m = R_m[i0:imax + 1]

    if M_m.size < 2:
        raise ValueError("Not enough points after trimming to build 11-point curve.")

    # Arc-length parameter along the curve in (M_scaled, R_scaled) space
    dM_seg = np.diff(M_m)
    dR_seg = np.diff(R_m)
    s = np.concatenate(
        ([0.0], np.cumsum(np.sqrt(dM_seg * dM_seg + dR_seg * dR_seg)).astype(np.float32))
    )

    if float(s[-1]) <= 0.0:
        raise ValueError("Degenerate MR segment (zero arc-length). Check input data ordering/values.")

    # Sample 11 points uniformly in arc-length from start (~1 Msun) to end (M_max)
    s_targets = np.linspace(0.0, float(s[-1]), 11, dtype=np.float32)
    M_11 = np.interp(s_targets, s, M_m).astype(np.float32)
    R_11 = np.interp(s_targets, s, R_m).astype(np.float32)

    # dM, dR for the 11-point curve (placeholders as you had)
    dM = np.ones_like(M_11, dtype=np.float32)
    dR = np.ones_like(R_11, dtype=np.float32)

    # ---------------- debug prints ----------------
    print("M_11 (scaled):", M_11)
    print("R_11 (scaled):", R_11)
    print("rho_scaled (full):", rho_scaled)

    # Return rho as (Np,1) like you wanted, keep P as (Np,)
    return rho_scaled.reshape(-1, 1), P_scaled, M_11, R_11, dM, dR

def eos_realistic_load_and_preprocess(
    mr_csv: str,
    eos_csv: str,
    Np: int = 32,
    Nsamples: int = 500,
    device=None
):
    observed_mr = pd.read_csv(mr_csv)
    observed_eos = pd.read_csv(eos_csv)

    # ---------------- global scaling constants (must match TOV training) ----------------
    # These should be computed the same way as in tov_load_and_preprocess: ID != 19248
    M_MAX = observed_mr.query("ID != 19248")["M"].max()
    R_MAX = 16.0
    R_MIN = observed_mr.query("ID != 19248")["R"].min()

    # ---------------- M–R for the "observed" curve (ID 19248, RMFNL) ----------------
    mr_19248 = observed_mr.query("ID == 19248").query("model == 'RMFNL'")

    # assume rows are ordered along the physical sequence (central density)
    M_values = mr_19248["M"].to_numpy(dtype=np.float32)
    R_values = mr_19248["R"].to_numpy(dtype=np.float32)

    # stable branch
    imax = int(np.argmax(M_values))
    M_stable = M_values[:imax + 1].astype(np.float32)
    R_stable = R_values[:imax + 1].astype(np.float32)

    # keep only M >= 1 on the stable branch
    mask = M_stable >= 1.0
    M_ge1 = M_stable[mask]
    R_ge1 = R_stable[mask]

    if M_ge1.size == 0:
        raise ValueError("Stable branch has no points with M >= 1.0")

    Mmax_phys = float(M_ge1.max())

    # 11 target masses
    M_targets = np.linspace(1.0, Mmax_phys, 100, dtype=np.float32)

    # nearest-neighbor selection (no interpolation)
    idxs = [int(np.argmin(np.abs(M_ge1 - mt))) for mt in M_targets]
    idxs = np.unique(idxs)  # remove duplicates if targets hit same point

    # ensure exactly 11 points (pad or trim)
    if idxs.size < 11:
        idxs = np.pad(idxs, (0, 11 - idxs.size), mode="edge")
    elif idxs.size > 11:
        # keep them spread out
        idxs = idxs[np.linspace(0, idxs.size - 1, 11).round().astype(int)]

    # final observed points (from data only)
    M_obs = M_ge1[idxs]
    R_obs = R_ge1[idxs]

    # ---------------- scale to match training ----------------
    M_scaled = M_obs / M_MAX
    R_scaled = (R_obs - R_MIN) / (R_MAX - R_MIN)

    # 10% relative uncertainties (careful: relative-to-scaled is ok if that’s what your likelihood expects)
    sigma_M = 0.1 * M_scaled
    sigma_R = 0.1 * R_scaled

    # ---------------- sample noisy curves ----------------
    M_samples = np.random.normal(M_scaled, sigma_M, size=(Nsamples, 11)).astype(np.float32)
    R_samples = np.random.normal(R_scaled, sigma_R, size=(Nsamples, 11)).astype(np.float32)

    dM_samples = np.broadcast_to(sigma_M, M_samples.shape).astype(np.float32)
    dR_samples = np.broadcast_to(sigma_R, R_samples.shape).astype(np.float32)

    # ---------------- ρ-grid from observed EoS ----------------
    eos_19248 = (
        observed_eos
        .query("ID == 19248")
        .query("model == 'RMFNL'")
        .sort_values("rho")
    )

    rho_phys = eos_19248["rho"].values.astype(np.float32)
    rho_min, rho_max = rho_phys.min(), rho_phys.max()

    rho_phys_targets = np.linspace(rho_min, rho_max, Np, dtype=np.float32)
    rho_scaled = 0.1 * (rho_phys_targets - rho_min) / (rho_max - rho_min)

    # debug prints if you still want them
    print("M_scaled (true):", M_scaled)
    print("R_scaled (true):", R_scaled)
    print("rho_scaled:", rho_scaled)
    print("M_samples shape:", M_samples.shape)  # (500, 11)
    print("R_samples shape:", R_samples.shape)  # (500, 11)

    # final return:
    #   rho_scaled:          (Np, 1)
    #   M_samples:           (Nsamples, 11)
    #   R_samples:           (Nsamples, 11)
    #   dM_samples, dR_samples: same shapes
    return rho_scaled.reshape(-1, 1), M_samples, R_samples, dM_samples, dR_samples,  M_MAX, R_MIN, R_MAX,
    

