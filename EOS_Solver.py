# train_eos_solver.py
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from TOV_Solver import WaveNetTOV
from preprocessing import eos_load_and_preprocess, eos_realistic_load_and_preprocess
import torch.nn.utils.parametrize as parametrize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# Helper functions
# ---------------------------

class NonNegative(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, W):
        return F.softplus(W) + self.eps
    
# ---------------------------
# EoS Network (ρ -> p_scaled)
# ---------------------------

class EoSNetwork(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        self.act = nn.ELU()

        self.conv1 = nn.Conv1d(input_channels, 128, 1, padding='same')
        self.conv2 = nn.Conv1d(128, 128, 1,padding='same')
        self.conv3 = nn.Conv1d(128, output_channels, 1, padding='same')

        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.zeros_(m.bias)

            #parametrize.register_parametrization(m, "weight", NonNegative())

            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.normal_(m.bias, mean=0.0, std=0.05)


    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, C, L)

        # lift 1 -> 64 (no residual)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))

        # projection
        x = torch.sigmoid(self.conv3(x))
        return x.permute(0, 2, 1)

# ---------------------------
# χ² loss (Eq. 4.1/4.2)
# ---------------------------
def chi2_loss(M_obs, R_obs, dM, dR):
    """
    Exact paper loss with gradient preservation
    """
    def loss(mr_pred):
        M_seq, R_seq = mr_pred[0,:,0], mr_pred[0,:,1]
        total_loss = 0.0
        
        # Exact paper computation
        for i in range(len(M_obs)):
            distances = ((M_seq - M_obs[i])/(dM[i] + 1e-6))**2 + \
                       ((R_seq - R_obs[i])/(dR[i] + 1e-6))**2
            
            min_idx = torch.argmin(distances)
            
            chi2_term = ((M_seq[min_idx] - M_obs[i])/(dM[i] + 1e-6))**2 + \
                       ((R_seq[min_idx] - R_obs[i])/(dR[i] + 1e-6))**2
            total_loss += chi2_term

        return total_loss
    return loss

    

def get_lr_schedule(epoch):
    if epoch < 2500:
        return 4e-3
    elif epoch < 3000:
        return 2e-3
    elif epoch < 4500:
        return 1e-3
    elif epoch < 5000:
        return 3e-3
    elif epoch < 7000:
        return 2e-3
    else:
        return 3e-3


# ---- 2. Training function ----
def train_eos(
    tov_weights_path: str = "models/tov_solver.pt",
    mr_csv: str = "data/sample_mr.csv",
    eos_csv: str = "data/sample_eos.csv",
    save_path: str = "models/eos_solver.pt",
    epochs: int = 7100,
    lr: float = 1e-4,
    weight_decay: float = 1e-8,
    Np: int = 32,
    seed: int = 42,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # -------------------- FROZEN TOV SOLVER --------------------
    tov = WaveNetTOV()
    tov.load_state_dict(torch.load(tov_weights_path, map_location=device))  # or map_location=device
    tov = tov.to(device)
    for p in tov.parameters():
        p.requires_grad = False
    tov.eval()

    # -------------------- TRAINABLE NETWORK --------------------
    eos_net = EoSNetwork().to(device)

    # -------------------- SPLIT DATA --------------------
    rho_scaled, P_scaled, M_obs, R_obs, dM, dR = eos_load_and_preprocess()

    rho_scaled_t = torch.tensor(rho_scaled, dtype=torch.float32, device=device).unsqueeze(0)
    M_obs = torch.tensor(M_obs, dtype=torch.float32, device=device)
    R_obs = torch.tensor(R_obs, dtype=torch.float32, device=device)
    dM = torch.tensor(dM, dtype=torch.float32, device=device)
    dR = torch.tensor(dR, dtype=torch.float32, device=device)

    # -------------------- LOSS + OPTIMIZER --------------------
    optimizer = torch.optim.AdamW(eos_net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = chi2_loss(M_obs, R_obs, dM, dR)
    best_loss = float("inf")
    best_state = None
    patience = 0

    # -------------------- TRAINING LOOP --------------------
    for epoch in range(epochs):

        current_lr = get_lr_schedule(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # --- Forward + backward ---
        eos_net.train()
        optimizer.zero_grad()
        p_scaled_pred = eos_net(rho_scaled_t)
        mr_pred = tov(p_scaled_pred)
        loss = criterion(mr_pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eos_net.parameters(), max_norm=1.0)
        optimizer.step()

        if loss < best_loss:
            patience = 0
            best_loss = loss
            best_state = {k: v.detach().cpu().clone() for k, v in eos_net.state_dict().items()}
        else:
            patience+=1

        print(f"[{epoch:5d}/{epochs}] LR={current_lr:.6f} | chi²={loss:.6f} | best={best_loss:.6f}") 
    
    print(f"[{epoch:5d}/{epochs}] | chi²={loss:.6f} | best={best_loss:.6f}")

    # -------------------- SAVE --------------------
    if best_state is not None:
        eos_net.load_state_dict(best_state)
    torch.save(eos_net.state_dict(), save_path)
    print(f"Saved EoS network weights to: {save_path}")

# this is for the realistic case
def train_eos_realistic(
    tov_weights_path: str = "models/tov_solver.pt",
    mr_csv: str = "data/sample_mr.csv",
    eos_csv: str = "data/sample_eos.csv",
    save_path: str = "results/eos_realistic.npz",  # now used for predictions
    epochs: int = 7100,
    lr: float = 1e-4,
    weight_decay: float = 1e-8,
    Np: int = 32,
    seed: int = 42,
    # --- early stopping params ---
    patience: int = 100,      # how many epochs with no improvement before stopping
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- 1. Frozen TOV ----
    tov = WaveNetTOV().to(device)
    tov.load_state_dict(torch.load(tov_weights_path, map_location=device))
    for p in tov.parameters():
        p.requires_grad = False
    tov.eval()

    Nsamples = 200

    # ---- 2. Realistic data: one ρ-grid + many noisy M-R curves ----
    (
        rho_scaled,
        M_samples,
        R_samples,
        dM_samples,
        dR_samples,
        M_MAX,
        R_MIN,
        R_MAX,
    ) = eos_realistic_load_and_preprocess(
        mr_csv=mr_csv,
        eos_csv=eos_csv,
        Np=Np,
        Nsamples=Nsamples,
        device=device,
    )

    # storage for predictions
    M_pred_samples = []
    R_pred_samples = []
    p_pred_samples = []

    rho_t = torch.tensor(
        rho_scaled, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, Np, 1)
    
    # ---- 3. Loop over each noisy MR dataset ----
    for k in range(Nsamples):
        print(f"\n=== Training on realistic sample {k+1}/{Nsamples} ===")

        # observed M-R and uncertainties for this sample (already scaled)
        M_obs = M_samples[k]      # (Nobs,)
        R_obs = R_samples[k]      # (Nobs,)
        dM    = dM_samples[k]
        dR    = dR_samples[k]

        eos_net = EoSNetwork().to(device)

        M_obs_t = torch.tensor(M_obs, dtype=torch.float32, device=device)
        R_obs_t = torch.tensor(R_obs, dtype=torch.float32, device=device)
        dM_t    = torch.tensor(dM,    dtype=torch.float32, device=device)
        dR_t    = torch.tensor(dR,    dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(eos_net.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float("inf")
        best_state = None

        # --- early stopping bookkeeping ---
        no_improve_epochs = 0

        for epoch in range(epochs):
            #tau0 = 1e-2
            tau = 1e-8
            criterion = chi2_loss(M_obs_t, R_obs_t, dM_t, dR_t, tau=tau)
            current_lr = get_lr_schedule(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            eos_net.train()
            optimizer.zero_grad()

            p_scaled_pred = eos_net(rho_t)      # (1, Np, 1)
            mr_pred = tov(p_scaled_pred)        # (1, Nseq, 2)

            loss = criterion(mr_pred) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(eos_net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()

            # --- track best model + early stopping ---
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in eos_net.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

           
            print(
                f"[sample {k+1:03d}] "
                f"epoch {epoch:5d}/{epochs} | "
                f"LR={current_lr:.6f} | chi²={loss_val:.6f} | "
                f"best={best_loss:.6f} | "
            )

            # --- EARLY STOP if no improvement for `patience` epochs ---
            if no_improve_epochs >= patience:
                print(
                    f"Early stopping on sample {k+1}: "
                    f"no improvement for {patience} epochs (best chi²={best_loss:.6f})."
                )
                break

        # ---- restore best weights ----
        if best_state is not None:
            eos_net.load_state_dict(best_state)
        
        # ---- final prediction for this sample ----
        if best_loss < 50: # check if it converged 
            eos_net.eval()
            with torch.no_grad():
                p_scaled_pred = eos_net(rho_t)       # (1, Np, 1)
                mr_pred       = tov(p_scaled_pred)   # (1, Nseq, 2)

            # Convert to NumPy
            p_np = p_scaled_pred.detach().cpu().numpy().squeeze()    # (Np,)
            mr_np = mr_pred.detach().cpu().numpy().squeeze()         # (Nseq, 2)

            # Unscale M and R
            M_scaled_pred = mr_np[:, 0]
            R_scaled_pred = mr_np[:, 1]

            mass_pred   = M_scaled_pred * M_MAX
            radius_pred = R_scaled_pred * (R_MAX - R_MIN) + R_MIN

            # store
            p_pred_samples.append(p_np)        # (Np,)
            M_pred_samples.append(mass_pred)   # (Nseq,)
            R_pred_samples.append(radius_pred) # (Nseq,)

    # ---- 4. Stack everything ----
    p_pred_samples = np.stack(p_pred_samples, axis=0)  # (Nsamples, Np)
    M_pred_samples = np.stack(M_pred_samples, axis=0)  # (Nsamples, Nseq)
    R_pred_samples = np.stack(R_pred_samples, axis=0)  # (Nsamples, Nseq)

    # ---- 5. Save or just return ----
    if save_path is not None:
        np.savez(
            save_path,
            M_samples=M_samples,
            R_samples=R_samples,
            dM_samples=dM_samples,
            dR_samples=dR_samples,
            rho_scaled=rho_scaled,      
            p_pred_samples=p_pred_samples,
            M_pred_samples=M_pred_samples,
            R_pred_samples=R_pred_samples,
            M_MAX=M_MAX,
            R_MIN=R_MIN,
            R_MAX=R_MAX,
        )
        print(f"Saved realistic predictions to {save_path}")

    return (
        M_samples,
        R_samples,
        dM_samples,
        dR_samples,
        M_MAX,
        R_MIN,
        R_MAX,
        rho_scaled, 
        M_pred_samples, 
        R_pred_samples, 
        p_pred_samples,
    )

# ---- 3. Main script ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EoS Network with frozen TOV WaveNet (PyTorch)")
    parser.add_argument("--tov", type=str, default="models/tov_solver.pt", help="Path to frozen TOV model weights")
    parser.add_argument("--mr_csv", type=str, default="data/sample_mr.csv", help="Observed MR CSV")
    parser.add_argument("--eos_csv", type=str, default="data/sample_eos.csv", help="True EoS CSV for densities")
    parser.add_argument("--save_model", type=str, default="models/eos_solver.pt", help="Where to save EoS network weights / results")
    parser.add_argument("--epochs", type=int, default=7100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--Np", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--realistic", action="store_true", help="Use realistic noisy MR training")
    args = parser.parse_args()

    if not args.realistic:
        # original synthetic / clean case
        train_eos(
            tov_weights_path=args.tov,
            mr_csv=args.mr_csv,
            eos_csv=args.eos_csv,
            save_path=args.save_model,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            Np=args.Np,
            seed=args.seed,
        )
    else:
        # realistic noisy MR case
        train_eos_realistic(
            tov_weights_path=args.tov,
            mr_csv=args.mr_csv,
            eos_csv=args.eos_csv,
            save_path="data/realistic/realistic_predictions.npz",   # here: npz with predictions
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            Np=args.Np,
            seed=args.seed,
        )
