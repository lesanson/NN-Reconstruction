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
from preprocessing import eos_load_and_preprocess
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
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, W):
        return W*W + self.eps

# ---------------------------
# EoS Network (ρ -> p_scaled)
# ---------------------------

class EoSNetwork(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        self.act = nn.ELU()

        self.conv1 = nn.Conv1d(input_channels, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.conv4 = nn.Conv1d(64, output_channels, 1)

        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            # 1) init the *unconstrained* weight
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.normal_(m.bias, mean=0.0, std=0.05)

            # 2) then register the non-negative parametrization
            parametrize.register_parametrization(m, "weight", NonNegative())


    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, C, L)

        # lift 1 -> 64 (no residual)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        # projection
        x = torch.sigmoid(self.conv4(x))
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
    # Warmup-ish / coarse search
    if epoch < 200:
        return 3e-3
    # Still exploring, but smaller
    elif epoch < 600:
        return 1e-3
    # Fine-tuning
    elif epoch < 1200:
        return 7.5e-4
    # Very fine tuning
    elif epoch < 20000:
        return 7e-4
    elif epoch < 30000:
        return 6e-4
    else:
        return 5e-4


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
    rho_scaled, M_obs, R_obs, dM, dR = eos_load_and_preprocess(
        mr_csv, eos_csv, Np=Np, device=device
    )

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

# ---- 3. Main script ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EoS Network with frozen TOV WaveNet (PyTorch)")
    parser.add_argument("--tov", type=str, default="models/tov_solver.pt", help="Path to frozen TOV model weights")
    parser.add_argument("--mr_csv", type=str, default="data/sample_mr.csv", help="Observed MR CSV")
    parser.add_argument("--eos_csv", type=str, default="data/sample_eos.csv", help="True EoS CSV for densities")
    parser.add_argument("--save", type=str, default="models/eos_solver.pt", help="Where to save EoS network weights")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--Np", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_eos(
        tov_weights_path=args.tov,
        mr_csv=args.mr_csv,
        eos_csv=args.eos_csv,
        save_path=args.save,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        Np=args.Np,
        seed=args.seed,
    )
