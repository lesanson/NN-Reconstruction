import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from preprocessing import tov_load_and_preprocess
import time

"""
python TOV_Solver.py 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- 1. Define WaveNet model ----
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation

        # unconstrained parameters
        self.weight_raw = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.dilation = dilation

        nn.init.xavier_uniform_(self.weight_raw)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))

        # enforce non-negativity
        weight = F.leaky_relu(self.weight_raw)

        return F.conv1d(
            x,
            weight,
            bias=self.bias,
            dilation=self.dilation
        )


class WaveNetTOV(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, filters=32):
        super().__init__()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        # First layer
        self.input_conv = CausalConv1d(input_channels, filters, kernel_size=2, dilation=1)

        # Hidden dilated layers
        dilations = [1, 2, 4, 8, 16, 32, 16, 8, 16, 32, 64]
        self.hidden_layers = nn.ModuleList([
            CausalConv1d(filters, filters, kernel_size=2, dilation=d)
            for d in dilations
        ])

        # Output layer
        self.output_conv = CausalConv1d(filters, output_channels, kernel_size=2, dilation=128)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

        x = self.elu(self.input_conv(x))
        for conv in self.hidden_layers:
            x =  self.elu(conv(x))

        x = self.sigmoid(self.output_conv(x))
        return x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

def r2_score(y_true, y_pred, eps=1e-7):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + eps)

# ---- 2. Training function ----
def train_model(model, X, Y, epochs=3000, batch_size=256, lr=3e-4, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'tov_solver.pt')

    # -------------------- SPLIT DATA --------------------
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2
                                                    , random_state=42)

    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

    # -------------------- CONVERT TO TENSORS --------------------
    X_train = torch.tensor(X_train[:, :, 1:2], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val[:, :, 1:2], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-8)
    best_val_loss = float('inf')

    start_time = time.time()
    criterion = nn.MSELoss()
    # -------------------- TRAINING LOOP --------------------
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))  # shuffle each epoch

        train_loss = 0.0
        n_batches = 0

        # iterate over batches
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x, batch_y = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(  
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches  # average across batches in the epoch

        # -------------------- VALIDATION --------------------
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
            r2 = r2_score(y_val.cpu(), val_output.cpu())

        current_lr = optimizer.param_groups[0]['lr']

        # -------------------- SAVE BEST MODEL --------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        # -------------------- LOGGING --------------------
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val R²: {r2:.6f} | "
              f"LR: {current_lr:.6e}")

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n✅ Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    return model




# ---- 3. Evaluation ----
def evaluate_model(model, X_test, y_test, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to same device as data
    model = model.to(device)
    model.eval()

    # Prepare tensors
    X_test = torch.tensor(X_test[:, :, 1:2], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Evaluation
    with torch.no_grad():
        pred = model(X_test)
        mse_loss = nn.MSELoss()(pred, y_test).item()
        r2 = r2_score(y_test, pred).item()

    metrics = {"MSE": mse_loss, "R2": r2}
    preds = pred.cpu().numpy()
    return metrics, preds


# ---- 4. Main script ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train WaveNet TOV model (PyTorch)')
    parser.add_argument('--input', type=str, default="data/sample_eos.csv")
    parser.add_argument('--output', type=str, default="data/sample_mr.csv")
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--np', type=int, default=64, dest='Np')
    args = parser.parse_args()

    if (
    os.path.exists("data/X_scaled.npy")
    and os.path.exists("data/Y_scaled.npy")
    ):
        X_scaled = np.load("data/X_scaled.npy")
        Y_scaled = np.load("data/Y_scaled.npy")
    else:
        X_scaled, Y_scaled = tov_load_and_preprocess(
            args.input, args.output, Np=args.Np
    )
        
    model = WaveNetTOV()
    model = train_model(model, X_scaled, Y_scaled, epochs=args.epochs, batch_size=args.batch)
