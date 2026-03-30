import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

csv_path = "canvas_state_001_uvz.csv"

# 1. Load uvz data
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
u = data[:, 0]
v = data[:, 1]
z = data[:, 2]

# Synthetic sag (same as in baselines)
sag_amplitude_mm = 5.0
sigma_mm         = 100.0

u_center = 0.5 * (u.max() + u.min())
v_center = 0.5 * (v.max() + v.min())

sag = sag_amplitude_mm * np.exp(
    -(((u - u_center) ** 2 + (v - v_center) ** 2) / (2.0 * sigma_mm ** 2))
)
z_effective = z + sag

# 2. Normalize inputs AND target
u_mean, u_std = u.mean(), u.std()
v_mean, v_std = v.mean(), v.std()
z_mean, z_std = z_effective.mean(), z_effective.std()

uv = np.column_stack([
    (u - u_mean) / u_std,
    (v - v_mean) / v_std,
])
z_norm = (z_effective - z_mean) / z_std

X_train, X_val, y_train, y_val = train_test_split(
    uv, z_norm, test_size=0.2, random_state=42
)

X_train = torch.from_numpy(X_train).float()
X_val   = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(1)
y_val   = torch.from_numpy(y_val).float().unsqueeze(1)

# 3. Slightly wider MLP with small weight decay
class NCFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

model = NCFMLP()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.MSELoss()

n_epochs = 1000
batch_size = 64

for epoch in range(1, n_epochs + 1):
    model.train()
    perm = torch.randperm(X_train.size(0))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    for i in range(0, X_train.size(0), batch_size):
        xb = X_train_shuffled[i:i+batch_size]
        yb = y_train_shuffled[i:i+batch_size]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        print(f"Epoch {epoch:4d} | train_loss = {loss.item():.6f} | val_loss = {val_loss:.6f}")

# 4. Compute RMSE in mm (denormalize z)
model.eval()
with torch.no_grad():
    val_pred_norm = model(X_val).squeeze(1)
    val_pred_mm = val_pred_norm * z_std + z_mean
    y_val_mm    = y_val.squeeze(1) * z_std + z_mean
    rmse_val_mm = torch.sqrt(torch.mean((val_pred_mm - y_val_mm) ** 2)).item()

print(f"\nUpgraded NCF MLP validation RMSE (mm): {rmse_val_mm:.4f}")