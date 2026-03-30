import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

csv_path = "canvas_state_001_uvz.csv"

# 1. Load uvz data
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
u = data[:, 0]   # mm
v = data[:, 1]
z = data[:, 2]

# Add the same synthetic sag so NCF learns что‑то не нулевое
sag_amplitude_mm = 5.0
sigma_mm         = 100.0

u_center = 0.5 * (u.max() + u.min())
v_center = 0.5 * (v.max() + v.min())

sag = sag_amplitude_mm * np.exp(
    -(((u - u_center) ** 2 + (v - v_center) ** 2) / (2.0 * sigma_mm ** 2))
)
z_effective = z + sag   # target for NCF

# 2. Normalize inputs (rough but fine for now)
u_mean, u_std = u.mean(), u.std()
v_mean, v_std = v.mean(), v.std()

uv = np.column_stack([
    (u - u_mean) / u_std,
    (v - v_mean) / v_std,
])

X_train, X_val, y_train, y_val = train_test_split(
    uv, z_effective, test_size=0.2, random_state=42
)

X_train = torch.from_numpy(X_train).float()
X_val   = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(1)
y_val   = torch.from_numpy(y_val).float().unsqueeze(1)

# 3. Tiny MLP: 2 → 64 → 64 → 1
class NCFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

model = NCFMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 4. Training loop (CPU‑friendly)
n_epochs = 500
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

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        print(f"Epoch {epoch:4d} | train_loss = {loss.item():.6f} | val_loss = {val_loss:.6f}")

# 5. Final RMSE on validation set
model.eval()
with torch.no_grad():
    val_pred = model(X_val).squeeze(1)
    rmse_val = torch.sqrt(torch.mean((val_pred - y_val.squeeze(1))**2)).item()
print(f"\nNCF MLP validation RMSE (mm): {rmse_val:.4f}")