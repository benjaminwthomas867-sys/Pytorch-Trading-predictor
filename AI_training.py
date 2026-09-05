import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from Model_Class import StockMovement
import torch.nn as nn
from pathlib import Path
import joblib


if torch.cuda.is_available(): torch.set_default_device("cuda") # enables compuatation on CUDA cores if you have a CUDA-enabled gpu

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def producepercentagefeatures(df: pd.DataFrame) -> pd.DataFrame:

    data = pd.DataFrame(index=df.index)

    data['close_return'] = df['close'].pct_change()
    data['intraday_return'] = (df['close'] - df['open']) / df['open']
    data['high_low_ratio'] = (df['high'] - df['low']) / df['low']
    data['gap_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    data['volume_change'] = df['volume'].pct_change()


    data['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna()

def createsequences(X_data: np.ndarray, y_data: np.ndarray, lookback: int = 30):
    X_seq, y_seq = [], []
    for i in range(lookback-1, len(X_data)):
        X_seq.append(X_data[i - lookback:i])
        y_seq.append(y_data[i])
    return np.array(X_seq), np.array(y_seq)

def buildpipeline(df: pd.DataFrame, lookback: int = 30, train_ratio: float = 0.8, batch_size: int = 32):

    data = producepercentagefeatures(df)

    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols].values
    y = data['target'].values

    split_idx = int(len(X) * train_ratio)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train_seq, y_train_seq = createsequences(X_train_scaled, y_train_raw, lookback)
    X_test_seq, y_test_seq = createsequences(X_test_scaled, y_test_raw, lookback)

    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)



    #will create 'models' path in the file directory where all trained models shall be saved
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)


    SCALER_SAVE_PATH = MODEL_PATH / "ftse_scaler.pkl"
    joblib.dump(scaler, SCALER_SAVE_PATH)

    print(f"Scaler saved successfully to: {SCALER_SAVE_PATH}")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler

#change the ticker for different bits of data - works with TIDM codes e.g. VUKG.L which is a FTSE 100 etf.
df = yf.download('^FTSE', start='2000-01-01', end='2026-08-01') #used from 2000-01-01 as that is the point from which yahoo finance had access to volume data.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.columns = df.columns.str.lower()


if 'adj close' in df.columns:
    df = df.drop(columns=['adj close'])


train_loader, test_loader, scaler = buildpipeline(df, lookback=30)

epochs = 50 # play around with this to get different results

model = StockMovement(input_dim=5, hidden_dim=64, num_layers=2)
loss_fn = nn.BCEWithLogitsLoss()  # added () to instantiate the class
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01) # playing around with lr is interesting...


for epoch in range(epochs):
    print(f"Epoch {epoch + 1} training")
    model.train()
    running_loss = 0.0


    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        y_pred = model(batch_X)

        loss = loss_fn(y_pred, batch_y.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch: {epoch}")

for name, param in model.named_parameters():
    print(name, param.shape)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok=True)

MODEL_NAME = "01_model_finance.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(model.state_dict(),MODEL_SAVE_PATH)
