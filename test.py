from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from Model_Class import StockMovement


def engineer_percentage_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Flatten MultiIndex columns if present (from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # standardizes column names otherwise the program doesn't try to access the right column names and throws an error.
    df = df.copy()
    df.columns = [str(col).capitalize() for col in df.columns]


    data = pd.DataFrame(index=df.index)

    data["close_return"] = df["Close"].pct_change()
    data["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]
    data["high_low_ratio"] = (df["High"] - df["Low"]) / df["Low"]
    data["gap_return"] = (df["Open"] - df["Close"].shift(1)) / df[
        "Close"
    ].shift(1)
    data["volume_change"] = df["Volume"].pct_change()

    data["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return data.replace([np.inf, -np.inf], np.nan).dropna()


def engineer_percentage_features(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.copy()
    df.columns = [str(col).capitalize() for col in df.columns]

    data = pd.DataFrame(index=df.index)

    data["close_return"] = df["Close"].pct_change()
    data["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]
    data["high_low_ratio"] = (df["High"] - df["Low"]) / df["Low"]
    data["gap_return"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    data["volume_change"] = df["Volume"].pct_change()

    if not is_inference:
        data["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        return data.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        return data.replace([np.inf, -np.inf], np.nan)



def predict_recent_ftse(
        #change the date to fit the period over which you would like to predict. Obviously, for any day in the past, then it is rather useless.
    model, scaler, lookback=30, start_date="2026-01-01", holdout_start="2026-01-01"
):

    model.eval()

    rdf = yf.download("^FTSE", start=start_date, progress=False)
    df = engineer_percentage_features(rdf)

#cleaning data which is used - there are some extra columns like dividend yield which for something like the ftse 100 just comes up as 0 from yahoo finance so is useless.
    feature_cols = [
        "close_return",
        "intraday_return",
        "high_low_ratio",
        "gap_return",
        "volume_change",
    ]
    features = df[feature_cols].values
    targets = df["target"].values


    scaled_features = scaler.transform(features)

    X_seq = []
    y_seq = []
    dates = []

    for i in range(lookback, len(scaled_features)+1):
        X_seq.append(scaled_features[i - lookback : i])
        y_seq.append(targets[i - 1])
        dates.append(df.index[i - 1])

    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_true = np.array(y_seq)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).squeeze(-1).numpy()
        binary_preds = (probabilities > 0.5).astype(int)

    results_df = pd.DataFrame({
        "Date": dates,
        "Predicted_Prob_Up": probabilities,
        "Predicted_Signal": binary_preds,
        "Actual_Direction": y_true,
    }).set_index("Date")

    ftse_holdout = results_df.loc[holdout_start:]

    if not ftse_holdout.empty:
        accuracy = (
            ftse_holdout["Predicted_Signal"]
            == ftse_holdout["Actual_Direction"]
        ).mean()

        print(ftse_holdout[["Predicted_Prob_Up", "Predicted_Signal"]]) # 1,0 buy,sell
    else:
        print(
            f"No data available for dates on or after {holdout_start}. Displaying recent predictions:"
        )
        print(results_df[["Predicted_Prob_Up", "Predicted_Signal"]].tail(10))

    return ftse_holdout


if __name__ == "__main__":
    MODEL_PATH = Path("models")

#loads saved model so best to use in the same project folder where the 'models' path is accessible although obviously can be accessed with a few tweaks from a different project.
    scaler = joblib.load(MODEL_PATH / "ftse_scaler.pkl")

    MODEL_NAME = "01_model_finance.pth_200 epochs + 0.3 droupout" # edit this to the presaved file to load.
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    model = StockMovement(input_dim=5, hidden_dim=64, num_layers=2)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))


    predict_recent_ftse(model, scaler)
