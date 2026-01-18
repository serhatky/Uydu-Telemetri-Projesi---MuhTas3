import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pickle

# ========= Config =========
BASE_DIR = r"C:\Users\emirh\Desktop\uydu telemetri data\ESA-Mission1"
YEAR_DIR = os.path.join(BASE_DIR, "multichannel_by_year")
ANOM_FILE = os.path.join(BASE_DIR, "labels.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ANOMALY_LEN = 40


# ========= Model Definition (Training ile AYNI) =========
class CNN1DForecast(nn.Module):
    def __init__(self, input_dim, num_filters=[50, 100, 165],
                 kernel_sizes=[7, 5, 3], pool_size=2,
                 fc_hidden=380, dropout=0.25):
        super().__init__()

        self.input_dim = input_dim

        # Conv Block 1
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters[0],
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2
        )
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.pool1 = nn.MaxPool1d(pool_size)

        # Conv Block 2
        self.conv2 = nn.Conv1d(
            in_channels=num_filters[0],
            out_channels=num_filters[1],
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2
        )
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.pool2 = nn.MaxPool1d(pool_size)

        # Conv Block 3
        self.conv3 = nn.Conv1d(
            in_channels=num_filters[1],
            out_channels=num_filters[2],
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2
        )
        self.bn3 = nn.BatchNorm1d(num_filters[2])
        self.pool3 = nn.MaxPool1d(pool_size)

        # Activation & Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(num_filters[2], fc_hidden)
        self.bn_fc = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, input_dim)

    def forward(self, x):
        x = x.transpose(1, 2)

        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.squeeze(-1)

        # Fully Connected
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ========= Helper Functions =========
def load_years(years, year_dir):
    dfs = []
    print(f"📂 Test verisi yükleniyor: {years} ...")
    for y in years:
        p = os.path.join(year_dir, f"{y}.parquet")
        if not os.path.exists(p):
            print(f"⚠️  {y}.parquet bulunamadı")
            continue
        df = pd.read_parquet(p)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
            else:
                df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.sort_index().ffill().bfill()
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].astype(np.float32)
        dfs.append(df)

    if not dfs: return None
    return pd.concat(dfs, axis=0)


def load_anomaly_intervals(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️  {csv_path} bulunamadı")
        return []
    a = pd.read_csv(csv_path)
    if "StartTime" in a.columns and "EndTime" in a.columns:
        a = a.rename(columns={"StartTime": "start", "EndTime": "end"})
    else:
        print("⚠️  labels.csv'de StartTime/EndTime kolonları yok")
        return []

    a["start"] = pd.to_datetime(a["start"], utc=True).dt.tz_localize(None)
    a["end"] = pd.to_datetime(a["end"], utc=True).dt.tz_localize(None)
    return list(a[["start", "end"]].itertuples(index=False, name=None))


def get_true_labels(timestamps, intervals):
    y_true = np.zeros(len(timestamps), dtype=int)
    ts_series = pd.Series(timestamps)
    for (s, e) in intervals:
        mask = (ts_series >= s) & (ts_series <= e)
        y_true[mask] = 1
    return y_true


def make_windows(df_scaled, seq_len, stride, pred_h=1):
    arr = df_scaled.values
    idx = df_scaled.index
    Xs, ys, ts = [], [], []
    for s in range(0, len(arr) - seq_len - pred_h + 1, stride):
        e = s + seq_len
        t_idx = e + pred_h - 1
        Xs.append(arr[s:e, :])
        ys.append(arr[t_idx, :])
        ts.append(idx[t_idx])
    if len(Xs) == 0: return np.array([]), np.array([]), np.array([])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(ts)


class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.y[i]


def get_loss_distribution(model, loader, device=DEVICE):
    model.eval()
    crit = nn.MSELoss(reduction='none')
    losses = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            batch_loss = crit(pred, yb).mean(dim=1).cpu().numpy()
            losses.append(batch_loss)
    return np.concatenate(losses)


def prune_false_positives(y_pred, min_len=MIN_ANOMALY_LEN):
    pruned = y_pred.copy()
    n = len(pruned)
    i = 0
    while i < n:
        if pruned[i] == 1:
            start = i
            while i < n and pruned[i] == 1:
                i += 1
            end = i
            length = end - start
            if length < min_len:
                pruned[start:end] = 0
        else:
            i += 1
    return pruned


def evaluate_with_threshold(loss_scores, y_true, threshold):
    y_pred_raw = (loss_scores > threshold).astype(int)
    y_pred_pruned = prune_false_positives(y_pred_raw, MIN_ANOMALY_LEN)

    f1 = f1_score(y_true, y_pred_pruned, zero_division=0)
    prec = precision_score(y_true, y_pred_pruned, zero_division=0)
    rec = recall_score(y_true, y_pred_pruned, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_pruned)

    return f1, prec, rec, cm, y_pred_pruned


def plot_test_results(timestamps, y_true, y_pred, loss_scores, threshold):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Loss scores
    ax1.plot(timestamps, loss_scores, label='Reconstruction Loss', color='blue', linewidth=1, alpha=0.7)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.6f})')
    ax1.fill_between(timestamps, 0, loss_scores, where=(loss_scores > threshold),
                     color='red', alpha=0.3, label='Predicted Anomaly')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('1D-CNN Test Set - Reconstruction Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Predictions
    ax2.fill_between(timestamps, 0, 1, where=(y_true == 1),
                     color='red', alpha=0.3, label='True Anomaly', step='mid')
    ax2.fill_between(timestamps, 0, 0.5, where=(y_pred == 1),
                     color='blue', alpha=0.5, label='Predicted Anomaly', step='mid')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel('Anomaly', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_title('1D-CNN Test Set - Anomaly Detection', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_results_cnn.png', dpi=150, bbox_inches='tight')
    print("✅ Grafik kaydedildi: test_results_cnn.png")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("🧪 1D-CNN MODEL TEST")
    print("=" * 70)

    # Load model config
    if not os.path.exists("model_config_cnn.pkl"):
        print("❌ model_config_cnn.pkl bulunamadı!")
        print("   Önce train_cnn_model.py ile modeli eğitin.")
        return

    with open("model_config_cnn.pkl", 'rb') as f:
        config = pickle.load(f)

    print("\n📋 Model Config:")
    print(f"   Model Type: {config.get('model_type', '1D-CNN')}")
    print(f"   Input Dim: {config['input_dim']}")
    print(f"   Filters: {config['num_filters']}")
    print(f"   FC Hidden: {config['fc_hidden']}")
    print(f"   Seq Len: {config['seq_len']}")
    print(f"   Stride: {config['stride']}")
    print(f"   Best Epoch: {config['best_epoch']}")
    print(f"   Best Val Loss: {config['best_val_loss']:.6f}")
    print(f"   Threshold: {config['best_threshold']:.6f}")

    # Load scaler
    if not os.path.exists("scaler_cnn.pkl"):
        print("❌ scaler_cnn.pkl bulunamadı!")
        return

    with open("scaler_cnn.pkl", 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler yüklendi")

    # Load model
    if not os.path.exists("best_cnn_model.pth"):
        print("❌ best_cnn_model.pth bulunamadı!")
        return

    model = CNN1DForecast(
        input_dim=config['input_dim'],
        num_filters=config['num_filters'],
        kernel_sizes=config['kernel_sizes'],
        pool_size=config['pool_size'],
        fc_hidden=config['fc_hidden'],
        dropout=config['dropout']
    ).to(DEVICE)

    model.load_state_dict(torch.load("best_cnn_model.pth", map_location=DEVICE))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model yüklendi ({total_params:,} params)")

    # Load test data
    test_start = pd.Timestamp(config['test_start'])
    test_end = pd.Timestamp(config['test_end'])

    print(f"\n📅 Test Periyodu: {test_start.date()} → {test_end.date()}")

    df_all = load_years([2007], YEAR_DIR)
    if df_all is None:
        print("❌ Test verisi yüklenemedi!")
        return

    test_df = df_all[(df_all.index >= test_start) & (df_all.index <= test_end)].copy()
    print(f"📊 Test Data: {len(test_df)} satır")

    # Scale test data
    test_scaled = pd.DataFrame(
        scaler.transform(test_df),
        index=test_df.index,
        columns=test_df.columns
    )

    # Create windows
    X_test, y_test, t_test = make_windows(
        test_scaled,
        seq_len=config['seq_len'],
        stride=config['stride']
    )

    if len(X_test) == 0:
        print("❌ Test windows oluşturulamadı!")
        return

    print(f"🔢 Test Windows: {len(X_test):,}")

    # Load anomaly labels
    intervals = load_anomaly_intervals(ANOM_FILE)
    y_true_test = get_true_labels(t_test, intervals)

    num_anomalies = np.sum(y_true_test)
    print(f"🔴 Test setinde {num_anomalies} anomali penceresi var")

    if num_anomalies == 0:
        print("⚠️  Test setinde anomali yok!")
        return

    # Get predictions
    test_loader = DataLoader(SeqDS(X_test, y_test), batch_size=256, shuffle=False)
    test_losses = get_loss_distribution(model, test_loader)

    # Smooth losses
    test_losses_smoothed = pd.Series(test_losses).rolling(window=15, min_periods=1).mean().values

    # Use threshold from validation
    threshold = config['best_threshold']

    print(f"\n🎯 Eşik Değeri (Validation'dan): {threshold:.6f}")

    # Evaluate
    f1, prec, rec, cm, y_pred_test = evaluate_with_threshold(
        test_losses_smoothed,
        y_true_test,
        threshold
    )

    print("\n" + "=" * 70)
    print("📊 TEST SET SONUÇLARI (1D-CNN)")
    print("=" * 70)
    print(f"🏆 F1-Score : {f1:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"📡 Recall   : {rec:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 70)

    # Detailed metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📈 Detaylı Metrikler:")
    print(f"   True Negatives  (TN): {tn:,}")
    print(f"   False Positives (FP): {fp:,}")
    print(f"   False Negatives (FN): {fn:,}")
    print(f"   True Positives  (TP): {tp:,}")
    print(f"   Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")

    # Loss statistics
    print(f"\n📊 Loss İstatistikleri:")
    print(f"   Min Loss:    {test_losses_smoothed.min():.6f}")
    print(f"   Max Loss:    {test_losses_smoothed.max():.6f}")
    print(f"   Mean Loss:   {test_losses_smoothed.mean():.6f}")
    print(f"   Median Loss: {np.median(test_losses_smoothed):.6f}")
    print(f"   Std Loss:    {test_losses_smoothed.std():.6f}")

    # Plot results
    plot_test_results(t_test, y_true_test, y_pred_test, test_losses_smoothed, threshold)

    # Save results
    results = {
        'f1_score': f1,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm,
        'threshold': threshold,
        'test_losses': test_losses_smoothed,
        'y_true': y_true_test,
        'y_pred': y_pred_test,
        'timestamps': t_test,
        'model_type': '1D-CNN'
    }

    with open('test_results_cnn.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n💾 Test sonuçları kaydedildi:")
    print("   ✅ test_results_cnn.pkl")
    print("   ✅ test_results_cnn.png")

    print("\n" + "=" * 70)
    print("✅ TEST TAMAMLANDI!")
    print("=" * 70)
    print("\n💡 LSTM ile karşılaştırma için:")
    print("   1. test_lstm_model.py'yi çalıştırın")
    print("   2. Her iki modelin test_results_*.pkl dosyalarını karşılaştırın")


if __name__ == "__main__":
    main()