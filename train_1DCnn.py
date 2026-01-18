import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle

# ========= Config =========
BASE_DIR = r"C:\Users\emirh\Desktop\uydu telemetri data\ESA-Mission1"
YEAR_DIR = os.path.join(BASE_DIR, "multichannel_by_year")
ANOM_FILE = os.path.join(BASE_DIR, "labels.csv")

ALL_YEARS = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]
TRAIN_END = "2006-08-31"
VAL_START = "2006-09-01"
VAL_END = "2006-12-31"
TEST_START = "2007-01-01"
TEST_END = "2007-05-31"

SEQ_LEN = 120
STRIDE = 30
BATCH = 128
LR = 1e-3
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MIN_ANOMALY_LEN = 40

# ✅ LSTM İLE TAM AYNI PARAMETRELERİ KULLAN
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25  # ✅ LSTM ile aynı
GRAD_CLIP = 1.5

# ✅ CNN parametreleri (~174K params için optimize edildi)
NUM_FILTERS = [50, 100, 165]
KERNEL_SIZES = [7, 5, 3]
POOL_SIZE = 2
FC_HIDDEN = 380  # ✅ ~174K params için

USE_EARLY_STOPPING = False
PATIENCE = 8
MIN_DELTA = 1e-6

pd.options.mode.copy_on_write = True


def set_seed(seed=SEED):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ========= IO & Preprocessing (LSTM ile AYNI) =========
def load_years(years):
    dfs = []
    print(f"📂 Tüm veri yükleniyor: {years} ...")
    for y in years:
        p = os.path.join(YEAR_DIR, f"{y}.parquet")
        if not os.path.exists(p): continue
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
    if not os.path.exists(csv_path): return []
    a = pd.read_csv(csv_path)
    if "StartTime" in a.columns and "EndTime" in a.columns:
        a = a.rename(columns={"StartTime": "start", "EndTime": "end"})
    else:
        return []

    a["start"] = pd.to_datetime(a["start"], utc=True).dt.tz_localize(None)
    a["end"] = pd.to_datetime(a["end"], utc=True).dt.tz_localize(None)
    return list(a[["start", "end"]].itertuples(index=False, name=None))


def mask_anomalies(df, intervals):
    if not intervals: return df
    mask = pd.Series(False, index=df.index)
    for (s, e) in intervals: mask |= (df.index >= s) & (df.index <= e)
    return df.loc[~mask].copy()


def get_true_labels(timestamps, intervals):
    y_true = np.zeros(len(timestamps), dtype=int)
    ts_series = pd.Series(timestamps)
    for (s, e) in intervals:
        mask = (ts_series >= s) & (ts_series <= e)
        y_true[mask] = 1
    return y_true


def make_windows(df_scaled, seq_len=SEQ_LEN, stride=STRIDE, pred_h=1):
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


# ========= 1D-CNN Model =========
class CNN1DForecast(nn.Module):
    """
    1D-CNN for Time Series Forecasting
    ✅ LSTM ile eşit parametre sayısı (~174K)
    """

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
        # Input: (batch, seq_len, features)
        # Conv1D expects: (batch, features, seq_len)
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


def find_best_threshold_and_report(loss_scores, y_true):
    print("\n🔎 EŞİK OPTİMİZASYONU + BUDAMA...")

    min_thr = np.percentile(loss_scores, 75)
    max_thr = np.percentile(loss_scores, 99.5)
    thresholds = np.linspace(min_thr, max_thr, 100)

    best_f1 = -1;
    best_thr = 0;
    best_pred = None

    for thr in thresholds:
        y_pred_raw = (loss_scores > thr).astype(int)
        y_pred_pruned = prune_false_positives(y_pred_raw, MIN_ANOMALY_LEN)
        f1 = f1_score(y_true, y_pred_pruned, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1;
            best_thr = thr;
            best_pred = y_pred_pruned

    print(f"✅ En İyi Eşik: {best_thr:.6f}")

    prec = precision_score(y_true, best_pred, zero_division=0)
    rec = recall_score(y_true, best_pred, zero_division=0)
    cm = confusion_matrix(y_true, best_pred)

    print("=" * 40)
    print(f"🏆 F1-Score : {best_f1:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"📡 Recall   : {rec:.4f}")
    print("Confusion Matrix:\n", cm)
    print("=" * 40)

    return best_f1, prec, rec, best_thr


def plot_learning_curve(train_losses, val_losses, best_epoch=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)
    window = 5
    val_losses_smooth = pd.Series(val_losses).rolling(window=window, center=True, min_periods=1).mean().values

    # Normal
    ax1.plot(epochs, train_losses, label="Train", color="blue", linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax1.plot(epochs, val_losses, label="Val (Raw)", color="orange", linewidth=1, alpha=0.3)
    ax1.plot(epochs, val_losses_smooth, label="Val (Smoothed)", color="darkorange", linewidth=2, marker='s',
             markersize=4)

    if best_epoch:
        ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best ({best_epoch})')
        ax1.scatter(best_epoch, val_losses[best_epoch - 1], color='red', s=100, zorder=5)

    ax1.set_title("1D-CNN Training Curve", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch");
    ax1.set_ylabel("MSE Loss")
    ax1.legend();
    ax1.grid(True, alpha=0.3)

    # Log
    ax2.plot(epochs, train_losses, label="Train", color="blue", linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax2.plot(epochs, val_losses_smooth, label="Val", color="darkorange", linewidth=2, marker='s', markersize=4)

    if best_epoch:
        ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2)
        ax2.scatter(best_epoch, val_losses[best_epoch - 1], color='red', s=100, zorder=5)

    ax2.set_yscale('log')
    ax2.set_title("1D-CNN Training Curve (Log)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch");
    ax2.set_ylabel("MSE Loss (log)")
    ax2.legend();
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("learning_curve_cnn.png", dpi=150)
    print("✅ Grafik: learning_curve_cnn.png")
    plt.close()


def plot_learning_rate(lr_history):
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, color='purple', linewidth=2, marker='o', markersize=4)
    plt.title("1D-CNN Learning Rate", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch");
    plt.ylabel("LR")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_rate_schedule_cnn.png", dpi=150)
    print("✅ LR grafik: learning_rate_schedule_cnn.png")
    plt.close()


def main():
    set_seed()

    intervals = load_anomaly_intervals(ANOM_FILE)
    if not intervals:
        print("❌ labels.csv okunamadı");
        return

    df_all = load_years(ALL_YEARS)
    if df_all is None: return

    print(f"\n📅 Veri 3 Parçaya Bölünüyor:")
    print(f"   🔹 TRAIN: 2000 - {TRAIN_END}")
    print(f"   🔹 VAL  : {VAL_START} - {VAL_END}")
    print(f"   🔹 TEST : {TEST_START} - {TEST_END} (eğitimde KULLANILMAYACAK!)")

    train_end_ts = pd.Timestamp(TRAIN_END)
    val_start_ts = pd.Timestamp(VAL_START)
    val_end_ts = pd.Timestamp(VAL_END)

    raw_train = df_all[df_all.index <= train_end_ts].copy()
    raw_val = df_all[(df_all.index >= val_start_ts) & (df_all.index <= val_end_ts)].copy()

    print(f"\n   📊 Train: {raw_train.index.min()} → {raw_train.index.max()} ({len(raw_train)} satır)")
    print(f"   📊 Val  : {raw_val.index.min()} → {raw_val.index.max()} ({len(raw_val)} satır)")
    print(f"   ⚠️  Test seti eğitimde KULLANILMAYACAK!")

    clean_train = mask_anomalies(raw_train, intervals)
    clean_val = mask_anomalies(raw_val, intervals)

    print(f"\n📊 Temiz Veri:")
    print(f"   Train (Temiz): {len(clean_train)}")
    print(f"   Val (Temiz)  : {len(clean_val)}")
    print(f"   Val (Raw)    : {len(raw_val)}")

    all_clean = pd.concat([clean_train, clean_val], axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(all_clean)

    train_scaled = pd.DataFrame(scaler.transform(clean_train),
                                index=clean_train.index,
                                columns=clean_train.columns)

    val_scaled_clean = pd.DataFrame(scaler.transform(clean_val),
                                    index=clean_val.index,
                                    columns=clean_val.columns)

    val_scaled_raw = pd.DataFrame(scaler.transform(raw_val),
                                  index=raw_val.index,
                                  columns=raw_val.columns)

    X_train, y_train, _ = make_windows(train_scaled)
    X_val_clean, y_val_clean, _ = make_windows(val_scaled_clean)
    X_val_raw, y_val_raw, t_val = make_windows(val_scaled_raw)

    if len(X_val_raw) == 0:
        print("❌ Validation verisi oluşmadı!");
        return

    print(f"\n🔢 Windows:")
    print(f"   Train: {len(X_train):,}")
    print(f"   Val (clean): {len(X_val_clean):,}")
    print(f"   Val (raw): {len(X_val_raw):,}")

    train_loader = DataLoader(SeqDS(X_train, y_train), batch_size=BATCH, shuffle=True)
    val_loader_clean = DataLoader(SeqDS(X_val_clean, y_val_clean), batch_size=256, shuffle=False)
    val_loader_raw = DataLoader(SeqDS(X_val_raw, y_val_raw), batch_size=256, shuffle=False)

    # ✅ 1D-CNN Model (LSTM ile eşit params)
    model = CNN1DForecast(
        input_dim=train_scaled.shape[1],
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        pool_size=POOL_SIZE,
        fc_hidden=FC_HIDDEN,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 Model Parametreleri:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   ✅ LSTM ile karşılaştırılabilir (~174K)")

    # ✅ LSTM ile AYNI optimizer ve scheduler
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.7, patience=8, min_lr=1e-6
    )
    crit = nn.MSELoss()

    history_train, history_val, lr_history = [], [], []
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    print(f"\n🚀 1D-CNN Eğitimi Başlıyor ({EPOCHS} epochs)...")
    print(f"   🔹 LR: {LR}, Weight Decay: {WEIGHT_DECAY}, Dropout: {DROPOUT}")
    print(f"   🔹 Filters: {NUM_FILTERS}, FC Hidden: {FC_HIDDEN}")
    print(f"   🔹 Grad Clip: {GRAD_CLIP}")
    print("=" * 70)

    for ep in range(1, EPOCHS + 1):
        # Train
        model.train()
        batch_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            batch_losses.append(loss.item())

        avg_train = np.mean(batch_losses)
        history_train.append(avg_train)

        # Val
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader_clean:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                loss = crit(pred, yb)
                val_batch_losses.append(loss.item())

        avg_val = np.mean(val_batch_losses)
        history_val.append(avg_val)

        current_lr = opt.param_groups[0]['lr']
        lr_history.append(current_lr)
        scheduler.step(avg_val)

        print(f"Epoch {ep:02d}/{EPOCHS} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {current_lr:.2e}", end="")

        if avg_val < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val
            best_epoch = ep
            best_model_state = model.state_dict().copy()
            print(" ⭐ [BEST]")
        else:
            print()

    print("=" * 70)

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Best model (Epoch {best_epoch}) yüklendi!")

    # Threshold
    val_losses = get_loss_distribution(model, val_loader_raw)
    val_losses_smoothed = pd.Series(val_losses).rolling(window=15, min_periods=1).mean().values
    y_true_val = get_true_labels(t_val, intervals)

    best_threshold = None
    if np.sum(y_true_val) > 0:
        _, _, _, best_threshold = find_best_threshold_and_report(val_losses_smoothed, y_true_val)

    # Save
    config = {
        'input_dim': train_scaled.shape[1],
        'num_filters': NUM_FILTERS,
        'kernel_sizes': KERNEL_SIZES,
        'pool_size': POOL_SIZE,
        'fc_hidden': FC_HIDDEN,
        'dropout': DROPOUT,
        'seq_len': SEQ_LEN,
        'stride': STRIDE,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_threshold': best_threshold,
        'train_end': TRAIN_END,
        'val_start': VAL_START,
        'val_end': VAL_END,
        'test_start': TEST_START,
        'test_end': TEST_END,
        'model_type': '1D-CNN'
    }

    torch.save(model.state_dict(), "best_cnn_model.pth")
    with open("scaler_cnn.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open("model_config_cnn.pkl", 'wb') as f:
        pickle.dump(config, f)

    print("\n💾 Model kaydedildi:")
    print("   ✅ best_cnn_model.pth")
    print("   ✅ scaler_cnn.pkl")
    print("   ✅ model_config_cnn.pkl")

    plot_learning_curve(history_train, history_val, best_epoch)
    plot_learning_rate(lr_history)

    # Summary
    if np.sum(y_true_val) > 0:
        f1, prec, rec, _ = find_best_threshold_and_report(val_losses_smoothed, y_true_val)

        print("\n" + "=" * 70)
        print("📋 1D-CNN EĞİTİM ÖZETİ")
        print("=" * 70)
        print(f"🔹 Toplam Epoch: {len(history_train)}")
        print(f"🔹 Best Epoch: {best_epoch}")
        print(f"🔹 Best Val Loss: {best_val_loss:.6f}")
        print(f"🔹 Final Train: {history_train[-1]:.6f}")
        print(f"🔹 Final Val: {history_val[-1]:.6f}")
        print(f"🔹 Train/Val Gap: {abs(history_train[-1] - history_val[-1]):.6f}")
        print(f"🔹 Final LR: {lr_history[-1]:.2e}")
        print(f"🔹 Total Params: {total_params:,}")
        print("-" * 70)
        print(f"🏆 F1-Score: {f1:.4f}")
        print(f"🎯 Precision: {prec:.4f}")
        print(f"📡 Recall: {rec:.4f}")
        print("=" * 70)
        print("\n⚠️  TEST ile karşılaştırın: test_cnn_model.py")


if __name__ == "__main__":
    main()