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

# ✅ STANDART 3-WAY SPLIT (Güncellenmiş Tarihler)
ALL_YEARS = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]  # 2007 eklendi
TRAIN_END = "2006-08-31"  # Train: 2000 - 2006 Ağustos
VAL_START = "2006-09-01"  # Validation: 2006 Eylül - Aralık
VAL_END = "2006-12-31"
TEST_START = "2007-01-01"  # Test: 2007 Ocak - Mayıs (eğitimde KULLANILMAYACAK!)
TEST_END = "2007-05-31"

SEQ_LEN = 120
STRIDE = 30
BATCH = 128
LR = 1e-3
EPOCHS = 50  # ✅ 36 → 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MIN_ANOMALY_LEN = 40

# Model parametreleri
WEIGHT_DECAY = 1e-5  # 5e-6 → 1e-5 (biraz daha güçlü regularization)
DROPOUT = 0.25
GRAD_CLIP = 1.5  # 2.0 → 1.5 (daha aggressive clipping)
HIDDEN_SIZE = 160
NUM_LAYERS = 2

# Early stopping
USE_EARLY_STOPPING = False
PATIENCE = 8
MIN_DELTA = 1e-6

pd.options.mode.copy_on_write = True


def set_seed(seed=SEED):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ========= IO & Preprocessing =========
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


class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden=160, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.ln(out)
        out = self.dropout(out)
        return self.head(out)


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

    # Normal scale
    ax1.plot(epochs, train_losses, label="Train Loss", color="blue", linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax1.plot(epochs, val_losses, label="Val Loss (Raw)", color="orange", linewidth=1, alpha=0.3)
    ax1.plot(epochs, val_losses_smooth, label="Val Loss (Smoothed)", color="darkorange", linewidth=2, marker='s',
             markersize=4)

    if best_epoch is not None:
        ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax1.scatter(best_epoch, val_losses[best_epoch - 1], color='red', s=100, zorder=5, label='Best Model')

    ax1.set_title("Eğitim ve Doğrulama Kaybı", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Log scale
    ax2.plot(epochs, train_losses, label="Train Loss", color="blue", linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax2.plot(epochs, val_losses, label="Val Loss (Raw)", color="orange", linewidth=1, alpha=0.3)
    ax2.plot(epochs, val_losses_smooth, label="Val Loss (Smoothed)", color="darkorange", linewidth=2, marker='s',
             markersize=4)

    if best_epoch is not None:
        ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax2.scatter(best_epoch, val_losses[best_epoch - 1], color='red', s=100, zorder=5)

    ax2.set_yscale('log')
    ax2.set_title("Eğitim ve Doğrulama Kaybı (Log Scale)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("MSE Loss (Log)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6, which="both")

    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    print("✅ Grafik kaydedildi: learning_curve.png")
    plt.close()


def plot_learning_rate(lr_history):
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, color='purple', linewidth=2, marker='o', markersize=4)
    plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("learning_rate_schedule.png", dpi=150)
    print("✅ Learning rate grafiği kaydedildi: learning_rate_schedule.png")
    plt.close()


def main():
    set_seed()

    intervals = load_anomaly_intervals(ANOM_FILE)
    if not intervals:
        print("❌ HATA: labels.csv okunamadı.");
        return

    df_all = load_years(ALL_YEARS)
    if df_all is None: return

    # ✅ 3-WAY SPLIT
    print(f"\n📅 Veri 3 Parçaya Bölünüyor:")
    print(f"   🔹 TRAIN: 2000 - {TRAIN_END}")
    print(f"   🔹 VAL  : {VAL_START} - {VAL_END}")
    print(f"   🔹 TEST : {TEST_START} - {TEST_END} (eğitimde KULLANILMAYACAK!)")

    train_end_ts = pd.Timestamp(TRAIN_END)
    val_start_ts = pd.Timestamp(VAL_START)
    val_end_ts = pd.Timestamp(VAL_END)
    test_start_ts = pd.Timestamp(TEST_START)

    # Train set
    raw_train = df_all[df_all.index <= train_end_ts].copy()

    # Validation set
    raw_val = df_all[(df_all.index >= val_start_ts) & (df_all.index <= val_end_ts)].copy()

    # ⚠️ Test set - sadece bilgi için yükle, eğitimde KULLANMA!
    raw_test = df_all[df_all.index >= test_start_ts].copy()

    print(f"\n   📊 Train  : {raw_train.index.min()} → {raw_train.index.max()} ({len(raw_train)} satır)")
    print(f"   📊 Val    : {raw_val.index.min()} → {raw_val.index.max()} ({len(raw_val)} satır)")
    print(f"   📊 Test   : {raw_test.index.min()} → {raw_test.index.max()} ({len(raw_test)} satır)")
    print(f"   ⚠️  Test seti eğitimde KULLANILMAYACAK!")

    # Anomalileri temizle
    clean_train = mask_anomalies(raw_train, intervals)
    clean_val = mask_anomalies(raw_val, intervals)

    print(f"\n📊 Temiz Veri:")
    print(f"   Train (Temiz): {len(clean_train)}")
    print(f"   Val (Temiz)  : {len(clean_val)}")
    print(f"   Val (Raw)    : {len(raw_val)}")

    # Scaler: Sadece train+val temiz veri ile fit et
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
        print("❌ HATA: Validation verisi oluşmadı!")
        return

    train_loader = DataLoader(SeqDS(X_train, y_train), batch_size=BATCH, shuffle=True)
    val_loader_clean = DataLoader(SeqDS(X_val_clean, y_val_clean), batch_size=256, shuffle=False)
    val_loader_raw = DataLoader(SeqDS(X_val_raw, y_val_raw), batch_size=256, shuffle=False)

    model = LSTMForecast(input_dim=train_scaled.shape[1],
                         hidden=HIDDEN_SIZE,
                         num_layers=NUM_LAYERS,
                         dropout=DROPOUT).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.7, patience=8, min_lr=1e-6
    )

    crit = nn.MSELoss()

    history_train = []
    history_val = []
    lr_history = []

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    print(f"\n🚀 Eğitim Başlıyor (Max {EPOCHS} epochs)...")
    print(f"   🔹 Learning Rate: {LR}")
    print(f"   🔹 Weight Decay: {WEIGHT_DECAY}")
    print(f"   🔹 Dropout: {DROPOUT}")
    print(f"   🔹 Hidden Size: {HIDDEN_SIZE}")
    print(f"   🔹 Num Layers: {NUM_LAYERS}")
    print(f"   🔹 Gradient Clipping: {GRAD_CLIP}")
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

        avg_train_loss = np.mean(batch_losses)
        history_train.append(avg_train_loss)

        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader_clean:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                loss = crit(pred, yb)
                val_batch_losses.append(loss.item())

        avg_val_loss = np.mean(val_batch_losses)
        history_val.append(avg_val_loss)

        current_lr = opt.param_groups[0]['lr']
        lr_history.append(current_lr)

        scheduler.step(avg_val_loss)

        print(f"Epoch {ep:02d}/{EPOCHS} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.2e}",
              end="")

        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            best_epoch = ep
            best_model_state = model.state_dict().copy()
            print(" ⭐ [BEST]")
        else:
            print()

    print("=" * 70)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Best model (Epoch {best_epoch}) yüklendi!")

    # Model kaydetme
    print("\n💾 Model, scaler ve config kaydediliyor...")

    val_losses = get_loss_distribution(model, val_loader_raw)
    val_losses_smoothed = pd.Series(val_losses).rolling(window=15, min_periods=1).mean().values
    y_true_val = get_true_labels(t_val, intervals)

    best_threshold = None
    if np.sum(y_true_val) > 0:
        _, _, _, best_threshold = find_best_threshold_and_report(val_losses_smoothed, y_true_val)

    config = {
        'input_dim': train_scaled.shape[1],
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
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
        'test_end': TEST_END
    }

    torch.save(model.state_dict(), "best_lstm_model.pth")

    with open("scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    with open("model_config.pkl", 'wb') as f:
        pickle.dump(config, f)

    print("✅ Model kaydedildi: best_lstm_model.pth")
    print("✅ Scaler kaydedildi: scaler.pkl")
    print("✅ Config kaydedildi: model_config.pkl")

    # Grafikleri çiz
    plot_learning_curve(history_train, history_val, best_epoch)
    plot_learning_rate(lr_history)

    # Değerlendirme (Validation set üzerinde)
    print("\n📊 Validation Set Değerlendirmesi:")

    if np.sum(y_true_val) == 0:
        print(f"⚠ Validation setinde hiç anomali yok.")
    else:
        print(f"ℹ Validation setinde {np.sum(y_true_val)} adet anomali penceresi var.")
        f1, prec, rec, _ = find_best_threshold_and_report(val_losses_smoothed, y_true_val)

        # Final özet
        print("\n" + "=" * 70)
        print("📋 EĞİTİM ÖZETİ")
        print("=" * 70)
        print(f"🔹 Toplam Epoch      : {len(history_train)}")
        print(f"🔹 Best Epoch       : {best_epoch}")
        print(f"🔹 Best Val Loss    : {best_val_loss:.6f}")
        print(f"🔹 Final Train Loss : {history_train[-1]:.6f}")
        print(f"🔹 Final Val Loss   : {history_val[-1]:.6f}")
        print(f"🔹 Train/Val Gap    : {abs(history_train[-1] - history_val[-1]):.6f}")
        print(f"🔹 Final LR         : {lr_history[-1]:.2e}")
        print("-" * 70)
        print(f"🏆 F1-Score (Val)   : {f1:.4f}")
        print(f"🎯 Precision (Val)  : {prec:.4f}")
        print(f"📡 Recall (Val)     : {rec:.4f}")
        print("=" * 70)
        print("\n⚠️  TEST SETİ henüz kullanılmadı - test_lstm_model.py ile test edin!")


if __name__ == "__main__":
    main()