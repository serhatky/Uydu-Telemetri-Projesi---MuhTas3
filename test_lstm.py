import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import pickle

# ========= Config =========
BASE_DIR = r"C:\Users\emirh\Desktop\uydu telemetri data\ESA-Mission1"
YEAR_DIR = os.path.join(BASE_DIR, "multichannel_by_year")
ANOM_FILE = os.path.join(BASE_DIR, "labels.csv")

# Model paths
MODEL_PATH = "best_lstm_model.pth"
SCALER_PATH = "scaler.pkl"
CONFIG_PATH = "model_config.pkl"

# ✅ TEST PERİYODU (Modelin HİÇ GÖRMEDİĞİ VERİ)
TEST_START = "2007-01-01"  # 2007 Ocak
TEST_END = "2007-05-31"  # 2007 Mayıs

BATCH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MIN_ANOMALY_LEN = 40

pd.options.mode.copy_on_write = True


def set_seed(seed=SEED):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ========= Model Definition =========
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


# ========= Data Loading =========
def load_test_period(start_date, end_date):
    """Belirli tarih aralığındaki veriyi yükle"""
    print(f"📂 Test verisi yükleniyor: {start_date} → {end_date}")

    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    years_to_load = list(range(start_year, end_year + 1))

    dfs = []
    for y in years_to_load:
        p = os.path.join(YEAR_DIR, f"{y}.parquet")
        if not os.path.exists(p):
            print(f"   ⚠  {y}.parquet bulunamadı, atlanıyor...")
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

    df_all = pd.concat(dfs, axis=0)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df_filtered = df_all[(df_all.index >= start_ts) & (df_all.index <= end_ts)].copy()

    return df_filtered


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


# ========= Evaluation Functions =========
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


def evaluate_model(model, loader, y_true, timestamps, threshold):
    """Model değerlendirmesi (eğitimden gelen threshold ile)"""

    # Reconstruction errors
    loss_scores = get_loss_distribution(model, loader)
    loss_scores_smooth = pd.Series(loss_scores).rolling(window=15, min_periods=1).mean().values

    # Threshold uygula
    y_pred_raw = (loss_scores_smooth > threshold).astype(int)
    y_pred = prune_false_positives(y_pred_raw, MIN_ANOMALY_LEN)

    # Metrikler
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Sonuçları yazdır
    print("\n" + "=" * 70)
    print("📊 TEST SONUÇLARI (Validation threshold ile)")
    print("=" * 70)
    print(f"🏆 F1-Score   : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"🎯 Precision  : {prec:.4f} ({prec * 100:.2f}%)")
    print(f"📡 Recall     : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"✅ Threshold  : {threshold:.6f} (validation'dan)")
    print("\n📋 Confusion Matrix:")
    print(f"  TN: {cm[0, 0]:5d}  |  FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d}  |  TP: {cm[1, 1]:5d}")
    print("\n📈 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], digits=4))
    print("=" * 70)

    return {
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'threshold': threshold,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_true': y_true,
        'loss_scores': loss_scores_smooth,
        'timestamps': timestamps
    }


def plot_test_results(results):
    """Test sonuçlarını görselleştir"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    timestamps = results['timestamps']
    y_true = results['y_true']
    y_pred = results['y_pred']
    loss_scores = results['loss_scores']
    threshold = results['threshold']

    # 1. Loss distribution
    ax = axes[0, 0]
    ax.hist(loss_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Reconstruction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Loss Distribution (Test Set)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Time series plot
    ax = axes[0, 1]
    time_indices = np.arange(len(timestamps))

    anomaly_mask = y_true == 1
    normal_mask = y_true == 0

    ax.scatter(time_indices[normal_mask], loss_scores[normal_mask],
               c='blue', s=1, alpha=0.3, label='Normal')
    ax.scatter(time_indices[anomaly_mask], loss_scores[anomaly_mask],
               c='red', s=10, alpha=0.7, label='True Anomaly')
    ax.axhline(threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Reconstruction Error', fontsize=12)
    ax.set_title('Anomaly Detection (2006 Jul-Dec)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Confusion Matrix Heatmap
    ax = axes[1, 0]
    cm = results['confusion_matrix']
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black",
                           fontsize=20, fontweight='bold')

    plt.colorbar(im, ax=ax)

    # 4. Metrics Bar Chart
    ax = axes[1, 1]
    metrics = ['F1-Score', 'Precision', 'Recall']
    values = [results['f1'], results['precision'], results['recall']]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test Set Performance', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.3f}\n({value * 100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Test sonuçları grafiği kaydedildi: test_results.png")
    plt.close()


def load_model_and_config():
    """Model, scaler ve config yükle"""

    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    model = LSTMForecast(
        input_dim=config['input_dim'],
        hidden=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"✅ Model yüklendi: {MODEL_PATH}")
    print(f"✅ Scaler yüklendi: {SCALER_PATH}")
    print(f"✅ Config yüklendi: {CONFIG_PATH}")
    print(f"   🔹 Input Dim: {config['input_dim']}")
    print(f"   🔹 Hidden Size: {config['hidden_size']}")
    print(f"   🔹 Num Layers: {config['num_layers']}")
    print(f"   🔹 Seq Length: {config['seq_len']}")
    print(f"   🔹 Best Epoch: {config.get('best_epoch', 'N/A')}")
    print(f"   🔹 Best Val Loss: {config.get('best_val_loss', 'N/A'):.6f}")
    print(f"   🔹 Val Threshold: {config.get('best_threshold', 'N/A'):.6f}")

    return model, scaler, config


def main():
    set_seed()

    print("=" * 70)
    print("🧪 LSTM MODEL TEST - UNSEEN DATA")
    print("=" * 70)
    print(f"📅 Test Periyodu: {TEST_START} → {TEST_END}")
    print("⚠️  Bu veri eğitim ve validation'da HİÇ GÖRÜLMEDİ!")
    print("=" * 70)

    # Model kontrolü
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model bulunamadı: {MODEL_PATH}")
        print("⚠  Önce modeli eğitip kaydetmelisiniz!")
        print("💡 train_lstm_3way_split.py dosyasını çalıştırın.")
        return

    # Anomali intervals
    intervals = load_anomaly_intervals(ANOM_FILE)
    if not intervals:
        print("❌ HATA: labels.csv okunamadı.");
        return

    # Model yükle
    print("\n📥 Model yükleniyor...")
    model, scaler, config = load_model_and_config()

    # Threshold kontrolü
    if config.get('best_threshold') is None:
        print("\n❌ HATA: Config'de threshold bulunamadı!")
        print("⚠  Modeli yeniden eğitin (train_lstm_3way_split.py)")
        return

    threshold = config['best_threshold']

    # Test verisi yükle
    print(f"\n📂 Test verisi yükleniyor...")
    df_test = load_test_period(TEST_START, TEST_END)

    if df_test is None or len(df_test) == 0:
        print("❌ Test verisi yüklenemedi veya boş!")
        return

    print(f"   📊 Test veri boyutu: {len(df_test)} satır")
    print(f"   📅 Test aralığı: {df_test.index.min()} → {df_test.index.max()}")

    # Scale test data
    test_scaled = pd.DataFrame(
        scaler.transform(df_test),
        index=df_test.index,
        columns=df_test.columns
    )

    # Windows oluştur
    X_test, y_test, t_test = make_windows(
        test_scaled,
        seq_len=config['seq_len'],
        stride=config['stride']
    )
    print(f"   🔢 Test window sayısı: {len(X_test)}")

    if len(X_test) == 0:
        print("❌ Test window'ları oluşturulamadı!")
        return

    # DataLoader
    test_loader = DataLoader(SeqDS(X_test, y_test), batch_size=BATCH, shuffle=False)

    # True labels
    y_true = get_true_labels(t_test, intervals)
    print(f"   ⚠  Anomali window sayısı: {np.sum(y_true)} / {len(y_true)} ({np.sum(y_true) / len(y_true) * 100:.2f}%)")

    # Değerlendirme
    print("\n🔬 Model değerlendiriliyor...")

    results = evaluate_model(
        model,
        test_loader,
        y_true,
        t_test,
        threshold=threshold
    )

    # Görselleştirme
    print("\n📊 Sonuçlar görselleştiriliyor...")
    plot_test_results(results)

    print("\n" + "=" * 70)
    print("✅ TEST TAMAMLANDI!")
    print("=" * 70)
    print("\n📝 ÖZET:")
    print(f"   🔹 Model validation threshold'u ile test edildi")
    print(f"   🔹 Test periyodu: {TEST_START} → {TEST_END}")
    print(f"   🔹 F1-Score: {results['f1']:.4f}")
    print(f"   🔹 Precision: {results['precision']:.4f}")
    print(f"   🔹 Recall: {results['recall']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()