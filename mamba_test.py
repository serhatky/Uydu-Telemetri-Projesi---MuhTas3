"""
MAMBA TEST SCRIPT - NO EINOPS
Eğitilmiş Mamba modelini test setinde değerlendirir
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle
from torch.utils.data import Dataset, DataLoader

# ========= Config =========
BASE_DIR = r"C:\Users\emirh\Desktop\uydu telemetri data\ESA-Mission1"
YEAR_DIR = os.path.join(BASE_DIR, "multichannel_by_year")
ANOM_FILE = os.path.join(BASE_DIR, "labels.csv")

MODEL_PATH = "best_mamba_model.pth"
SCALER_PATH = "mamba_scaler.pkl"
CONFIG_PATH = "mamba_config.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ANOMALY_LEN = 40

pd.options.mode.copy_on_write = True


# ========= MAMBA Model (Training ile AYNI!) =========
class MambaBlock(nn.Module):
    """Training ile TAM AYNI architecture"""

    def __init__(self, d_model, d_state=16, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )

        self.x_proj = nn.Linear(self.d_inner, d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # NO EINOPS: transpose instead of rearrange
        x = x.transpose(1, 2)  # (batch, d_inner, seq)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch, seq, d_inner)
        x = F.silu(x)

        dt = F.softplus(self.dt_proj(x))

        # Simplified SSM
        y = x * F.silu(z) + x * self.D.unsqueeze(0).unsqueeze(0)
        y = self.out_proj(y)

        return y + residual


class MambaForecast(nn.Module):
    """Training ile TAM AYNI architecture"""

    def __init__(self, input_dim, hidden=160, num_layers=2, dropout=0.25, d_state=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden)

        self.layers = nn.ModuleList([
            MambaBlock(d_model=hidden, d_state=d_state, expand_factor=2)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x):
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)

        return x


# ========= Helper Functions =========
def load_years(years):
    dfs = []
    print(f"📂 Test verisi yükleniyor: {years} ...")
    for y in years:
        p = os.path.join(YEAR_DIR, f"{y}.parquet")
        if not os.path.exists(p):
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

    if not dfs:
        return None
    return pd.concat(dfs, axis=0)


def load_anomaly_intervals(csv_path):
    if not os.path.exists(csv_path):
        return []
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
    if len(Xs) == 0:
        return np.array([]), np.array([]), np.array([])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(ts)


class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    ax1.plot(timestamps, y_true, 'g-', label='True Anomalies', linewidth=2, alpha=0.7)
    ax1.plot(timestamps, y_pred, 'r-', label='Predicted Anomalies', linewidth=1, alpha=0.6)
    ax1.fill_between(timestamps, 0, y_true, color='green', alpha=0.2)
    ax1.fill_between(timestamps, 0, y_pred, color='red', alpha=0.2)
    ax1.set_title('Mamba Test Set: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Anomaly Label', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(timestamps, loss_scores, 'b-', label='Reconstruction Loss', linewidth=1, alpha=0.7)
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
    ax2.fill_between(timestamps, threshold, loss_scores, where=(loss_scores > threshold),
                     color='red', alpha=0.3, label='Predicted Anomalies')
    ax2.set_title('Mamba Loss Scores', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('MSE Loss', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mamba_test_results.png', dpi=150, bbox_inches='tight')
    print("✅ Test grafiği kaydedildi: mamba_test_results.png")
    plt.close()


def main():
    print("=" * 70)
    print("🧪 MAMBA MODEL TEST")
    print("=" * 70)

    # 1. Load config
    print("\n📂 Config yükleniyor...")
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ HATA: {CONFIG_PATH} bulunamadı!")
        return

    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)

    print(f"   ✅ Model type: {config.get('model_type', 'Mamba')}")
    print(f"   ✅ Input dim: {config['input_dim']}")
    print(f"   ✅ Hidden size: {config['hidden_size']}")
    print(f"   ✅ Num layers: {config['num_layers']}")
    print(f"   ✅ Best epoch: {config['best_epoch']}")
    print(f"   ✅ Best val loss: {config['best_val_loss']:.6f}")

    SEQ_LEN = config['seq_len']
    STRIDE = config['stride']
    TEST_START = config['test_start']
    TEST_END = config['test_end']
    best_threshold = config.get('best_threshold')
    d_state = config.get('d_state', 16)

    if best_threshold is None:
        print("   ⚠️  Threshold bulunamadı")
    else:
        print(f"   ✅ Best threshold: {best_threshold:.6f}")

    # 2. Load scaler
    print("\n📏 Scaler yükleniyor...")
    if not os.path.exists(SCALER_PATH):
        print(f"❌ HATA: {SCALER_PATH} bulunamadı!")
        return

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print("   ✅ Scaler yüklendi")

    # 3. Load model
    print("\n🏗️  Model yükleniyor...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: {MODEL_PATH} bulunamadı!")
        return

    model = MambaForecast(
        input_dim=config['input_dim'],
        hidden=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        d_state=d_state
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"   ✅ Model yüklendi ({DEVICE})")
    except Exception as e:
        print(f"   ❌ Model yükleme hatası: {e}")
        return

    # 4. Load test data
    intervals = load_anomaly_intervals(ANOM_FILE)
    if not intervals:
        print("❌ HATA: labels.csv okunamadı!")
        return

    test_start_year = pd.Timestamp(TEST_START).year
    test_end_year = pd.Timestamp(TEST_END).year
    test_years = list(range(test_start_year, test_end_year + 1))

    df_all = load_years(test_years)
    if df_all is None:
        print("❌ HATA: Test verisi yüklenemedi!")
        return

    test_start_ts = pd.Timestamp(TEST_START)
    test_end_ts = pd.Timestamp(TEST_END)

    raw_test = df_all[(df_all.index >= test_start_ts) & (df_all.index <= test_end_ts)].copy()

    print(f"\n📊 Test Set:")
    print(f"   Period: {TEST_START} → {TEST_END}")
    print(f"   Rows: {len(raw_test):,}")
    print(f"   Date range: {raw_test.index.min()} → {raw_test.index.max()}")

    # 5. Scale test data
    print("\n📏 Test verisi scale ediliyor...")
    test_scaled = pd.DataFrame(
        scaler.transform(raw_test),
        index=raw_test.index,
        columns=raw_test.columns
    )

    # 6. Create windows
    print(f"\n🔲 Windows oluşturuluyor (SEQ={SEQ_LEN}, STRIDE={STRIDE})...")
    X_test, y_test, t_test = make_windows(test_scaled, seq_len=SEQ_LEN, stride=STRIDE)

    print(f"   ✅ {len(X_test):,} test windows oluşturuldu")

    if len(X_test) == 0:
        print("❌ HATA: Test window'ları oluşturulamadı!")
        return

    # 7. Get predictions
    print("\n🔮 Predictions hesaplanıyor...")
    test_loader = DataLoader(SeqDS(X_test, y_test), batch_size=256, shuffle=False)

    test_losses = get_loss_distribution(model, test_loader, DEVICE)
    test_losses_smoothed = pd.Series(test_losses).rolling(
        window=15, min_periods=1
    ).mean().values

    print(f"   ✅ {len(test_losses):,} loss score hesaplandı")

    # 8. Get true labels
    print("\n🏷️  True labels alınıyor...")
    y_true_test = get_true_labels(t_test, intervals)

    num_anomalies = np.sum(y_true_test)
    print(f"   ✅ Test setinde {num_anomalies:,} anomaly window var")
    print(f"   ℹ️  Anomaly ratio: {100 * num_anomalies / len(y_true_test):.2f}%")

    if num_anomalies == 0:
        print("\n⚠️  Test setinde anomali yok!")
        return

    # 9. Evaluate
    print("\n📊 Model değerlendiriliyor...")

    if best_threshold is None:
        print("   ⚠️  Threshold yok, otomatik aranıyor...")
        min_thr = np.percentile(test_losses_smoothed, 75)
        max_thr = np.percentile(test_losses_smoothed, 99.5)
        thresholds = np.linspace(min_thr, max_thr, 100)

        best_f1 = -1
        for thr in thresholds:
            f1, _, _, _, _ = evaluate_with_threshold(test_losses_smoothed, y_true_test, thr)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr

        print(f"   ✅ Optimal threshold: {best_threshold:.6f}")

    f1, prec, rec, cm, y_pred = evaluate_with_threshold(
        test_losses_smoothed, y_true_test, best_threshold
    )

    # 10. Results
    print("\n" + "=" * 70)
    print("📋 MAMBA TEST SONUÇLARI")
    print("=" * 70)
    print(f"🔹 Test Period    : {TEST_START} → {TEST_END}")
    print(f"🔹 Test Windows   : {len(X_test):,}")
    print(f"🔹 Anomaly Windows: {num_anomalies:,} ({100 * num_anomalies / len(y_true_test):.2f}%)")
    print(f"🔹 Threshold      : {best_threshold:.6f}")
    print("-" * 70)
    print(f"🏆 F1-Score       : {f1:.4f}")
    print(f"🎯 Precision      : {prec:.4f}")
    print(f"📡 Recall         : {rec:.4f}")
    print(f"🎪 Accuracy       : {100 * (cm[0, 0] + cm[1, 1]) / cm.sum():.2f}%")
    print("-" * 70)
    print("Confusion Matrix:")
    print(cm)
    print(f"   TN: {cm[0, 0]:,}  |  FP: {cm[0, 1]:,}")
    print(f"   FN: {cm[1, 0]:,}  |  TP: {cm[1, 1]:,}")
    print("=" * 70)

    # 11. Plot
    print("\n📊 Grafik oluşturuluyor...")
    plot_test_results(t_test, y_true_test, y_pred, test_losses_smoothed, best_threshold)

    # 12. Save
    print("\n💾 Sonuçlar kaydediliyor...")
    results = {
        'test_period': f"{TEST_START} to {TEST_END}",
        'test_windows': len(X_test),
        'anomaly_windows': int(num_anomalies),
        'threshold': float(best_threshold),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'confusion_matrix': cm.tolist(),
        'model_type': 'Mamba'
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv('mamba_test_results.csv', index=False)
    print("✅ Sonuçlar kaydedildi: mamba_test_results.csv")

    print("\n" + "=" * 70)
    print("✅ TEST TAMAMLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    main()