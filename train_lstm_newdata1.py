"""
LSTM TRAINING - OVERFITTING FIX + EARLY STOPPING + PLOTTING
============================================================
Önceki eğitimdeki train > val loss sorununu düzeltir:
- Early Stopping
- Daha güçlü regularization
- Learning Rate scheduling
- Eğitim grafikleri (loss, F1, precision, recall)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import psutil
import gc
warnings.filterwarnings("ignore")

# =========================
# 📂 PATHS
# =========================
BASE_DIR = r"D:\Esa_Xai_Dataset"
SPLIT_DIR = os.path.join(BASE_DIR, "final_splitsV2")
OUTPUT_DIR = os.path.join(BASE_DIR, "lstm_outputs_last")  # Yeni output klasörü

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(SPLIT_DIR, "train", "train.parquet")
VAL_FILE = os.path.join(SPLIT_DIR, "validation", "validation.parquet")
TEST_FILE = os.path.join(SPLIT_DIR, "test", "test.parquet")
METADATA_FILE = os.path.join(SPLIT_DIR, "split_metadata.json")

# =========================
# ⚙️ HYPERPARAMETERS - OPTIMIZED
# =========================
SEQ_LEN = 120
STRIDE = 30
BATCH_SIZE = 256  # Küçültüldü - daha iyi generalization
HIDDEN_SIZE = 32  # Küçültüldü - overfitting azaltmak için
NUM_LAYERS = 2
DROPOUT = 0.25 
LR = 5e-4  # Düşürüldü (1e-3 -> 5e-4)
EPOCHS = 25  # Daha fazla epoch, early stopping ile kontrol
WEIGHT_DECAY = 1e-4  # Artırıldı (1e-5 -> 1e-4)
GRAD_CLIP = 1.0

# EARLY STOPPING
PATIENCE = 7  # 7 epoch boyunca iyileşme yoksa dur
MIN_DELTA = 1e-4  # Minimum iyileşme eşiği

# CHUNK SIZE - RAM'e göre ayarla
CHUNK_SIZE = 50000

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ GPU (CUDA) kullanılıyor!")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ GPU (MPS) kullanılıyor!")
else:
    DEVICE = "cpu"
    print("⚠️  CPU kullanılıyor")

print(f"🎯 Device: {DEVICE}")

# =========================
# 🛑 EARLY STOPPING CLASS
# =========================

class EarlyStopping:
    """
    Early stopping based on validation loss
    """
    def __init__(self, patience=7, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, current_value, epoch):
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            print(f"   ⏳ EarlyStopping: {self.counter}/{self.patience} (best: epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False

# =========================
# 🔍 SYSTEM MONITOR
# =========================
class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
        
    def get_ram_usage(self):
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': mem.used / 1024**3,
            'available_gb': mem.available / 1024**3
        }
    
    def get_gpu_usage(self):
        if DEVICE == "cuda":
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3
            }
        return None
    
    def log_status(self, prefix=""):
        ram = self.get_ram_usage()
        elapsed = time.time() - self.start_time
        
        log = f"{prefix}RAM: {ram['percent']:.1f}% ({ram['used_gb']:.1f}GB)"
        
        if DEVICE == "cuda":
            gpu = self.get_gpu_usage()
            log += f" | GPU: {gpu['allocated_gb']:.2f}GB"
        
        log += f" | Time: {elapsed/60:.1f}min"
        print(log)
        
        if ram['percent'] > 85:
            print("⚠️  RAM UYARI: %85'in üzerinde!")
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

monitor = SystemMonitor()

# =========================
# 📊 FEATURE SELECTION
# =========================
def get_feature_columns(df):
    """Feature kolonlarını belirle"""
    # Channel values
    channel_cols = [c for c in df.columns if c.startswith('channel_')]
    
    # Extra features
    extra_features = []
    for feat in ['global_mean', 'global_std', 'global_max', 'global_min',
                 'hour', 'day_of_week', 'day_of_year',
                 'last_tc_priority', 'seconds_since_tc', 'tc_count']:
        if feat in df.columns:
            extra_features.append(feat)
    
    feature_cols = channel_cols + extra_features
    
    print(f"   📊 Features: {len(channel_cols)} channels + {len(extra_features)} extra = {len(feature_cols)} total")
    
    return feature_cols

# =========================
# 🔲 STREAMING WINDOWING DATASET
# =========================
class StreamingWindowDataset(IterableDataset):
    """
    RAM-efficient streaming dataset with optional augmentation
    """
    def __init__(self, filepath, seq_len, stride, scaler=None, fit_scaler=False, augment=False):
        self.filepath = filepath
        self.seq_len = seq_len
        self.stride = stride
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.augment = augment  # Training için veri augmentation
        
        # İlk chunk'ı oku ve feature cols belirle
        df_sample = pd.read_parquet(filepath)
        df_sample = df_sample.head(1000)
        
        self.feature_cols = get_feature_columns(df_sample)
        self.n_features = len(self.feature_cols)
        
        if 'anomaly' not in df_sample.columns:
            raise ValueError("❌ 'anomaly' kolonu bulunamadı!")
        
        del df_sample
        gc.collect()
    
    def _add_noise(self, x, noise_factor=0.01):
        """Gaussian noise ekle - regularization için"""
        if self.augment:
            noise = np.random.normal(0, noise_factor, x.shape).astype(np.float32)
            return x + noise
        return x
    
    def __iter__(self):
        """Iterator - chunk chunk oku ve window oluştur"""
        
        # Parquet'i chunk chunk oku
        parquet_file = pd.read_parquet(self.filepath)
        total_rows = len(parquet_file)
        
        print(f"   📂 {os.path.basename(self.filepath)}: {total_rows:,} rows")
        
        buffer_X = []
        buffer_y = []
        
        # Chunk'lar halinde işle
        for start_idx in range(0, total_rows, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_rows)
            chunk = parquet_file.iloc[start_idx:end_idx]
            
            # Feature extraction
            X_chunk = chunk[self.feature_cols].values.astype(np.float32)
            y_chunk = chunk['anomaly'].values.astype(np.float32)
            
            # Scaler fit (sadece training için)
            if self.fit_scaler and self.scaler is not None:
                if start_idx == 0:
                    self.scaler.fit(X_chunk)
                    print("   ✅ Scaler fitted on first chunk")
            
            # Scaler transform
            if self.scaler is not None:
                X_chunk = self.scaler.transform(X_chunk)
            
            # Buffer'a ekle
            buffer_X.extend(X_chunk)
            buffer_y.extend(y_chunk)
            
            # Window oluştur
            while len(buffer_X) >= self.seq_len + self.stride:
                # Window al
                X_window = np.array(buffer_X[:self.seq_len], dtype=np.float32)
                y_label = buffer_y[self.seq_len - 1]  # Son timestep'in label'ı
                
                # Training augmentation
                X_window = self._add_noise(X_window)
                
                yield torch.from_numpy(X_window), torch.tensor(y_label, dtype=torch.float32)
                
                # Buffer'ı stride kadar kaydır
                buffer_X = buffer_X[self.stride:]
                buffer_y = buffer_y[self.stride:]
            
            # Memory cleanup
            del chunk, X_chunk, y_chunk
            gc.collect()
        
        # Son kalan window'ları işle
        while len(buffer_X) >= self.seq_len:
            X_window = np.array(buffer_X[:self.seq_len], dtype=np.float32)
            y_label = buffer_y[self.seq_len - 1]
            
            X_window = self._add_noise(X_window)
            
            yield torch.from_numpy(X_window), torch.tensor(y_label, dtype=torch.float32)
            
            buffer_X = buffer_X[self.stride:]
            buffer_y = buffer_y[self.stride:]

# =========================
# 🏗️ LSTM MODEL - WITH MORE REGULARIZATION
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        # Layer Normalization for input
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Tek yönlü - daha basit
        )
        
        # Daha güçlü regularization
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden // 2, 1)
        
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        # Daha derin FC layers
        out = self.dropout1(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        
        return out.squeeze()

# =========================
# 📈 METRICS
# =========================
def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    
    return {'f1': f1, 'precision': prec, 'recall': rec}

# =========================
# 📊 PLOTTING FUNCTIONS
# =========================
def plot_training_history(history, output_dir):
    """Eğitim metriklerini görselleştir ve kaydet"""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
    
    # 1. Loss Plot
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting zone highlight
    train_losses = np.array(history['train_loss'])
    val_losses = np.array(history['val_loss'])
    overfit_mask = val_losses > train_losses * 1.1  # %10'dan fazla fark varsa
    if overfit_mask.any():
        ax1.fill_between(epochs, 0, max(val_losses), where=overfit_mask, 
                        alpha=0.2, color='red', label='Overfit Zone')
    
    # 2. F1 Score Plot
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Precision Plot
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['train_precision'], 'b-', label='Train Precision', linewidth=2, marker='o', markersize=4)
    ax3.plot(epochs, history['val_precision'], 'r-', label='Val Precision', linewidth=2, marker='s', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision over Epochs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Recall Plot
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['train_recall'], 'b-', label='Train Recall', linewidth=2, marker='o', markersize=4)
    ax4.plot(epochs, history['val_recall'], 'r-', label='Val Recall', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall over Epochs')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Plot saved: {plot_path}")
    
    plt.close()
    
    # Ayrıca ayrı loss grafiği
    plot_loss_only(history, output_dir)

def plot_loss_only(history, output_dir):
    """Sadece loss grafiği - daha detaylı"""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2.5, marker='o', markersize=5)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=5)
    
    # Best epoch işaretle
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss\n(Early Stopping Enabled)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Gap annotation
    final_gap = history['val_loss'][-1] - history['train_loss'][-1]
    ax.annotate(f'Final Gap: {final_gap:.4f}', 
                xy=(len(epochs), history['val_loss'][-1]),
                xytext=(len(epochs) - 2, max(history['val_loss']) * 0.9),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📈 Loss curve saved: {plot_path}")
    
    plt.close()

def plot_learning_rate(lr_history, output_dir):
    """Learning rate değişimini göster"""
    if len(lr_history) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'learning_rate.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   📉 LR curve saved: {plot_path}")
        
        plt.close()

# =========================
# 🚀 TRAINING
# =========================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())
        
        # Her 100 batch'te bir memory cleanup
        if batch_count % 100 == 0:
            gc.collect()
    
    avg_loss = total_loss / batch_count
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            batch_count += 1
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / batch_count
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics

# =========================
# 🎯 MAIN
# =========================
def main():
    print("="*70)
    print("🚀 LSTM TRAINING V2 - OPTIMIZED WITH EARLY STOPPING")
    print("="*70)
    
    # System info
    ram = monitor.get_ram_usage()
    print(f"\n💾 System Info:")
    print(f"   RAM Total: {ram['used_gb'] + ram['available_gb']:.1f} GB")
    print(f"   RAM Available: {ram['available_gb']:.1f} GB")
    print(f"   Device: {DEVICE}")
    
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Print hyperparameters
    print(f"\n⚙️  Hyperparameters (OPTIMIZED):")
    print(f"   Hidden Size: {HIDDEN_SIZE} (reduced)")
    print(f"   Dropout: {DROPOUT} (increased)")
    print(f"   Learning Rate: {LR}")
    print(f"   Weight Decay: {WEIGHT_DECAY} (increased)")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Early Stopping Patience: {PATIENCE}")
    
    # Scaler
    print("\n📏 Initializing scaler...")
    scaler = StandardScaler()
    
    # Create streaming datasets
    print("\n" + "="*70)
    print("📂 CREATING STREAMING DATASETS")
    print("="*70)
    
    train_dataset = StreamingWindowDataset(
        TRAIN_FILE, 
        seq_len=SEQ_LEN, 
        stride=STRIDE,
        scaler=scaler,
        fit_scaler=True,
        augment=True  # Training için augmentation açık
    )
    
    val_dataset = StreamingWindowDataset(
        VAL_FILE,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        scaler=scaler,
        fit_scaler=False,
        augment=False  # Validation için augmentation kapalı
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    print(f"   ✅ Streaming dataloaders hazır")
    print(f"   📊 Batch size: {BATCH_SIZE}")
    print(f"   🔲 Sequence length: {SEQ_LEN}")
    print(f"   👣 Stride: {STRIDE}")
    
    # Model
    print("\n" + "="*70)
    print("🏗️  BUILDING MODEL")
    print("="*70)
    
    input_dim = train_dataset.n_features
    model = LSTMClassifier(input_dim=input_dim).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   📊 Model Parameters: {total_params:,}")
    print(f"   📐 Input dim: {input_dim}")
    print(f"   🧠 Hidden size: {HIDDEN_SIZE}")
    print(f"   📚 Layers: {NUM_LAYERS}")
    
    # Loss & optimizer
    pos_weight = torch.tensor([9.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # AdamW
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, mode='min')
    
    print(f"   ⚖️  Pos weight: {pos_weight.item():.1f}")
    print(f"   🛑 Early stopping: patience={PATIENCE}")
    
    # Training loop
    print("\n" + "="*70)
    print(f"🎓 TRAINING (max {EPOCHS} EPOCHS with Early Stopping)")
    print("="*70)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        print("🏋️  Training...")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        print("🔍 Validating...")
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)
        
        # Print
        print(f"\n📊 Results:")
        print(f"   Loss    : Train {train_loss:.6f} | Val {val_loss:.6f} | Gap {val_loss - train_loss:.6f}")
        print(f"   F1      : Train {train_metrics['f1']:.4f} | Val {val_metrics['f1']:.4f}")
        print(f"   Prec    : Train {train_metrics['precision']:.4f} | Val {val_metrics['precision']:.4f}")
        print(f"   Recall  : Train {train_metrics['recall']:.4f} | Val {val_metrics['recall']:.4f}")
        print(f"   LR      : {current_lr:.2e}")
        
        # Overfitting warning
        if val_loss > train_loss * 1.2:
            print(f"   ⚠️  OVERFITTING DETECTED: Val/Train ratio = {val_loss/train_loss:.2f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Best model (based on val_loss for stability)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"   ⭐ Best model saved! (Val Loss: {best_val_loss:.6f}, F1: {best_val_f1:.4f})")
        
        # Early stopping check
        if early_stopping(val_loss, epoch):
            print(f"\n🛑 Early stopping triggered at epoch {epoch}!")
            print(f"   Best epoch was: {early_stopping.best_epoch}")
            break
        
        # Time
        epoch_time = time.time() - epoch_start
        print(f"\n⏱️  Epoch time: {epoch_time:.1f}s")
        monitor.log_status("   ")
        
        # Memory cleanup
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    # Complete
    print("\n" + "="*70)
    print("✅ TRAINING TAMAMLANDI!")
    print("="*70)
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"🏆 Best Val Loss: {best_val_loss:.6f}")
    print(f"🏆 Best Val F1: {best_val_f1:.4f}")
    print(f"📊 Total epochs trained: {len(history['train_loss'])}")
    
    # Plot training history
    print("\n📊 Generating plots...")
    plot_training_history(history, OUTPUT_DIR)
    plot_learning_rate(history['lr'], OUTPUT_DIR)
    
    # Save artifacts
    print("\n💾 Saving artifacts...")
    
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("   ✅ scaler.pkl")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)
    print("   ✅ training_history.csv")
    
    config = {
        'seq_len': SEQ_LEN,
        'stride': STRIDE,
        'batch_size': BATCH_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'epochs_trained': len(history['train_loss']),
        'max_epochs': EPOCHS,
        'input_dim': input_dim,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1,
        'early_stopping_patience': PATIENCE,
        'chunk_size': CHUNK_SIZE
    }
    
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("   ✅ config.json")
    
    print(f"\n📁 Çıktılar: {OUTPUT_DIR}")
    print("\n📊 Generated plots:")
    print(f"   - training_history.png (4 subplots)")
    print(f"   - loss_curve.png (detailed loss)")
    print(f"   - learning_rate.png (LR schedule)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 Cleanup complete")
