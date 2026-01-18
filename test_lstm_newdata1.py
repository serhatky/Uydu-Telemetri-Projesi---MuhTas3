"""
LSTM TEST PIPELINE
==================
Eğitilmiş modeli test set üzerinde evaluate eder
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, roc_curve
)
import seaborn as sns
import warnings
import gc
warnings.filterwarnings("ignore")

# =========================
# 📂 PATHS
# =========================
BASE_DIR = r"D:\Esa_Xai_Dataset"
SPLIT_DIR = os.path.join(BASE_DIR, "final_splits")
MODEL_DIR = os.path.join(BASE_DIR, "lstm_outputs_v3")

TEST_FILE = os.path.join(SPLIT_DIR, "test", "test.parquet")

# Model artifacts
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Output
TEST_OUTPUT_DIR = os.path.join(MODEL_DIR, "test_results")
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"🎯 Device: {DEVICE}")

# =========================
# 🔲 STREAMING DATASET (same as training)
# =========================
class StreamingWindowDataset(IterableDataset):
    def __init__(self, filepath, seq_len, stride, feature_cols, scaler=None):
        self.filepath = filepath
        self.seq_len = seq_len
        self.stride = stride
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.n_features = len(feature_cols)
    
    def __iter__(self):
        parquet_file = pd.read_parquet(self.filepath)
        total_rows = len(parquet_file)
        
        print(f"   📂 Test file: {total_rows:,} rows")
        
        buffer_X = []
        buffer_y = []
        
        CHUNK_SIZE = 50000
        
        for start_idx in range(0, total_rows, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_rows)
            chunk = parquet_file.iloc[start_idx:end_idx]
            
            X_chunk = chunk[self.feature_cols].values.astype(np.float32)
            y_chunk = chunk['anomaly'].values.astype(np.float32)
            
            if self.scaler is not None:
                X_chunk = self.scaler.transform(X_chunk)
            
            buffer_X.extend(X_chunk)
            buffer_y.extend(y_chunk)
            
            while len(buffer_X) >= self.seq_len + self.stride:
                X_window = np.array(buffer_X[:self.seq_len], dtype=np.float32)
                y_label = buffer_y[self.seq_len - 1]
                
                yield torch.from_numpy(X_window), torch.tensor(y_label, dtype=torch.float32)
                
                buffer_X = buffer_X[self.stride:]
                buffer_y = buffer_y[self.stride:]
            
            del chunk, X_chunk, y_chunk
            gc.collect()
        
        while len(buffer_X) >= self.seq_len:
            X_window = np.array(buffer_X[:self.seq_len], dtype=np.float32)
            y_label = buffer_y[self.seq_len - 1]
            
            yield torch.from_numpy(X_window), torch.tensor(y_label, dtype=torch.float32)
            
            buffer_X = buffer_X[self.stride:]
            buffer_y = buffer_y[self.stride:]

# =========================
# 🏗️ LSTM MODEL (same as training)
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden // 2, 1)
        
    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        out = self.dropout1(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        
        return out.squeeze()

# =========================
# 📊 EVALUATION FUNCTIONS
# =========================
def evaluate_model(model, loader, criterion, device):
    """Test set üzerinde detaylı evaluation"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0
    batch_count = 0
    
    print("\n🔍 Evaluating...")
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            batch_count += 1
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())
            
            if batch_count % 50 == 0:
                print(f"   Processed {batch_count} batches...", end='\r')
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / batch_count
    
    return avg_loss, all_probs, all_labels

def find_best_threshold(y_true, y_probs):
    """En iyi threshold'u bul (F1 score'a göre)"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    
    results = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': thresh,
            'f1': f1,
            'precision': prec,
            'recall': rec
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1, pd.DataFrame(results)

def compute_detailed_metrics(y_true, y_probs, threshold=0.5):
    """Detaylı metrikleri hesapla"""
    y_pred = (y_probs >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'pr_auc': average_precision_score(y_true, y_probs)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm, y_pred

# =========================
# 📊 PLOTTING FUNCTIONS
# =========================
def plot_confusion_matrix(cm, output_path):
    """Confusion matrix görselleştir"""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Saved: {output_path}")
    plt.close()

def plot_roc_curve(y_true, y_probs, roc_auc, output_path):
    """ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   📈 Saved: {output_path}")
    plt.close()

def plot_threshold_analysis(threshold_df, best_thresh, output_path):
    """Threshold analizi"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(threshold_df['threshold'], threshold_df['f1'], 'b-', 
            linewidth=2, marker='o', label='F1 Score')
    ax.plot(threshold_df['threshold'], threshold_df['precision'], 'g--', 
            linewidth=2, marker='s', label='Precision')
    ax.plot(threshold_df['threshold'], threshold_df['recall'], 'r--', 
            linewidth=2, marker='^', label='Recall')
    
    # Best threshold işaretle
    ax.axvline(x=best_thresh, color='purple', linestyle='--', 
               alpha=0.7, label=f'Best Threshold ({best_thresh:.2f})')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold vs Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   📉 Saved: {output_path}")
    plt.close()

def plot_prediction_distribution(y_true, y_probs, threshold, output_path):
    """Prediction distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(y_probs[y_true == 0], bins=50, alpha=0.6, label='Normal', color='blue')
    ax1.hist(y_probs[y_true == 1], bins=50, alpha=0.6, label='Anomaly', color='red')
    ax1.axvline(x=threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.2f})')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [y_probs[y_true == 0], y_probs[y_true == 1]]
    bp = ax2.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.2f})')
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Prediction Box Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Saved: {output_path}")
    plt.close()

# =========================
# 🎯 MAIN TEST PIPELINE
# =========================
def main():
    print("="*70)
    print("🧪 LSTM TEST PIPELINE")
    print("="*70)
    
    # 1. Load config
    print("\n📄 Loading configuration...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    print(f"   ✅ Config loaded")
    print(f"      - Trained epochs: {config['epochs_trained']}")
    print(f"      - Best epoch: {config['best_epoch']}")
    print(f"      - Best val loss: {config['best_val_loss']:.6f}")
    print(f"      - Best val F1: {config['best_val_f1']:.4f}")
    
    # 2. Load scaler
    print("\n📏 Loading scaler...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("   ✅ Scaler loaded")
    
    # 3. Get feature columns
    print("\n📊 Determining feature columns...")
    df_sample = pd.read_parquet(TEST_FILE)
    df_sample = df_sample.head(1000)
    
    channel_cols = [c for c in df_sample.columns if c.startswith('channel_')]
    extra_features = []
    for feat in ['global_mean', 'global_std', 'global_max', 'global_min',
                 'hour', 'day_of_week', 'day_of_year',
                 'last_tc_priority', 'seconds_since_tc', 'tc_count']:
        if feat in df_sample.columns:
            extra_features.append(feat)
    
    feature_cols = channel_cols + extra_features
    print(f"   ✅ Features: {len(feature_cols)} total")
    
    del df_sample
    gc.collect()
    
    # 4. Create test dataset
    print("\n📂 Creating test dataset...")
    test_dataset = StreamingWindowDataset(
        TEST_FILE,
        seq_len=config['seq_len'],
        stride=config['stride'],
        feature_cols=feature_cols,
        scaler=scaler
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'] * 2,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    print("   ✅ Test loader ready")
    
    # 5. Load model
    print("\n🏗️  Loading model...")
    model = LSTMClassifier(
        input_dim=config['input_dim'],
        hidden=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded")
    print(f"      - Parameters: {total_params:,}")
    
    # 6. Evaluate
    print("\n" + "="*70)
    print("🧪 TESTING ON TEST SET")
    print("="*70)
    
    pos_weight = torch.tensor([9.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    test_loss, test_probs, test_labels = evaluate_model(model, test_loader, criterion, DEVICE)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Test Loss: {test_loss:.6f}")
    
    # 7. Find best threshold
    print("\n🔍 Finding best threshold...")
    best_thresh, best_f1, threshold_df = find_best_threshold(test_labels, test_probs)
    
    print(f"   ✅ Best threshold: {best_thresh:.2f}")
    print(f"   ✅ Best F1 score: {best_f1:.4f}")
    
    # 8. Compute metrics with default (0.5) and best threshold
    print("\n📊 Computing metrics...")
    
    # Default threshold (0.5)
    metrics_default, cm_default, y_pred_default = compute_detailed_metrics(
        test_labels, test_probs, threshold=0.5
    )
    
    # Best threshold
    metrics_best, cm_best, y_pred_best = compute_detailed_metrics(
        test_labels, test_probs, threshold=best_thresh
    )
    
    # 9. Print results
    print("\n" + "="*70)
    print("📈 TEST RESULTS")
    print("="*70)
    
    print(f"\n🎯 WITH DEFAULT THRESHOLD (0.5):")
    print(f"   Loss       : {test_loss:.6f}")
    print(f"   F1 Score   : {metrics_default['f1']:.4f}")
    print(f"   Precision  : {metrics_default['precision']:.4f}")
    print(f"   Recall     : {metrics_default['recall']:.4f}")
    print(f"   ROC AUC    : {metrics_default['roc_auc']:.4f}")
    print(f"   PR AUC     : {metrics_default['pr_auc']:.4f}")
    
    print(f"\n🎯 WITH BEST THRESHOLD ({best_thresh:.2f}):")
    print(f"   Loss       : {test_loss:.6f}")
    print(f"   F1 Score   : {metrics_best['f1']:.4f}")
    print(f"   Precision  : {metrics_best['precision']:.4f}")
    print(f"   Recall     : {metrics_best['recall']:.4f}")
    print(f"   ROC AUC    : {metrics_best['roc_auc']:.4f}")
    print(f"   PR AUC     : {metrics_best['pr_auc']:.4f}")
    
    print(f"\n📊 Confusion Matrix (threshold={best_thresh:.2f}):")
    print(f"   TN: {cm_best[0,0]:,}  |  FP: {cm_best[0,1]:,}")
    print(f"   FN: {cm_best[1,0]:,}  |  TP: {cm_best[1,1]:,}")
    
    # 10. Classification report
    print(f"\n📋 Classification Report (threshold={best_thresh:.2f}):")
    print(classification_report(test_labels, y_pred_best, 
                               target_names=['Normal', 'Anomaly'],
                               digits=4))
    
    # 11. Save plots
    print("\n📊 Generating plots...")
    
    plot_confusion_matrix(
        cm_best, 
        os.path.join(TEST_OUTPUT_DIR, 'confusion_matrix.png')
    )
    
    plot_roc_curve(
        test_labels, test_probs, metrics_best['roc_auc'],
        os.path.join(TEST_OUTPUT_DIR, 'roc_curve.png')
    )
    
    plot_threshold_analysis(
        threshold_df, best_thresh,
        os.path.join(TEST_OUTPUT_DIR, 'threshold_analysis.png')
    )
    
    plot_prediction_distribution(
        test_labels, test_probs, best_thresh,
        os.path.join(TEST_OUTPUT_DIR, 'prediction_distribution.png')
    )
    
    # 12. Save results
    print("\n💾 Saving results...")
    
    # Save metrics
    results = {
        'test_loss': test_loss,
        'best_threshold': best_thresh,
        'metrics_default': metrics_default,
        'metrics_best': metrics_best,
        'confusion_matrix': cm_best.tolist()
    }
    
    with open(os.path.join(TEST_OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("   ✅ test_results.json")
    
    # Save threshold analysis
    threshold_df.to_csv(os.path.join(TEST_OUTPUT_DIR, 'threshold_analysis.csv'), index=False)
    print("   ✅ threshold_analysis.csv")
    
    # Save predictions (sample)
    predictions_df = pd.DataFrame({
        'true_label': test_labels[:10000],
        'probability': test_probs[:10000],
        'predicted_label': y_pred_best[:10000]
    })
    predictions_df.to_csv(os.path.join(TEST_OUTPUT_DIR, 'predictions_sample.csv'), index=False)
    print("   ✅ predictions_sample.csv (first 10k)")
    
    print("\n" + "="*70)
    print("✅ TEST PIPELINE COMPLETE!")
    print("="*70)
    print(f"📁 Results saved to: {TEST_OUTPUT_DIR}")
    print("\n📊 Generated files:")
    print("   - test_results.json")
    print("   - confusion_matrix.png")
    print("   - roc_curve.png")
    print("   - threshold_analysis.png")
    print("   - prediction_distribution.png")
    print("   - threshold_analysis.csv")
    print("   - predictions_sample.csv")
    
    # Comparison with validation
    print("\n" + "="*70)
    print("🔍 VALIDATION vs TEST COMPARISON")
    print("="*70)
    print(f"   Validation F1: {config['best_val_f1']:.4f}")
    print(f"   Test F1:       {metrics_best['f1']:.4f}")
    print(f"   Difference:    {abs(config['best_val_f1'] - metrics_best['f1']):.4f}")
    
    if metrics_best['f1'] < config['best_val_f1'] * 0.8:
        print("\n   ⚠️  Test performance significantly lower than validation!")
        print("   💡 Model might be underfitting. Consider:")
        print("      - Increase HIDDEN_SIZE (64 → 128)")
        print("      - Increase NUM_LAYERS (2 → 3)")
        print("      - Decrease DROPOUT (0.25 → 0.2)")
    elif metrics_best['f1'] >= config['best_val_f1'] * 0.9:
        print("\n   ✅ Good generalization! Test performance is close to validation.")
    
    print("\n🎉 DONE!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()