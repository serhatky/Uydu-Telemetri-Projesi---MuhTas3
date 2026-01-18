import pandas as pd
import time
from pathlib import Path
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# =====================================
# 1️⃣ PATH AYARLARI
# =====================================
BASE_DIR = r"C:\Users\computer\Desktop\uydu telemetri"
CHANNEL_DIR = os.path.join(BASE_DIR, "channels")
TELECOMMAND_DIR = os.path.join(BASE_DIR, "telecommands")
ANOMALY_LABEL_FILE = os.path.join(BASE_DIR, "labels.csv")
ANOMALY_META_FILE = os.path.join(BASE_DIR, "anomaly_types.csv")
CHANNEL_CSV = os.path.join(BASE_DIR, "channels.csv")
TELECOMMAND_CSV = os.path.join(BASE_DIR, "telecommands.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_channels")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# 2️⃣ PARAMETRELER
# =====================================
ANOMALY_TOLERANCE = pd.Timedelta("2min")
COMMAND_TOLERANCE = pd.Timedelta("3min")
CHANNEL_MERGE_TOLERANCE = pd.Timedelta("1min")

# =====================================
# 3️⃣ CHANNEL YÜKLEME
# =====================================
def load_channel_data(path_or_name):
    try:
        if path_or_name.endswith(".csv"):
            df = pd.read_csv(path_or_name)
        else:
            df = pd.read_pickle(path_or_name)
    except Exception as e:
        print(f"❌ {path_or_name} okunamadı: {e}")
        return None

    # datetime düzenleme
    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'datetime'}, inplace=True)
        elif "timestamp" in df.columns:
            df.rename(columns={"timestamp": "datetime"}, inplace=True)
        else:
            print(f"⚠️ {path_or_name} datetime bilgisi yok, atlanıyor.")
            return None

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)

    # channel sütunu
    ch_cols = [c for c in df.columns if c not in ["datetime"]]
    if not ch_cols:
        print(f"⚠️ {path_or_name} içinde ölçüm kolonu yok.")
        return None
    ch_col = ch_cols[0]
    df.rename(columns={ch_col: Path(path_or_name).stem}, inplace=True)

    return df

# =====================================
# 4️⃣ ANOMALİLERİ YÜKLEME
# =====================================
def load_anomaly_labels(path, channel_name):
    try:
        labels = pd.read_csv(path)
    except Exception as e:
        print(f"❌ labels.csv okunamadı: {e}")
        return pd.DataFrame(columns=["timestamp", "anomaly_type"])

    pure_name = Path(channel_name).stem.split("_final")[0]
    labels = labels[labels["Channel"] == pure_name].copy()
    if labels.empty:
        return pd.DataFrame(columns=["timestamp", "anomaly_type"])

    labels["StartTime"] = pd.to_datetime(labels["StartTime"], utc=True, errors="coerce")
    labels["EndTime"] = pd.to_datetime(labels["EndTime"], utc=True, errors="coerce")

    records = []
    for _, row in labels.iterrows():
        if pd.isna(row["StartTime"]) or pd.isna(row["EndTime"]):
            continue
        try:
            rng = pd.date_range(start=row["StartTime"], end=row["EndTime"], freq="1min", tz="UTC")
            for t in rng:
                records.append({"timestamp": t, "anomaly_type": row["ID"]})
        except Exception:
            continue

    return pd.DataFrame(records)

# =====================================
# 5️⃣ TELECOMMANDLARI YÜKLEME
# =====================================
def load_telecommands():
    print("📡 Telecommand verileri yükleniyor...")

    all_cmds = []
    files = list(Path(TELECOMMAND_DIR).glob("telecommand_*"))
    for f in files:
        try:
            df = pd.read_pickle(f)
            df.index = pd.to_datetime(df.index).tz_localize("UTC")
            col = df.columns[0]
            active = df[df[col] == 1].copy()
            if active.empty:
                continue
            active["timestamp"] = active.index
            active["command_type"] = col
            all_cmds.append(active[["timestamp", "command_type"]])
        except Exception:
            continue

    if not all_cmds and os.path.exists(TELECOMMAND_CSV):
        print("⚙️ telecommands klasörü boş, CSV okunuyor...")
        tele = pd.read_csv(TELECOMMAND_CSV)
        if "timestamp" not in tele.columns:
            raise ValueError("telecommands.csv içinde 'timestamp' kolonu yok.")
        tele["timestamp"] = pd.to_datetime(tele["timestamp"], utc=True, errors="coerce")
        tele = tele[["timestamp", "command_type"]]
        return tele

    if not all_cmds:
        print("⚠️ Hiç telecommand bulunamadı.")
        return pd.DataFrame(columns=["timestamp", "command_type"])

    tele = pd.concat(all_cmds).sort_values("timestamp")
    print(f"✅ {len(tele):,} telecommand yüklendi.")
    return tele

# =====================================
# 6️⃣ TEK CHANNEL İŞLEME
# =====================================
def process_single_channel(channel_path, tele_df, anomaly_labels_path, anomaly_meta_path):
    channel_name = Path(channel_path).stem
    print(f"\n🚀 {channel_name} işleniyor...")

    ch = load_channel_data(channel_path)
    if ch is None or ch.empty:
        print(f"⚠️ {channel_name} boş, atlandı.")
        return None

    # 🔹 ANOMALY MERGE
    labels = load_anomaly_labels(anomaly_labels_path, channel_name)
    if labels.empty:
        ch["anomaly_type"] = "Normal"
    else:
        ch = pd.merge_asof(
            ch.sort_values("datetime"),
            labels.sort_values("timestamp"),
            left_on="datetime",
            right_on="timestamp",
            direction="backward",
            tolerance=ANOMALY_TOLERANCE
        )
        ch["anomaly_type"].fillna("Normal", inplace=True)
        ch.drop(columns=["timestamp"], inplace=True, errors="ignore")

    ch["anomaly_flag"] = (ch["anomaly_type"] != "Normal").astype(int)

    # 🔹 TELECOMMAND MERGE
    if not tele_df.empty:
        ch = pd.merge_asof(
            ch.sort_values("datetime"),
            tele_df.sort_values("timestamp"),
            left_on="datetime",
            right_on="timestamp",
            direction="backward",
            tolerance=COMMAND_TOLERANCE
        )
        ch["last_command_minutes_ago"] = (ch["datetime"] - ch["timestamp"]).dt.total_seconds() / 60
        ch["last_command_minutes_ago"].fillna(-1, inplace=True)
        ch["last_command_type"] = ch["command_type"].fillna("NoCommand")
        ch["any_telecommand_flag"] = (ch["last_command_type"] != "NoCommand").astype(int)
        ch.drop(columns=["timestamp", "command_type"], inplace=True, errors="ignore")
    else:
        ch["last_command_minutes_ago"] = -1
        ch["last_command_type"] = "NoCommand"
        ch["any_telecommand_flag"] = 0

    # 🔹 META EKLEME
    try:
        meta = pd.read_csv(anomaly_meta_path)
        meta["Subclass"] = meta["Subclass"].fillna("Subclass_Unknown")
        ch = pd.merge(ch, meta, left_on="anomaly_type", right_on="ID", how="left")
        ch.drop(columns=["ID"], inplace=True, errors="ignore")
    except Exception as e:
        print(f"⚠️ Meta yüklenemedi: {e}")

    fill_values = {
        "Class": "N/A",
        "Subclass": "N/A",
        "Category": "N/A",
        "Dimensionality": "N/A",
        "Locality": "N/A",
        "Length": "N/A",
        "Subsystem": "N/A",
        "Physical Unit": "N/A",
        "Group": -1,
        "Target": "N/A",
        "Categorical": "N/A"
    }
    ch.fillna(value=fill_values, inplace=True)

    # 🔹 KAYIT
    out_path = os.path.join(OUTPUT_DIR, f"{channel_name}_final_enriched.parquet")
    ch.set_index("datetime", inplace=True)
    ch.to_parquet(out_path)
    print(f"💾 Kaydedildi: {out_path}")

    return {
        "Channel": channel_name,
        "Rows": len(ch),
        "Anomalies": ch["anomaly_flag"].sum(),
        "Telecommands": ch["any_telecommand_flag"].sum()
    }

# =====================================
# 7️⃣ TÜM CHANNEL’LAR PIPELINE
# =====================================
def main():
    print("="*70)
    print("🚀 TÜM CHANNEL'LAR İÇİN ENRICHMENT PIPELINE (SAYISAL SIRALAMA)")
    print("="*70)
    t0 = time.time()

    tele_df = load_telecommands()

    # Channel dosyalarını oku
    channel_files = []
    if os.path.exists(CHANNEL_DIR):
        channel_files += [
            str(Path(CHANNEL_DIR) / f)
            for f in os.listdir(CHANNEL_DIR)
            if f.startswith("channel_")
        ]
    if os.path.exists(CHANNEL_CSV):
        channel_files.append(CHANNEL_CSV)

    if not channel_files:
        print("⚠️ Hiç channel dosyası bulunamadı!")
        return

    # 🧩 Sayısal sıralama (channel_1, channel_2, ... channel_76)
    def extract_num(name):
        try:
            return int(Path(name).stem.split("_")[1])
        except:
            return 9999

    channel_files = sorted(channel_files, key=extract_num)

    summary = []
    for ch_path in tqdm(channel_files, desc="🔄 Channel işleniyor", ncols=100):
        try:
            info = process_single_channel(ch_path, tele_df, ANOMALY_LABEL_FILE, ANOMALY_META_FILE)
            if info:
                summary.append(info)
        except Exception as e:
            print(f"❌ {ch_path} hata: {e}")

    if summary:
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(os.path.join(OUTPUT_DIR, "merge_summary.csv"), index=False)
        print("\n📊 ÖZET TABLO:")
        print(df_summary)
    else:
        print("\n⚠️ Hiç channel işlenemedi!")

    print(f"\n⏰ Toplam süre: {time.time() - t0:.2f} sn")
    print("🎉 Pipeline başarıyla tamamlandı!")

# =====================================
if __name__ == "__main__":
    main()