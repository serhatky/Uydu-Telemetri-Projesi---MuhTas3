import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =========================
# 📂 PATHLER
# =========================
BASE_DIR = r"C:\Users\computer\Desktop\uydu telemetri"
INPUT_DIR = os.path.join(BASE_DIR, "processed_channels")
OUTPUT_DIR = os.path.join(BASE_DIR, "multichannel_by_year")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# ⚙️ PARAMETRELER
# =========================
YEARS = list(range(2000, 2014))
TIME_STEP = "1min"               # zaman çözünürlüğü (örneğin: "10s" veya "5min")
TOLERANCE = pd.Timedelta("30s")  # channel zaman eşleştirme toleransı

# =========================
# 🧩 CHANNEL YÜKLEME
# =========================
def load_channel(filepath):
    try:
        df = pd.read_parquet(filepath)
        if "datetime" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df.dropna(subset=["datetime"], inplace=True)

        # sadece ana kolonları koru
        ch_cols = [c for c in df.columns if c.startswith("channel_")]
        if not ch_cols:
            print(f"⚠️ {filepath} içinde channel_* kolonu yok.")
            return None
        ch = ch_cols[0]

        df = df[["datetime", ch, "anomaly_flag", "any_telecommand_flag"]].sort_values("datetime")
        return df
    except Exception as e:
        print(f"⚠️ {filepath} okunamadı: {e}")
        return None

# =========================
# 🧭 ORTAK ZAMAN INDEX
# =========================
def get_time_range(files, year):
    min_t, max_t = None, None
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["datetime"])
            if "datetime" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df.dropna(subset=["datetime"], inplace=True)
            df = df[df["datetime"].dt.year == year]
            if df.empty:
                continue
            start, end = df["datetime"].min(), df["datetime"].max()
            min_t = start if min_t is None or start < min_t else min_t
            max_t = end if max_t is None or end > max_t else max_t
        except Exception:
            continue

    if min_t is None or max_t is None:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(min_t, max_t, freq=TIME_STEP, tz="UTC")

# =========================
# 🪄 YIL BAZLI MERGE
# =========================
def process_year(year):
    print(f"\n📅 {year} yılı işleniyor...")
    all_files = sorted(
        [os.path.join(INPUT_DIR, f)
         for f in os.listdir(INPUT_DIR)
         if f.endswith("_final_enriched.parquet")],
        key=lambda x: int(Path(x).stem.split("_")[1])  # channel_1, channel_2, ... sırayla
    )

    if not all_files:
        print("⚠️ Channel dosyası bulunamadı.")
        return

    time_index = get_time_range(all_files, year)
    if time_index.empty:
        print(f"⚠️ {year} yılına ait veri yok, atlanıyor.")
        return

    df_year = pd.DataFrame(index=time_index)

    for f in tqdm(all_files, desc=f"🔄 {year} için birleştirme", ncols=100):
        df = load_channel(f)
        if df is None or df.empty:
            continue

        df = df[df["datetime"].dt.year == year]
        if df.empty:
            continue

        ch_name = [c for c in df.columns if c.startswith("channel_")][0]

        merged = pd.merge_asof(
            df_year.reset_index().rename(columns={"index": "datetime"}),
            df.sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=TOLERANCE
        ).set_index("datetime")

        # channel verisi ve anomaly flag’ini ayrı kolonlara koy
        df_year[ch_name] = merged[ch_name]
        df_year[f"anomaly_flag__{ch_name}"] = merged["anomaly_flag"]
        df_year[f"telecommand_flag__{ch_name}"] = merged["any_telecommand_flag"]

    df_year.interpolate(method="linear", inplace=True, limit_direction="both")

    out_path = os.path.join(OUTPUT_DIR, f"{year}.parquet")
    df_year.to_parquet(out_path)
    print(f"💾 {out_path} kaydedildi ({len(df_year):,} satır, {df_year.shape[1]} kolon)")

# =========================
# 🚀 MAIN
# =========================
def main():
    print("="*60)
    print("🚀 Çok-kanallı yıllık dataset oluşturucu")
    print("="*60)

    for year in YEARS:
        process_year(year)

    print("\n🎉 Tüm yıllar başarıyla kaydedildi!")
    print(f"📁 Çıktı klasörü: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()