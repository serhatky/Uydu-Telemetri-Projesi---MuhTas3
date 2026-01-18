# enrich_multichannel_full.py

import pandas as pd
import numpy as np
from pathlib import Path

class MultiChannelEnricher:
    """
    Add global TC context features + cross-channel features
    
    Input:  multichannel_by_year/{year}.parquet (228 cols, timestamp in index)
    Output: enriched_by_year/{year}.parquet (239 cols = 229 + 7 TC + 3 cross)
    """
    
    def __init__(self, data_dir, tc_timeline_file='tc_timeline_global.parquet'):
        self.data_dir = Path(data_dir)
        
        # Load TC timeline
        print("Loading TC timeline...")
        self.tc_timeline = pd.read_parquet(tc_timeline_file)
        self.tc_timeline['timestamp'] = pd.to_datetime(self.tc_timeline['timestamp'])
        print(f"  ✓ TC events loaded: {len(self.tc_timeline):,}")
        
        # Output directory
        self.output_dir = self.data_dir / 'enriched_by_year'
        self.output_dir.mkdir(exist_ok=True)
    
    def process_all_years(self, years=None):
        """Process all years"""
        if years is None:
            years = range(2002, 2014)
        
        for year in years:
            print(f"\n{'='*60}")
            print(f"Processing Year: {year}")
            print(f"{'='*60}")
            
            try:
                self.process_single_year(year)
                print(f"✓ {year} completed successfully")
            except Exception as e:
                print(f"✗ {year} failed: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def process_single_year(self, year):
        """Process single year"""
        
        # ─── LOAD ───
        input_file = self.data_dir / 'multichannel_by_year' / f'{year}.parquet'
        df = pd.read_parquet(input_file)
        
        print(f"  Loaded: {df.shape}")
        print(f"  Index type: {type(df.index).__name__}")
        
        # ─── RESET INDEX (Timestamp'i column yap) ───
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'timestamp'})  # İlk kolon timestamp
        
        print(f"  ✓ Timestamp column created")
        print(f"  ✓ Shape: {df.shape}")
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # ─── ADD GLOBAL TC FEATURES ───
        print(f"  Adding global TC features...")
        df = self._add_global_tc_features(df, year)
        print(f"  ✓ Shape after TC features: {df.shape}")
        
        # ─── ADD CROSS-CHANNEL FEATURES ───
        print(f"  Adding cross-channel features...")
        df = self._add_cross_channel_features(df)
        print(f"  ✓ Shape after cross-channel: {df.shape}")
        
        # ─── SAVE ───
        output_file = self.output_dir / f'{year}.parquet'
        df.to_parquet(output_file, index=False, compression='snappy')
        
        print(f"  ✓ Saved to: {output_file.name}")
        print(f"  ✓ Final shape: {df.shape}")
        print(f"  ✓ Total columns: {len(df.columns)}")
        
        return df
    
    def _add_global_tc_features(self, df, year):
        """
        Add GLOBAL TC context features
        
        Features (7):
        1. minutes_since_last_tc       - Global: en son TC'den beri süre
        2. tc_density_5min             - Global: son 5dk TC yoğunluğu
        3. tc_density_15min            - Global: son 15dk TC yoğunluğu
        4. tc_density_30min            - Global: son 30dk TC yoğunluğu
        5. last_tc_id                  - Global: en son TC ID
        6. tc_count_last_hour          - Global: son 1 saatte TC sayısı
        7. is_tc_active_window         - Global: son 5dk içinde TC var mı
        """
        # Filter TCs for this year
        tc_year = self.tc_timeline[
            self.tc_timeline['timestamp'].dt.year == year
        ].copy()
        
        print(f"    TC events in {year}: {len(tc_year):,}")
        
        if len(tc_year) == 0:
            print(f"    ⚠ No TCs, using defaults")
            df['minutes_since_last_tc'] = 9999
            df['tc_density_5min'] = 0
            df['tc_density_15min'] = 0
            df['tc_density_30min'] = 0
            df['last_tc_id'] = -1
            df['tc_count_last_hour'] = 0
            df['is_tc_active_window'] = 0
            return df
        
        # ─── TIMEZONE FIX ───
        # Telemetry timestamp'i UTC'ye çevir veya naive yap
        if df['timestamp'].dt.tz is not None:
            # Telemetry UTC, TC'yi de UTC yap
            tc_year['timestamp'] = tc_year['timestamp'].dt.tz_localize('UTC')
            print(f"    ✓ Localized TC timestamps to UTC")
        else:
            # Telemetry naive, TC'yi de naive bırak
            pass
        
        # ─── Feature 1: Minutes since last TC ───
        tc_timestamps = tc_year[['timestamp']].drop_duplicates().sort_values('timestamp')
        
        df_merged = pd.merge_asof(
            df[['timestamp']],
            tc_timestamps.rename(columns={'timestamp': 'last_tc_time'}),
            left_on='timestamp',
            right_on='last_tc_time',
            direction='backward'
        )
        
        df['minutes_since_last_tc'] = (
            (df['timestamp'] - df_merged['last_tc_time']).dt.total_seconds() / 60
        ).fillna(9999).clip(upper=9999)
        
        # ─── Features 2-4, 6: TC Density ───
        # 1-minute bins
        tc_per_minute = tc_year.set_index('timestamp').resample('1min').size()
        telemetry_idx = df.set_index('timestamp').index
        
        # 5-min
        df['tc_density_5min'] = (
            tc_per_minute.rolling(window=5, min_periods=1).sum()
            .reindex(telemetry_idx, method='ffill')
            .fillna(0)
            .values
        )
        
        # 15-min
        df['tc_density_15min'] = (
            tc_per_minute.rolling(window=15, min_periods=1).sum()
            .reindex(telemetry_idx, method='ffill')
            .fillna(0)
            .values
        )
        
        # 30-min
        df['tc_density_30min'] = (
            tc_per_minute.rolling(window=30, min_periods=1).sum()
            .reindex(telemetry_idx, method='ffill')
            .fillna(0)
            .values
        )
        
        # 60-min
        df['tc_count_last_hour'] = (
            tc_per_minute.rolling(window=60, min_periods=1).sum()
            .reindex(telemetry_idx, method='ffill')
            .fillna(0)
            .values
        )
        
        # ─── Feature 5: Last TC ID ───
        df_with_tc_id = pd.merge_asof(
            df[['timestamp']],
            tc_year[['timestamp', 'tc_id']].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        df['last_tc_id'] = df_with_tc_id['tc_id'].fillna(-1).astype(int)
        
        # ─── Feature 7: Is TC active window ───
        df['is_tc_active_window'] = (df['minutes_since_last_tc'] <= 5).astype(int)
        
        return df
    
    def _add_cross_channel_features(self, df):
        """
        Add cross-channel aggregate features
        
        Features (3):
        1. power_group_mean
        2. thermal_group_std  
        3. attitude_group_delta
        """
        # Power group
        power_channels = ['channel_5', 'channel_12', 'channel_23']
        if all(ch in df.columns for ch in power_channels):
            df['power_group_mean'] = df[power_channels].mean(axis=1)
            print(f"    ✓ power_group_mean")
        else:
            df['power_group_mean'] = 0
            print(f"    ⚠ Power channels missing")
        
        # Thermal group
        thermal_channels = ['channel_8', 'channel_15', 'channel_31']
        if all(ch in df.columns for ch in thermal_channels):
            df['thermal_group_std'] = df[thermal_channels].std(axis=1)
            print(f"    ✓ thermal_group_std")
        else:
            df['thermal_group_std'] = 0
            print(f"    ⚠ Thermal channels missing")
        
        # Attitude group
        attitude_channels = ['channel_20', 'channel_21', 'channel_22']
        if all(ch in df.columns for ch in attitude_channels):
            attitude_mean = df[attitude_channels].mean(axis=1)
            df['attitude_group_delta'] = attitude_mean.diff().fillna(0)
            print(f"    ✓ attitude_group_delta")
        else:
            df['attitude_group_delta'] = 0
            print(f"    ⚠ Attitude channels missing")
        
        return df


# ═══════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    enricher = MultiChannelEnricher(
        data_dir='.',
        tc_timeline_file='tc_timeline_global.parquet'
    )
    
    # Test 2010
    print("\n" + "="*60)
    print("TESTING WITH YEAR 2010")
    print("="*60)
    enricher.process_single_year(2010)
    
    # Process all years
    print("\n" + "="*60)
    print("PROCESSING ALL YEARS")
    print("="*60)
    response = input("Process all years 2002-2013? (y/n): ")
    if response.lower() == 'y':
        enricher.process_all_years(years=range(2002, 2014))