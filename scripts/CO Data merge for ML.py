import pandas as pd
import numpy as np
import datetime
import sys # To exit gracefully on error

# --- Configuration ---
INPUT_CO_DATA_FILE = 'Cleaned_CO_with_Date.csv' # file with the added Date column
HOUSEHOLD_INFO_FILE = 'household_info.csv'    # file with Volume, StoveType, etc.
OUTPUT_ML_FILE = 'ML_Ready_Data.csv'          # File to be created

LAG_FEATURES_MINUTES = [1, 2, 5, 10]          # Which past CO values to use as features
# --------------------

print("--- Starting Data Preparation for ML ---")

# --- Load Input Data ---
try:
    print(f"Loading CO data from: {INPUT_CO_DATA_FILE}")
    data = pd.read_csv(INPUT_CO_DATA_FILE)
    # Rename columns immediately for clarity
    data.rename(columns={'Date/Time': 'Time', 'CO(ppm)': 'CO_ppm'}, inplace=True)
    print(f"  Successfully loaded {len(data)} rows.")
except FileNotFoundError:
    print(f"FATAL ERROR: File not found: {INPUT_CO_DATA_FILE}")
    print("Please ensure you have created this file with an added 'Date' column.")
    sys.exit(1)
except KeyError as e:
    print(f"FATAL ERROR: Missing expected column in {INPUT_CO_DATA_FILE}: {e}")
    print("Expected columns: Stage, Date, Time, CO_ppm")
    sys.exit(1)


try:
    print(f"Loading household info from: {HOUSEHOLD_INFO_FILE}")
    household_info = pd.read_csv(HOUSEHOLD_INFO_FILE)
    print(f"  Successfully loaded info for {len(household_info)} households.")
except FileNotFoundError:
    print(f"FATAL ERROR: File not found: {HOUSEHOLD_INFO_FILE}")
    print("Please create 'household_info.csv' with columns: HouseholdID, Volume_m3, StoveType, AmbientCO_ppm_Measured")
    sys.exit(1)
except KeyError as e:
    print(f"FATAL ERROR: Missing expected column in {HOUSEHOLD_INFO_FILE}: {e}")
    print("Expected columns: HouseholdID, Volume_m3, StoveType, AmbientCO_ppm_Measured")
    sys.exit(1)

# --- Initial Processing and Cleaning ---
print("Processing and cleaning data...")
try:
    # 1. Parse Stage
    data['Stage_str'] = data['Stage'].astype(str)
    data['HouseholdID'] = data['Stage_str'].apply(lambda x: int(x.split('.')[0]))
    data['Phase'] = data['Stage_str'].apply(lambda x: int(x.split('.')[1]))

    # 2. Create Timestamp (Crucial - Relies on 'Date' column existing)
    data['ParsedDate'] = pd.to_datetime(data['Date']).dt.date
    data['ParsedTime'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
    data['Timestamp'] = data.apply(lambda row: datetime.datetime.combine(row['ParsedDate'], row['ParsedTime']), axis=1)

    # 3. Sort Data (Essential for lags and time calculations)
    data = data.sort_values(by=['HouseholdID', 'Timestamp']).reset_index(drop=True)

    # 4. Clean CO_ppm
    data['CO_ppm'] = pd.to_numeric(data['CO_ppm'], errors='coerce') # Handle non-numeric values
    data.dropna(subset=['CO_ppm'], inplace=True) # Drop rows where CO is non-numeric
    data.loc[data['CO_ppm'] < 0, 'CO_ppm'] = 0 # Floor negative CO at 0

except Exception as e:
    print(f"FATAL ERROR during initial processing: {e}")
    print("Check Stage format, Date format, Time format, or CO_ppm values in input file.")
    sys.exit(1)

# --- Merge Household Info ---
print("Merging household info...")
try:
    data = pd.merge(data, household_info, on='HouseholdID', how='left')
    if data['Volume_m3'].isnull().any():
        missing_hhs = data[data['Volume_m3'].isnull()]['HouseholdID'].unique()
        print(f"  WARNING: Volume_m3 or StoveType missing for households: {missing_hhs}. Rows for these households might be dropped later.")
        # Don't exit, maybe user wants to proceed with remaining data
except Exception as e:
     print(f"FATAL ERROR during merge with {HOUSEHOLD_INFO_FILE}: {e}")
     print("Check HouseholdID consistency between files.")
     sys.exit(1)

# --- Determine Ambient CO Estimate ---
print("Determining Ambient CO estimate...")
data['AmbientCO_ppm_Est'] = np.nan # Initialize

# Try using measured value first
if 'AmbientCO_ppm_Measured' in data.columns:
     data['AmbientCO_ppm_Measured'] = pd.to_numeric(data['AmbientCO_ppm_Measured'], errors='coerce')
     data['AmbientCO_ppm_Est'] = data['AmbientCO_ppm_Est'].fillna(data['AmbientCO_ppm_Measured'])

# Estimate remaining from Phase 1 minimum
if data['AmbientCO_ppm_Est'].isnull().any():
    print("  Estimating missing ambient values from Phase 1 minimum...")
    # Calculate min CO in phase 1 for *each* household session (identified by HouseholdID and Date)
    phase1_min_co = data[data['Phase'] == 1].groupby(['HouseholdID', 'ParsedDate'])['CO_ppm'].min().reset_index()
    phase1_min_co.rename(columns={'CO_ppm': 'Phase1_Min_CO'}, inplace=True)

    # Merge these minimums back
    data = pd.merge(data, phase1_min_co, on=['HouseholdID', 'ParsedDate'], how='left')
    data['AmbientCO_ppm_Est'] = data['AmbientCO_ppm_Est'].fillna(data['Phase1_Min_CO'])
    data.drop(columns=['Phase1_Min_CO'], inplace=True)

    # Final fallback: If still missing (e.g., no Phase 1 data), use global *observed* min CO
    if data['AmbientCO_ppm_Est'].isnull().any():
        global_min_co = data['CO_ppm'].min()
        print(f"  WARNING: Some sessions still lack ambient CO. Using global minimum CO ({global_min_co:.2f} ppm) as fallback.")
        data['AmbientCO_ppm_Est'].fillna(global_min_co, inplace=True)

# Ensure ambient is not negative
data.loc[data['AmbientCO_ppm_Est'] < 0, 'AmbientCO_ppm_Est'] = 0


# --- Feature Engineering ---
print("Engineering features...")
# 1. Time Features
data['HourOfDay'] = data['Timestamp'].dt.hour
data['MinuteOfHour'] = data['Timestamp'].dt.minute

# 2. Event Features
data['IsCooking'] = (data['Phase'] == 2).astype(int)

# Find Cook Start time PER HOUSEHOLD SESSION (ID + Date)
cook_start_times = data[data['IsCooking'] == 1].groupby(['HouseholdID', 'ParsedDate'])['Timestamp'].min().reset_index()
cook_start_times.rename(columns={'Timestamp': 'CookStartTime'}, inplace=True)
data = pd.merge(data, cook_start_times, on=['HouseholdID', 'ParsedDate'], how='left')

# Calculate TimeSinceCookStart_min (handle NaT if no cooking phase)
data['TimeSinceCookStart_min'] = np.nan
valid_start_mask = data['Timestamp'] >= data['CookStartTime']
data.loc[valid_start_mask, 'TimeSinceCookStart_min'] = (data.loc[valid_start_mask, 'Timestamp'] - data.loc[valid_start_mask, 'CookStartTime']).dt.total_seconds() / 60.0
data['TimeSinceCookStart_min'].fillna(0, inplace=True) # Assume 0 if before cooking or cooking didn't happen

# 3. Lag Features
print(f"  Creating lag features for minutes: {LAG_FEATURES_MINUTES}")
for lag in LAG_FEATURES_MINUTES:
    data[f'CO_ppm_Lag{lag}'] = data.groupby(['HouseholdID', 'ParsedDate'])['CO_ppm'].shift(lag)

# 4. One-Hot Encode StoveType
print("  One-hot encoding StoveType...")
try:
    data = pd.get_dummies(data, columns=['StoveType'], prefix='StoveType', dummy_na=False, dtype=int)
except KeyError:
    print("  WARNING: 'StoveType' column not found after merge. Cannot create StoveType features.")


# --- Final Cleanup ---
print("Final cleanup...")
# 1. Select final columns (adjust if StoveType encoding failed)
feature_columns = [
    'HouseholdID', 'Timestamp', 'Volume_m3',
    'AmbientCO_ppm_Est', 'HourOfDay', 'MinuteOfHour',
    'IsCooking', 'TimeSinceCookStart_min'
]
# Add StoveType columns if they exist
if 'StoveType_ICS' in data.columns: feature_columns.append('StoveType_ICS')
if 'StoveType_TCS' in data.columns: feature_columns.append('StoveType_TCS')
# Add Lag columns
for lag in LAG_FEATURES_MINUTES:
    feature_columns.append(f'CO_ppm_Lag{lag}')

target_column = 'CO_ppm'

# Ensure all selected feature columns exist before selection
existing_feature_columns = [col for col in feature_columns if col in data.columns]
if len(existing_feature_columns) != len(feature_columns):
     missing_cols = set(feature_columns) - set(existing_feature_columns)
     print(f"  WARNING: Some feature columns are missing and will be excluded: {missing_cols}")

final_columns_list = existing_feature_columns + [target_column]
ml_data = data[final_columns_list].copy()


# 2. Drop rows with NaN in crucial columns (especially lags and volume)
initial_rows = len(ml_data)
cols_to_check_for_nan = existing_feature_columns # Check all features used
ml_data.dropna(subset=cols_to_check_for_nan, inplace=True)
rows_dropped = initial_rows - len(ml_data)
print(f"  Dropped {rows_dropped} rows due to missing values (mainly from initial lags).")

# --- Save Output ---
try:
    print(f"Saving ML-ready data to: {OUTPUT_ML_FILE}")
    ml_data.to_csv(OUTPUT_ML_FILE, index=False)
    print(f"  Successfully saved {len(ml_data)} rows.")
    print("\n--- Data Preparation Complete ---")
    print("Sample of the final data:")
    print(ml_data.head())
    print("\nColumns in final data:")
    print(ml_data.columns.tolist())
except Exception as e:
    print(f"FATAL ERROR saving output file: {e}")
    sys.exit(1)
