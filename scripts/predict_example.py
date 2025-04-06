# scripts/predict_example.py 
import joblib
import pandas as pd
import os
import sys

print("--- CO Prediction Example (Reading from sample_input.csv) ---")

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # Get project root directory
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'random_forest_co_model.joblib')
SAMPLE_INPUT_FILE = os.path.join(BASE_DIR, 'data', 'sample_input.csv')

# --- IMPORTANT: Define the EXACT feature order expected by the model ---
# This should still match the order used during training!
EXPECTED_FEATURE_ORDER = [
    'Volume_m3', 'AmbientCO_ppm_Est', 'HourOfDay', 'MinuteOfHour', 'IsCooking',
    'TimeSinceCookStart_min', 'StoveType_ICS', 'StoveType_TCS', 'CO_ppm_Lag1',
    'CO_ppm_Lag2', 'CO_ppm_Lag5', 'CO_ppm_Lag10'
]
# --------------------------------------------------------------------

# --- Load Model ---
try:
    print(f"Loading model from: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# --- Load Sample Input Data from CSV ---
try:
    print(f"Loading sample input data from: {SAMPLE_INPUT_FILE}")
    sample_input_df = pd.read_csv(SAMPLE_INPUT_FILE)
    print(f"Loaded {len(sample_input_df)} sample row(s).")
    # Ensure all expected columns are present
    missing_cols = set(EXPECTED_FEATURE_ORDER) - set(sample_input_df.columns)
    if missing_cols:
        print(f"ERROR: Sample input CSV is missing columns: {missing_cols}")
        sys.exit(1)
    # Reorder columns from CSV to match the training order
    sample_input_df = sample_input_df[EXPECTED_FEATURE_ORDER]
    print("Sample Input DataFrame (from CSV):")
    print(sample_input_df)
except FileNotFoundError:
     print(f"ERROR: Sample input file not found at {SAMPLE_INPUT_FILE}")
     sys.exit(1)
except Exception as e:
    print(f"ERROR reading or processing sample input CSV: {e}")
    sys.exit(1)

# --- Make Predictions ---
try:
    print("\nMaking prediction(s)...")
    predicted_co_ppm = model.predict(sample_input_df)

    # Print predictions for each row in the sample input
    print("\n--- Predictions ---")
    for i, prediction in enumerate(predicted_co_ppm):
        print(f"Input Row {i+1}: Predicted CO = {prediction:.2f} ppm")

except Exception as e:
    print(f"ERROR during prediction: {e}")

print("\n--- Example Finished ---")
print(f"Hint: You can modify '{SAMPLE_INPUT_FILE}' to test different scenarios.")
