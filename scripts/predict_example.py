# scripts/predict_example.py
import joblib
import pandas as pd
import os # To help with file paths

print("--- CO Prediction Example ---")

# --- Configuration ---
# Construct the path relative to the script's location
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest_co_model.joblib') # Use your actual model filename

# --- IMPORTANT: Define the EXACT feature order expected by the model ---
# Replace this list with the actual order from your training script!
EXPECTED_FEATURE_ORDER = [
    'Volume_m3', 'AmbientCO_ppm_Est', 'HourOfDay', 'MinuteOfHour', 'IsCooking',
    'TimeSinceCookStart_min', 'StoveType_ICS', 'StoveType_TCS', 'CO_ppm_Lag1',
    'CO_ppm_Lag2', 'CO_ppm_Lag5', 'CO_ppm_Lag10'
    # Add/remove/reorder columns EXACTLY as used in training X_train
]
# --------------------------------------------------------------------

# --- Load Model ---
try:
    print(f"Loading model from: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_FILE}")
    print("Ensure the model file exists in the 'models' directory.")
    exit()
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit()

# --- Prepare Sample Input Data ---
# Create a dictionary representing ONE row of input data
# Use plausible values for demonstration
print("\nPreparing sample input data...")
sample_input_dict = {
    'Volume_m3': [25.0],             # Example kitchen volume
    'AmbientCO_ppm_Est': [1.0],     # Example ambient CO
    'HourOfDay': [18],              # Example hour
    'MinuteOfHour': [5],            # Example minute
    'IsCooking': [1],               # Example: Currently cooking
    'TimeSinceCookStart_min': [30.0], # Example: 30 mins into cooking
    'StoveType_ICS': [1],           # Example: It's an ICS
    'StoveType_TCS': [0],
    'CO_ppm_Lag1': [15.5],          # Example: CO 1 min ago
    'CO_ppm_Lag2': [14.0],          # Example: CO 2 mins ago
    'CO_ppm_Lag5': [11.2],          # Example: CO 5 mins ago
    'CO_ppm_Lag10': [8.1]           # Example: CO 10 mins ago
    # --- Make sure ALL required features are included ---
}
try:
  sample_input_df = pd.DataFrame(sample_input_dict)
  # Reorder columns to match the training order
  sample_input_df = sample_input_df[EXPECTED_FEATURE_ORDER]
  print("Sample Input DataFrame:")
  print(sample_input_df)
except KeyError as e:
    print(f"ERROR: Feature mismatch in sample input. Missing/misnamed: {e}")
    print(f"       Ensure sample_input_dict keys match EXPECTED_FEATURE_ORDER.")
    exit()
except Exception as e:
    print(f"ERROR preparing sample input DataFrame: {e}")
    exit()


# --- Make Prediction ---
try:
    print("\nMaking prediction...")
    predicted_co_ppm = model.predict(sample_input_df)
    print(f"===> Predicted CO: {predicted_co_ppm[0]:.2f} ppm")
except Exception as e:
    print(f"ERROR during prediction: {e}")
    print("      Check if input data format matches model expectations.")

print("\n--- Example Finished ---")