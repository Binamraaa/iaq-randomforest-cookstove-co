import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, train_test_split # To split by household
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib # For saving the model
import sys
import datetime

# --- Configuration ---
ML_DATA_FILE = 'ML_Ready_Data.csv' # ML data file
TEST_SET_SIZE = 0.25           # Proportion of households to hold out for testing (e.g., 0.25 = 25%)
N_ESTIMATORS = 150             # Number of trees in the forest (more trees -> potentially better but slower)
MAX_DEPTH = 15                 # Max depth of trees (None=unlimited, controls complexity, try e.g., 10, 15, 20)
MIN_SAMPLES_LEAF = 5           # Minimum samples required at a leaf node (controls complexity, try 3, 5, 10)
N_JOBS = -1                    # Use all available CPU cores for training (-1)
RANDOM_STATE = 42              # For reproducible results
N_HOUSEHOLDS_TO_PLOT = 4       # How many test households to visualize
SAVE_MODEL_FILENAME = 'random_forest_co_model.joblib' # Optional filename to save the trained model
# --- End Configuration ---

print("--- Starting Random Forest Training & Evaluation ---")

# --- Load Data ---
try:
    print(f"Loading ML-ready data from: {ML_DATA_FILE}")
    ml_data = pd.read_csv(ML_DATA_FILE)
    # Ensure Timestamp is datetime object if needed for analysis later (though not a direct feature here)
    if 'Timestamp' in ml_data.columns:
        ml_data['Timestamp'] = pd.to_datetime(ml_data['Timestamp'])
    print(f"  Successfully loaded {len(ml_data)} rows and {len(ml_data.columns)} columns.")
    print(f"  Columns: {ml_data.columns.tolist()}")
except FileNotFoundError:
    print(f"FATAL ERROR: File not found: {ML_DATA_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR loading data: {e}")
    sys.exit(1)

# --- Define Features (X) and Target (y) ---
print("Defining features and target...")
try:
    target_column = 'CO_ppm'
    # Features are all columns except the target and potentially HouseholdID/Timestamp
    # Explicitly list features based on the previous preparation script's output
    feature_columns = [col for col in ml_data.columns if col not in [target_column, 'Timestamp', 'HouseholdID']]

    # Verify all expected feature columns exist
    if not all(col in ml_data.columns for col in feature_columns):
         missing = set(feature_columns) - set(ml_data.columns)
         print(f"WARNING: Some expected feature columns missing: {missing}. Excluding them.")
         feature_columns = [col for col in feature_columns if col in ml_data.columns]

    if not feature_columns:
        print("FATAL ERROR: No feature columns identified. Check column names in CSV.")
        sys.exit(1)

    X = ml_data[feature_columns]
    y = ml_data[target_column]
    groups = ml_data['HouseholdID'] # Needed for group-based splitting

    print(f"  Target: '{target_column}'")
    print(f"  Features ({len(feature_columns)}): {feature_columns}")

except KeyError as e:
    print(f"FATAL ERROR: Column mismatch. Missing column: {e}")
    print(f"Make sure '{ML_DATA_FILE}' contains the target and all feature columns.")
    sys.exit(1)
except Exception as e:
     print(f"FATAL ERROR preparing features/target: {e}")
     sys.exit(1)


# --- Split Data: Train/Test Holding Out Households ---
print(f"Splitting data into train/test sets ({1-TEST_SET_SIZE:.0%}/{TEST_SET_SIZE:.0%}), keeping households intact...")
# Using GroupShuffleSplit to ensure all data from a household is in EITHER train or test
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx] # Keep track of households in each set
timestamps_test = ml_data['Timestamp'].iloc[test_idx] # For plotting

train_households = groups_train.unique()
test_households = groups_test.unique()

print(f"  Training set: {len(X_train)} rows from {len(train_households)} households.")
print(f"  Test set:     {len(X_test)} rows from {len(test_households)} households.")
print(f"  Test Households: {np.sort(test_households)}")

if len(X_test) == 0:
    print("FATAL ERROR: Test set is empty. Maybe TEST_SET_SIZE is too large or too few households?")
    sys.exit(1)

# --- Train Random Forest Model ---
print(f"\nTraining RandomForestRegressor model...")
print(f"  Parameters: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}, n_jobs={N_JOBS}")

# Initialize the model
rf_model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    oob_score=False, # Can set to True for out-of-bag estimate on training data
    max_features=1.0 # Consider all features at each split (default in newer sklearn)
)

# Train the model
try:
    start_time = datetime.datetime.now()
    rf_model.fit(X_train, y_train)
    end_time = datetime.datetime.now()
    print(f"  Training complete. Time taken: {end_time - start_time}")
except Exception as e:
    print(f"FATAL ERROR during model training: {e}")
    sys.exit(1)

# --- Make Predictions on Test Set ---
print("Making predictions on the test set...")
try:
    y_pred = rf_model.predict(X_test)
except Exception as e:
    print(f"FATAL ERROR during prediction: {e}")
    sys.exit(1)

# --- Evaluate the Model ---
print("\nEvaluating model performance on the test set...")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Test Set Evaluation Metrics ---")
print(f"RMSE: {rmse:.2f} ppm")
print(f"MAE:  {mae:.2f} ppm")
print(f"R²:   {r2:.3f}")
print(f"-----------------------------------")
print(f"(Compare RMSE/MAE to the range of CO_ppm values in your data)")


# --- Visualize Results ---

# 1. Feature Importances
print("\nCalculating and plotting feature importances...")
try:
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, max(6, len(feature_columns) * 0.3))) # Adjust height based on num features
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importances')
    plt.gca().invert_yaxis() # Display most important at the top
    plt.tight_layout()
    plt.show()
    print("  Top 5 Features:")
    print(feature_importance_df.head())
except Exception as e:
    print(f"  Warning: Could not generate feature importance plot: {e}")


# 2. Observed vs. Predicted Time Series (for Sample Households)
print(f"\nPlotting observed vs. predicted for {N_HOUSEHOLDS_TO_PLOT} sample test households...")
if len(test_households) > 0:
    households_to_plot = np.random.choice(test_households, size=min(N_HOUSEHOLDS_TO_PLOT, len(test_households)), replace=False)

    plt.figure(figsize=(15, N_HOUSEHOLDS_TO_PLOT * 5))
    plt.suptitle('Observed vs. Predicted CO (Random Forest) - Test Set Samples', fontsize=16, y=1.02)

    plot_count = 0
    for i, household_id in enumerate(households_to_plot):
        plot_count += 1
        ax = plt.subplot(N_HOUSEHOLDS_TO_PLOT, 1, plot_count)

        # Find indices corresponding to this household in the *original test set*
        idx_in_test = groups_test[groups_test == household_id].index

        # Use these indices to get the correct timestamps, observed, and predicted values
        household_timestamps = ml_data.loc[idx_in_test, 'Timestamp']
        household_y_test = y_test.loc[idx_in_test]
        # Need to align y_pred with the original index of X_test before filtering
        household_y_pred = pd.Series(y_pred, index=X_test.index).loc[idx_in_test]

        if household_y_test.empty: continue # Skip if somehow empty

        ax.plot(household_timestamps, household_y_test, label=f'Observed CO (Peak: {household_y_test.max():.1f})', color='blue', marker='.', linestyle='-', alpha=0.7, markersize=4)
        ax.plot(household_timestamps, household_y_pred, label=f'Predicted CO (Peak: {household_y_pred.max():.1f})', color='red', marker=None, linestyle='--', alpha=0.8, linewidth=1.5)

        stove_type_plot = X_test.loc[idx_in_test, 'StoveType_TCS'].iloc[0] == 1 if 'StoveType_TCS' in X_test.columns else X_test.loc[idx_in_test, 'StoveType_ICS'].iloc[0] == 0 if 'StoveType_ICS' in X_test.columns else 'Unknown'
        stove_label = 'TCS' if stove_type_plot else 'ICS' if stove_type_plot != 'Unknown' else 'Unknown'

        ax.set_title(f'Test Household: {household_id} ({stove_label})')
        ax.set_ylabel('CO Concentration (ppm)')
        ax.set_xlabel('Time')
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=20, ha='right')


    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
    plt.show()
else:
    print("  No households available in the test set to plot.")


# --- Optional: Save the Trained Model ---
if SAVE_MODEL_FILENAME:
    print(f"\nSaving trained model to: {SAVE_MODEL_FILENAME}")
    try:
        joblib.dump(rf_model, SAVE_MODEL_FILENAME)
        print("  Model saved successfully.")
    except Exception as e:
        print(f"  Error saving model: {e}")

print("\n--- Script Finished ---")