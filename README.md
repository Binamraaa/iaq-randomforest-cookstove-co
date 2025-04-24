# iaq-randomforest-cookstove-co
Machine learning model (Random Forest) to predict indoor CO concentrations from cookstoves in rural Nepal. Part of MSc thesis (M.Sc. in Climate Change and Development).

# Predicting Indoor Air Pollution (CO) from Cookstoves using Random Forest

## Project Overview

This repository contains the code and model developed for predicting minute-by-minute Carbon Monoxide (CO) concentrations originating from traditional (TCS) and improved (ICS) biomass cookstoves in naturally ventilated rural kitchens in [Kavre District, Nepal]. This work addresses the significant public health challenge posed by Indoor Air Pollution (IAP) and explores the use of machine learning as a predictive tool where simpler mechanistic models face limitations.

**(Obs vs Pred plot)**
![Observed vs Predicted CO](https://github.com/Binamraaa/iaq-randomforest-cookstove-co/blob/main/images/hh13_ICS.PNG)

## Background & Challenge

Indoor air pollution from solid fuel combustion is a major health risk factor globally. Accurately predicting pollutant concentrations like CO is crucial for exposure assessment but challenging due to complex interactions between emission sources, variable ventilation, and kitchen characteristics. Initial attempts using standard box models demonstrated limitations in capturing the observed concentration dynamics accurately.

## Solution: Machine Learning Approach

To overcome these limitations, a data-driven approach using a **Random Forest Regressor** was implemented. The model was trained on time-series data collected from ~20 household cooking sessions involving both TCS and ICS.

**Key Features Used:**
*   Lagged CO Concentrations (`CO_ppm_Lag1`, `Lag2`, `Lag5`, `Lag10`)
*   Kitchen Volume (`Volume_m3`)
*   Stove Type (ICS/TCS - One-Hot Encoded)
*   Estimated Ambient CO (`AmbientCO_ppm_Est`)
*   Time Features (`HourOfDay`, `MinuteOfHour`)
*   Cooking Status (`IsCooking`, `TimeSinceCookStart_min`)

*(See `scripts/Cleaned_CO_with_date.py` for full feature engineering details)*

## Results & Performance

The trained Random Forest model demonstrated strong predictive performance on a held-out test set comprising unseen households:
*   **R²:** 0.915
*   **RMSE:** 16.69 ppm
*   **MAE:** 5.62 ppm

## Usage

This section explains how to use the provided code to load the trained Random Forest model and make predictions on new data.

**Prerequisites:**

*   **Git:** You need Git installed to clone the repository. ([Install Git](https://git-scm.com/downloads))
*   **Python:** Python 3.x installed. ([Install Python](https://www.python.org/downloads/))
*   **Virtual Environment (Recommended):** It's highly recommended to use a virtual environment to manage project dependencies. Create one using `venv` or `conda`.
    *   Example using `venv`:
        ```bash
        # Navigate to the project root directory in your terminal
        python -m venv venv_rf_iap  # Create environment named 'venv_rf_iap'
        source venv_rf_iap/bin/activate  # On Linux/macOS
        # OR
        .\venv_rf_iap\Scripts\activate  # On Windows
        ```

**Steps:**

1.  **Clone the Repository:**
    Open your terminal or command prompt and clone this repository to your local machine:
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git
    cd YourRepositoryName
    ```
    *(Replace `YourUsername/YourRepositoryName` with the actual path)*

2.  **Install Dependencies:**
    Install the required Python libraries listed in `requirements.txt` (activate your virtual environment first if you created one):
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Prediction Example:**
    Execute the example script provided to see the model load and predict on sample data. Run this command from the **root directory** of the cloned project (`YourRepositoryName/`):
    ```bash
    python scripts/predict_example.py
    ```
    *   **Expected Output:** The script will print confirmation messages, show the sample input data it loaded from `data/sample_input.csv`, and finally output the predicted CO concentration(s) in ppm for the sample row(s).

4.  **Making Predictions on Your Own Data (Interacting with the Model):**
    The easiest way to test the model with different inputs is to modify the `data/sample_input.csv` file:
    *   **Locate:** Open the file `data/sample_input.csv` using a spreadsheet program (like Excel, Google Sheets, LibreOffice Calc) or a simple text editor.
    *   **Understand the Format:**
        *   The **first row** is the **header row**. It contains the exact feature names required by the model, in the specific order it expects. **DO NOT CHANGE THE HEADER ROW.**
        *   Each subsequent **row** represents a single time point for which you want a prediction.
        *   The values in each row correspond to the features listed in the header.
    *   **Modify Values:** Change the numerical values in the existing data rows (below the header) to represent the scenario you want to test. Ensure you provide plausible values for each feature (e.g., `IsCooking` should be 0 or 1, `Volume_m3` should be positive, lagged CO values should be reasonable).
    *   **Add New Scenarios:** You can add new rows to the CSV file, ensuring each row has the correct number of values corresponding to the header columns.
    *   **Save:** Save your changes to `sample_input.csv` **as a CSV file**.
    *   **Re-run:** Execute the example script again from the project root directory:
        ```bash
        python scripts/predict_example.py
        ```
        The script will now load your modified data from the CSV and output predictions for each row you provided.

**Important Notes:**

*   **Feature Order:** The order of columns in `data/sample_input.csv` **must** exactly match the `EXPECTED_FEATURE_ORDER` list defined within the `scripts/predict_example.py` script. This order corresponds to the features used during model training.
*   **Model File:** The prediction script loads the trained model from `models/random_forest_co_model.joblib` using the `joblib` library.
*   **Large Datasets:** This example script is designed for making predictions on a small number of samples provided in the CSV. For predicting on large datasets, you would adapt the data loading part of the script accordingly.

