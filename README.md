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

*(See `scripts/prepare_data.py` for full feature engineering details)*

## Results & Performance

The trained Random Forest model demonstrated strong predictive performance on a held-out test set comprising unseen households:
*   **R²:** 0.915
*   **RMSE:** 16.69 ppm
*   **MAE:** 5.62 ppm

## Model Usage (`random_forest_co_model.joblib`)

The trained scikit-learn `RandomForestRegressor` model is saved in the `/models/` directory.

**Requirements:**
Install necessary libraries:
```bash
pip install -r requirements.txt
