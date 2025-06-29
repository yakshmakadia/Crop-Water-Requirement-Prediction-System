# Data Preprocessing Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
print("Done")

import pandas as pd

# Load datasets with encoding handling
climate_df = pd.read_csv('/kaggle/input/crop-water-requirement-prediction-using-ml/Climate_Data.csv')
crop_df = pd.read_csv('/kaggle/input/crop-water-requirement-prediction-using-ml/Crop_Calender.csv', encoding='ISO-8859-1')
soil_class_df = pd.read_csv('/kaggle/input/crop-water-requirement-prediction-using-ml/Soil_Classification.csv')
soil_moisture_df = pd.read_csv('/kaggle/input/crop-water-requirement-prediction-using-ml/Soil_Moisture.csv')

# Quick check of the loaded data
print("✅ Climate Data shape:", climate_df.shape)
print("✅ Crop Calendar shape:", crop_df.shape)
print("✅ Soil Classification shape:", soil_class_df.shape)
print("✅ Soil Moisture shape:", soil_moisture_df.shape)
print("Done")


# 1. Filter climate data to only August 2018 onwards
climate_aug2018_onwards = climate_processed_df[
    climate_processed_df['datetime'] >= '2018-08-01'
].copy()

# 2. Merge with soil moisture data
merged_df = pd.merge(
    climate_aug2018_onwards,
    final_df[['dt', 'Avg_smlvl_at15cm']],  # <- Corrected here
    left_on='datetime',
    right_on='dt',
    how='left'
).drop(columns=['dt'])

# 3. Verify no missing values in critical columns
print("\nData Quality Check:")
print(f"Rows with missing soil moisture: {merged_df['Avg_smlvl_at15cm'].isna().sum()}")
print(f"Date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")

# 4. Save with your requested filename
output_filename = 'climate_soil_moisture_merge.csv'
merged_df.to_csv(output_filename, index=False)

# 5. Show final confirmation
print("\nFirst 5 Rows of Merged Data:")
print(merged_df.head().to_string(index=False))
print(f"\n✅ Successfully saved to '{output_filename}'")

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Apply scaling to the numerical columns (using current column names)
scaled_data = scaler.fit_transform(merged_df[['air_temp', 'air_humidity', 'rainfall', 'Avg_smlvl_at15cm']])

# Create a new dataframe with the scaled data
scaled_df = pd.DataFrame(scaled_data, columns=['air_temp_scaled', 'air_humidity_scaled', 
                                              'rainfall_scaled', 'soil_moisture_scaled'])

# Keep the original columns and datetime
scaled_df = pd.concat([
    merged_df[['datetime', 'air_temp', 'air_humidity', 'rainfall', 'Avg_smlvl_at15cm']],
    scaled_df
], axis=1)

# Check the scaled dataframe
print("\nFirst 5 Rows of Scaled Data:")
print(scaled_df.head().to_string(index=False))

# Save the scaled data
scaled_df.to_csv('climate_soil_moisture_merge_scaled.csv', index=False)
print("\nSaved scaled data to 'climate_soil_moisture_merge_scaled.csv'")


# Before splitting, ensure no NaN in target (soil moisture)
print("Original NaN count:", merged_df['Avg_smlvl_at15cm'].isna().sum())

# Fill NaN using forward-fill + interpolation (for time series)
merged_df['Avg_smlvl_at15cm'] = merged_df['Avg_smlvl_at15cm'].interpolate().ffill().bfill()

# Verifymerged_df['Avg_smlvl_at15cm'] = merged_df['Avg_smlvl_at15cm'].interpolate(limit=3).ffill().bfill()
assert not merged_df['Avg_smlvl_at15cm'].isna().any(), "NaNs still exist!"

# Split ensuring no NaN in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    shuffle=False  # Critical for time series
)
print(f"Train dates: {X_train.shape[0]} samples\nTest dates: {X_test.shape[0]} samples")
# Final check
print("NaN in y_test:", np.isnan(y_test).any())