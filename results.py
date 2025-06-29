# Visualization and Results Script

print("Final dataset shape:", scaled_df.shape)

plt.figure(figsize=(12, 4))
plt.plot(scaled_df['datetime'], scaled_df['Avg_smlvl_at15cm'], label='Avg_smlvl_at15cm')
plt.title("Soil Moisture Over Time")
plt.xlabel("Date")
plt.ylabel("Avg_smlvl_at15cm")
plt.legend()
plt.tight_layout()
plt.show()


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Convergence')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# Sample prediction
test_sample = X_test[0:1]  # First test sample
prediction = model.predict(test_sample)[0][0]

# Create dummy array with ALL original features (fill others with 0)
dummy_array = np.zeros((1, 4))  # 4 features total
dummy_array[0, 2] = prediction  # Assuming soil_moisture is 3rd column (0-indexed)

# Convert back to original scale
pred_original = scaler.inverse_transform(dummy_array)[0, 2]  # Get 3rd column

print(f"Predicted moisture: {pred_original:.2f} (Original scale)")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Extract ONLY soil moisture from y_test (assuming last column)
y_test_moisture = y_test[:, -1]  # Use correct column index if different

# 2. Get predictions and flatten
y_pred = model.predict(X_test).flatten()

# 3. Calculate metrics
mae = mean_absolute_error(y_test_moisture, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_moisture, y_pred))
print(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}")

# 4. Visual comparison
plt.figure(figsize=(10,4))
plt.plot(y_test_moisture[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', alpha=0.7, marker='x')
plt.title("Soil Moisture: Actual vs Predicted")
plt.ylabel("Scaled Moisture")
plt.legend()
plt.show()

# Shape verification
print(f"\nShapes - y_test_moisture: {y_test_moisture.shape}, y_pred: {y_pred.shape}")

def predict_moisture(temp, humidity, rainfall):
    # Scale inputs (add dummy moisture)
    scaled = scaler.transform([[temp, humidity, rainfall, 0]])[0][:3]  
    prediction = model.predict(np.array([scaled]))[0][0]
    return scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]