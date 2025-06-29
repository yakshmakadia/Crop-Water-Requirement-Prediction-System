# LSTM Model Script

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Enhanced LSTM Model Architecture
model = Sequential([
    Input(shape=(n_steps, 3), name='input_layer'),
    LSTM(128, 
         activation='tanh',
         return_sequences=True,
         kernel_regularizer=l2(0.01),
         recurrent_dropout=0.2,
         name='lstm_1'),
    Dropout(0.3, name='dropout_1'),
    LSTM(64,
         activation='tanh',
         return_sequences=False,
         kernel_regularizer=l2(0.005),
         name='lstm_2'),
    Dropout(0.2, name='dropout_2'),
    Dense(1, activation='linear', name='output')
])

# Optimizer with learning rate scheduling
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.95
    ),
    clipvalue=0.5
)

# Model compilation
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae', 'mse']
)

# Model summary
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Initialize optimizer with reasonable defaults
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,  # Good starting point for LSTMs
    clipvalue=0.5        # Gradient clipping
)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

# Training configuration
history = model.fit(
    X_train, y_train,
    epochs=50,  # Reduced from 200 based on your observation
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[
        EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True),  # More aggressive early stopping
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1),  # Faster LR reduction
        ModelCheckpoint('best_model.keras', save_best_only=True),
        TensorBoard(log_dir='./logs')
    ],
    verbose=1
)

def predict_moisture(temp, humidity, rainfall):
    """Proper prediction with feature names handling"""
    # Create DataFrame with original feature names
    input_df = pd.DataFrame([[temp, humidity, rainfall, 0]], 
                          columns=['air_temp', 'air_humidity', 'rainfall', 'soil_moisture'])
    
    # Scale
    scaled = scaler.transform(input_df)[0, :3]  # Get first 3 scaled features
    
    # Reshape for LSTM
    prediction_input = scaled.reshape(1, 1, 3)
    
    # Predict and inverse scale
    prediction = model.predict(prediction_input)[0][0]
    dummy_output = np.array([[0, 0, 0, prediction]])
    return scaler.inverse_transform(dummy_output)[0][3]