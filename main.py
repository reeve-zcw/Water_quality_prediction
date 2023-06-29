import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
import numpy as np

# Read weather and water quality data (Use previous day)
weather_csv = pd.read_csv("dataset/previous_day/weather.csv").sort_values(by="Date")
water_csv = pd.read_csv("dataset/previous_day/water_quality.csv").sort_values(by="Sampling Date")

# Prepare the input features (X) and target variables (Y)
X = weather_csv.drop('Date', axis=1).values
Y = water_csv.drop('Sampling Date', axis=1).values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# increase the sample size
X_train = np.repeat(X_train, 8, axis=0)
y_train = np.repeat(y_train, 8, axis=0)

# Shuffle X_train and y_train
data = np.column_stack((X_train, y_train))
np.random.shuffle(data)
X_train = data[:, :-12]
y_train = data[:, -12:]

# Set the training parameters
hidden_layer_activation = 'relu'
output_layer = 'linear'
optimizer = 'adam'
loss_func = 'mean_squared_error'
training_epoch = 10000
training_batch_size = 140

# Build the sequential model
model = Sequential()
model.add(Dense(32, activation=hidden_layer_activation, input_dim=X_train.shape[1]))
model.add(Dense(64, activation=hidden_layer_activation))
model.add(Dense(64, activation=hidden_layer_activation))
model.add(BatchNormalization())
model.add(Dense(128, activation=hidden_layer_activation))
model.add(BatchNormalization())
model.add(Dense(128, activation=hidden_layer_activation))
model.add(Dense(256, activation=hidden_layer_activation))
model.add(Dropout(0.2))
model.add(Dense(256, activation=hidden_layer_activation))
model.add(Dropout(0.2))
model.add(Dense(256, activation=hidden_layer_activation))
model.add(Dropout(0.4))
model.add(Dense(128, activation=hidden_layer_activation))
model.add(BatchNormalization())
model.add(Dense(128, activation=hidden_layer_activation))
model.add(Dropout(0.2))
model.add(Dense(64, activation=hidden_layer_activation))
model.add(BatchNormalization())
model.add(Dense(64, activation=hidden_layer_activation))
model.add(Dropout(0.2))
model.add(Dense(64, activation=hidden_layer_activation))
model.add(BatchNormalization())
model.add(Dense(12, activation=output_layer))
model.compile(optimizer=optimizer, loss=loss_func)

# Define early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=400, verbose=1)

# TensorBoard callback for visualizing training progress and model
tf_callback = keras.callbacks.TensorBoard(log_dir="./logs")

# Train the model
history = model.fit(X_train, y_train, epochs=training_epoch, batch_size=training_batch_size,
                    verbose=1, validation_split=0.2, callbacks=[early_stopping, tf_callback])

# Test the model
y_pred = model.predict(X_test)

# evaluate the model by RMSE
mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
print("Test RMSE:", rmse)

# Save the model weights
from datetime import datetime

model.save_weights(f'model_weights/MLP_{rmse}_{datetime.now()}.h5')
