"""
    python TOV_Solver.py --input dataframes/sample_eos.csv --output dataframes/sample_mr.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers
from preprocessing import tov_load_and_preprocess

def build_wavenet(input_length=32, input_channels=1, output_channels=2, filters=64):
    inp = layers.Input(shape=(input_length, input_channels))
    x = inp

    x = layers.Conv1D(
        filters=filters, kernel_size=2, padding="causal",
        activation="elu",
        kernel_initializer=initializers.GlorotUniform(),
        kernel_regularizer=regularizers.l2(1e-7),
    )(inp)
    
    # hidden stack (ELU, causal, dilations as in the paper)
    dilations = [1, 2, 4, 8, 16, 32, 16, 32]
    for d in dilations:
        x = layers.Conv1D(
            filters=filters, kernel_size=2, dilation_rate=d, padding="causal",
            activation="elu",
            kernel_initializer=initializers.GlorotUniform(),
            kernel_regularizer=regularizers.l2(1e-7),
        )(x)

    # output conv (Sigmoid; 2 channels for [M, R])
    out = layers.Conv1D(
        filters=output_channels, kernel_size=2, dilation_rate=64, padding="causal",
        activation="sigmoid",
        kernel_initializer=initializers.GlorotUniform(),
        kernel_regularizer=regularizers.l2(1e-7),
    )(x)

    print(models.Model(inp, out).summary())

    return models.Model(inp, out, name="WaveNet_TOV")

def train_model(model: tf.keras.Model, X: np.ndarray, Y: np.ndarray, save_dir: str = 'models', epochs: int = 3000, batch_size: int = 32, val_split: float = 0.2):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'tov_solver.h5')
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='mse', metrics=['mae'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True, monitor='val_loss', mode='min'),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    ]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=val_split*2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
    
    np.save('dataframes/X_test.npy', X_test)
    np.save('dataframes/y_test.npy', y_test)

def evaluate_and_predict(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, mr_scaler: MinMaxScaler):
    test_metrics = model.evaluate(X_test, y_test, verbose=2)
    y_pred = model.predict(X_test)
    
    return test_metrics, y_pred


def save_model(model: tf.keras.Model, path: str):
    model.save(path)


def load_trained_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train WaveNet TOV model from CSVs')
    parser.add_argument('--input', required=True, type=str, default='sample_eos.csv')
    parser.add_argument('--output', required=True, type=str, default='sample_mr.csv')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--np', type=int, default=32, dest='Np')
    args = parser.parse_args()
    X_scaled, Y_scaled = tov_load_and_preprocess(args.input, args.output, Np=args.Np)
    model = build_wavenet(input_length=args.Np)
    train_model(model, X_scaled, Y_scaled, epochs=args.epochs, batch_size=args.batch)
