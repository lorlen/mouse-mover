from os import PathLike

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train(
    cursor_dataset: PathLike | str,
    model_dir: PathLike | str,
    scaler_file: PathLike | str,
    epochs: int,
):
    df = pd.read_parquet(cursor_dataset)

    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    seq_length = 10
    data = []
    for i in range(len(df) - seq_length):
        data.append(df.iloc[i : i + seq_length].values)
    data = np.array(data)

    X = data[:, :-1, :]
    y = data[:, -1, :]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(3))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    model.save(model_dir)

    joblib.dump(scaler, scaler_file)
