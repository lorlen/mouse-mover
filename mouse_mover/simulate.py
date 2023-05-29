import itertools
from os import PathLike

import joblib
import mouse
import numpy as np
import tensorflow as tf


def simulate(model_dir: PathLike | str, scaler_file: PathLike | str, iterations: int):
    model = tf.keras.models.load_model(model_dir)
    scaler = joblib.load(scaler_file)

    sequence = np.random.normal(loc=1, scale=0.3, size=(1, 9, 3))
    sequence[:, :, 2] = np.abs(sequence[:, :, 2])

    for _ in range(iterations) if iterations >= 0 else itertools.count():
        prediction = model.predict(sequence)
        prediction = scaler.inverse_transform(prediction)
        dx, dy, dt = prediction[0]

        mouse.move(dx, dy, absolute=False, duration=abs(float(dt)))

        sequence = np.roll(sequence, -1, axis=1)
        sequence[0, -1, :] = prediction
