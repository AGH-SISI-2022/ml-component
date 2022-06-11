import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers as layers
from tensorflow.keras import optimizers as optimizers
from tensorflow.keras import callbacks as callbacks
from tensorflow.keras import models as models


FORECAST_TOTAL_TIME = 7200
SAMPLE_TIME = 120
PEAK_TIME = 10

N_SAMPLES = FORECAST_TOTAL_TIME // SAMPLE_TIME
N_BINS_TOTAL = FORECAST_TOTAL_TIME // PEAK_TIME
N_BINS_SAMPLE = SAMPLE_TIME // PEAK_TIME

TAGS = ['SPORTS', 'NEWS', 'GAMING', 'FUNNY', 'DOCUMENTARY', 'COOKING', 'SCIENCE']


class Model():
    def  __init__(self, path: str):
        self.path = path
        self.model = self._build_model()
        self.y_avg_norm = None
        self.binarizer = LabelBinarizer().fit(TAGS)
        self.scaler_mean = None
        self.scaler_scale = None

    def train(
        self,
        X: np.array,
        Y: np.array,
        lr: int = 0.002,
        epochs: int = 500,
        bs: int = 1024,
    ):
        X = np.hstack([X[:, :2], self.binarizer.transform(X[:, 2])])
        X = X.astype(np.float16)

        scaler = StandardScaler()
        X[:, :2] = scaler.fit_transform(X[:, :2])
        
        self.y_avg_norm = np.mean(np.linalg.norm(Y))
        Y = Y / self.y_avg_norm

        json.dump(
            {
                "y_avg_norm": self.y_avg_norm,
                "scaler_mean": list(self.scaler.mean_),
                "scaler_scale": list(self.scaler.scale_)
            },
            open(self.path + "/config.json", "w+"),
        )

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.model.compile(
            optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model.fit(
            (Y_train, X_train), 
            Y_train,
            batch_size=bs,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            callbacks=[
                callbacks.TensorBoard(histogram_freq=1),
                callbacks.ModelCheckpoint(self.path +'/weights', save_best_only=True, save_weights_only=True),
            ],
        )

    def load_weights(self):
        self.model.load_weights(self.path + '/weights')
        config = json.load(open(self.path + "/config.json", "r"))
        self.y_avg_norm = config['y_avg_norm']
        self.scaler_mean = np.array(config['scaler_mean'])
        self.scaler_scale = np.array(config['scaler_scale'])

    def predict(self, movie_data: np.array) -> np.array:
        x = np.hstack([movie_data[:2], self.binarizer.transform([movie_data[2]])[0]])
        x = x.astype(np.float16)
        x[:2] = (x[:2] - self.scaler_mean) / self.scaler_scale
        y = self.model.predict([x]).reshape(-1)
        y *= self.y_avg_norm
        return y
    
    def _build_model(self) -> tf.keras.Model:
        model = AutoregressiveLSTM(18, N_SAMPLES)
        return model


class AutoregressiveLSTM(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = layers.LSTMCell(units)
        self.dense = layers.Dense(1)
    
    def call(self, inputs, training=None):
        predictions = []
        
        if training:
            time_series, features = inputs
            time_series = layers.Reshape((-1, 1))(time_series)
        else:
            features = inputs
        
        state = [
            tf.concat([features, features], axis=1),
            tf.concat([features, features], axis=1)
        ]
        x = tf.zeros((tf.shape(features)[0], 1))

        for step_i in range(self.out_steps):
            x, state = self.lstm_cell(x, states=state,
                                     training=training)
            prediction = self.dense(x)
            predictions.append(prediction)
            
            if training:
                x = time_series[:, step_i]
            else:
                x = prediction
                
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions