import pandas as pd
import numpy as np
import time

from model import Model, FORECAST_TOTAL_TIME, SAMPLE_TIME

class Service():
    def __init__(self, model_path: str):
        self.model = Model(model_path)
        self.model.load_weights()

        self.time_series = np.zeros(FORECAST_TOTAL_TIME // SAMPLE_TIME)
        self.start_time = None
    
    def add_movie(self, movie_data: dict) -> None:
        """
        Update time_series with new movie predictions
        :param movie_data: dict with the data describing a movie; must contain keys: 
        upload_time, subscribers, video_time, tag
        """
        upload_time = movie_data['upload_time']
        movie_data = np.array([
            movie_data['subscribers'],
            movie_data['video_time'],
            movie_data['tag']
        ])

        self._move_window(upload_time)

        pred_time_series = self.model.predict(movie_data)
        self.time_series += pred_time_series
    
    def predict_max_stress(self, current_time: int = None, forecast_time: int = 5 * SAMPLE_TIME) -> int:
        """
        Return the maximum stress for the next forecast_time seconds
        :param current_time: current time
        :param forecast_time: time of the prediction in seconds
        :return predicted stress
        """
        current_time = int(time.time()) if current_time is None else current_time
        self._move_window(current_time)
        n_to_predict = Service._adjust_time(forecast_time)
        return int(max(self.time_series[:n_to_predict]))

    def _move_window(self, current_time: int) -> None:
        current_time = Service._adjust_time(current_time)

        if self.start_time is None:  # for the first time
            self.start_time = current_time
        else:
            n_to_move = current_time - self.start_time
            if n_to_move > 0:
                self.start_time = current_time
                self.time_series[:-n_to_move] = self.time_series[n_to_move:]
                self.time_series[-n_to_move:] = np.zeros(n_to_move)

    @staticmethod
    def _adjust_time(time):
        return time // SAMPLE_TIME
