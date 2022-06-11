import pandas as pd
import numpy as np

from model import *


DATA_PATH = "data/requests_data.csv"
MODEL_SAVE_PATH = "lstm_model/"

def prepare_data(data: pd.DataFrame) -> tuple[np.array, np.array]:
    data_send = all_data[all_data['endpoint'] == '/send'].drop(columns=["endpoint", "request_count", "upload_time", "title", "youtuber_username"])
    data_send.set_index("id", inplace=True)

    data_watch = all_data[all_data['endpoint'] == '/watch'].drop(columns=["endpoint", "title", "youtuber_username", "video_time", "tag", "subscribers"])

    n_requests_dict = {}

    for movie_id in data_send.index:
        movie_requests_data = data_watch[np.logical_and(data_watch['id'] == movie_id, 
                                                        data_watch['current_time'] - data_watch['upload_time'] <= FORECAST_TOTAL_TIME)]
        
        upload_time = movie_requests_data['upload_time'].iloc[0]
        request_hist, bin_edges = np.histogram(
            a=movie_requests_data["current_time"],
            bins=N_BINS_TOTAL,
            range=(upload_time, upload_time + FORECAST_TOTAL_TIME),
            weights=movie_requests_data["request_count"]
        )
        
        request_max_count = []
        for start_bin in range(0, N_BINS_TOTAL, N_BINS_SAMPLE):
            request_max_count.append(max(request_hist[start_bin: start_bin + N_BINS_SAMPLE]))
        
        n_requests_dict[movie_id] = request_max_count

    feature_names = ['subscribers', 'video_time']

    X = np.hstack([data_send[feature_names].astype(np.float16), np.array(data_send["tag"]).reshape(-1, 1)])
    Y = np.array(list(n_requests_dict.values()))

    return X, Y


if __name__ == "__main__":
    all_data = pd.read_csv(DATA_PATH)

    X, Y = prepare_data(all_data)
    
    model = Model(MODEL_SAVE_PATH)
    model.train(X, Y)
