import numpy as np

from service import Service


if __name__ == "__main__":
    service = Service("lstm_model/")
    service.add_movie(
        {"subscribers": 100, "video_time": 10, "tag": "FUNNY", "upload_time": 10000}
    )
    service.add_movie(
        {"subscribers": 100, "video_time": 10, "tag": "FUNNY", "upload_time": 10000 + 120}
    )
    print(service.model.predict(np.array([100, 10, 'FUNNY']))[:5])
    print(service.predict_max_stress(10200))
