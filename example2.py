import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import requests


class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    def fit(self, X):
        self.clf.fit(X)

    def predict(self, X):
        return self.clf.fit_predict(X)

    def score_samples(self, X):
        return self.clf.negative_outlier_factor_

# CSV 파일에서 기존 데이터 읽기
file_path = 'C:/Users/차민석/Desktop/a/random_data1.csv'
data_csv = pd.read_csv(file_path)

# 기존 데이터 준비
X = data_csv.values

# LOF 모델 초기화 및 학습
lof_model = LOFModel(n_neighbors=20, contamination=0.1)
lof_model.fit(X)

mean_temperature = np.mean(X[:, 0])
std_temperature = np.std(X[:, 0])
mean_humidity = np.mean(X[:, 1])
std_humidity = np.std(X[:, 1])

# 새로운 센서 데이터 받아오기
new_sensor_data = []
for _ in range(100):
    data = requests.get('https://3d99-112-184-243-68.ngrok-free.app')
    data = data.json()
    temperature = data.get('temperature')
    humidity = data.get('humidity')

    # 정규화
    temperature_normalized = (temperature - mean_temperature) / std_temperature
    humidity_normalized = (humidity - mean_humidity) / std_humidity
    new_sensor_data.append([temperature_normalized, humidity_normalized])

new_sensor_data = np.array(new_sensor_data)

# 이상치 여부 예측
is_outlier = lof_model.predict(new_sensor_data)

# 이상치 점수 계산
outlier_scores = lof_model.score_samples(new_sensor_data)

# 이상치에 대한 정보를 서버에 전송
for i, outlier in enumerate(is_outlier):
    if outlier == -1:
        outlier_data = {
            'temperature': new_sensor_data[i][0],  # 이상치인 경우의 온도
            'humidity': new_sensor_data[i][1],     # 이상치인 경우의 습도
            'outlier_score': outlier_scores[i]     # 이상치 점수
        }
        response = requests.post('https://3d99-112-184-243-68.ngrok-free.app', json=outlier_data)
        url = 'https://3d99-112-184-243-68.ngrok-free.app'
        headers = {'Accept': 'application/json'}
        if response.status_code == 200:
            print(f"이상치를 서버에 성공적으로 전송했습니다: {outlier_data}")
        else:
            print(f"이상치 전송에 실패했습니다: {response.status_code}")
