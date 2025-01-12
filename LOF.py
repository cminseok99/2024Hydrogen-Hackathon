import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import random

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
file_path = 'C:/Users/차민석/Desktop/a/random_data2.csv'
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
for _ in range(10):
    temperature = random.uniform(20, 30)
    humidity = random.uniform(40, 60)

    # 정규화
    temperature_normalized = (temperature - mean_temperature) / std_temperature
    humidity_normalized = (humidity - mean_humidity) / std_humidity
    new_sensor_data.append([temperature_normalized, humidity_normalized])

new_sensor_data = np.array(new_sensor_data)

# 이상치 여부 예측
is_outlier = lof_model.predict(new_sensor_data)

# 이상치 점수 계산
outlier_scores = lof_model.score_samples(new_sensor_data)

# 이상치 시각화
# 시각화 코드를 추가하여 이상치를 적절히 표시할 수 있습니다.
import matplotlib.pyplot as plt

# 이상치 시각화
plt.figure(figsize=(10, 6))

# 이상치를 시각화할 때 사용할 색상
colors = np.array(['blue', 'red'])

# 이상치가 아닌 데이터 포인트를 먼저 플로팅합니다.
plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', s=20)

# 이상치를 빨간색으로 플로팅합니다.
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', edgecolor='k', s=80, marker='x', label='이상치')

plt.title('LOF 이상치 탐지')
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.legend()
plt.grid(True)
plt.show()

