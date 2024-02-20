import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import random
import matplotlib.pyplot as plt

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

# 새로운 센서 데이터 받아오기
new_sensor_data = []
for _ in range(10):
    temperature = random.uniform(20, 30)
    humidity = random.uniform(40, 60)
    pressure = random.uniform(900, 1100)
    new_sensor_data.append([temperature, humidity, pressure])

new_sensor_data = np.array(new_sensor_data)

# 학습 데이터셋에 새로운 센서 데이터 추가
X = np.vstack((X, new_sensor_data))

# LOF 모델 재학습
lof_model.fit(X)

# 이상치 여부 예측
is_outlier = lof_model.predict(new_sensor_data)

# 시각화
plt.figure(figsize=(10, 6))

# 기존 데이터 플로팅
plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', s=20)

# 이상치 플로팅
plt.scatter(X[lof_model.predict(X) == -1, 0], X[lof_model.predict(X) == -1, 1], c='red', edgecolor='k', s=80, marker='x', label='outlier')

# 새로운 센서 데이터 플로팅
plt.scatter(new_sensor_data[:, 0], new_sensor_data[:, 1], c='orange', edgecolor='k', s=80, label='new sensor data')

# 이상치인 경우 빨간색으로 표시
for i, outlier in enumerate(is_outlier):
    if outlier == -1:
        plt.scatter(new_sensor_data[i, 0], new_sensor_data[i, 1], c='yellow', edgecolor='k', s=80, label='Outlier')
    # 정상치인 경우 초록색으로 표시
    else:
        plt.scatter(new_sensor_data[i, 0], new_sensor_data[i, 1], c='green', edgecolor='k', s=80, label='Normal')

plt.title('LOF Outlier Detection with New Sensor Data')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.legend()
plt.grid(True)
plt.show()
