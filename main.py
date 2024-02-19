import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#LOF.py에서 class LOF 호출
from LOF import LOF

file_path = 'random_data.csv'
data = pd.read_csv(file_path)

# 원점으로부터 각 데이터 포인트의 거리 계산
distances_from_origin = torch.norm(data, dim=1)



# +-1 범위 내에 있는 데이터 포인트를 파란색으로, 나머지를 빨간색으로 설정
colors = ['blue' if torch.abs(distance) <= 1 else 'red' for distance in distances_from_origin]

# 데이터 시각화
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=colors)
plt.title('Data visualization according to distance from origin')
plt.xlabel('temperature and humidity')
plt.ylabel('press')
plt.show()