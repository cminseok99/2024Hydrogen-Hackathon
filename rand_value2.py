import csv
import random

# CSV 파일 경로
csv_file = "random_data2.csv"

# 데이터 생성
data = []
for _ in range(1000):
    temperature = round(random.uniform(20, 30), 2)
    humidity = round(random.uniform(40, 60), 2)
    data.append([temperature, humidity])

# CSV 파일에 데이터 쓰기
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Temperature', 'Humidity'])  # 헤더 쓰기
    writer.writerows(data)

print(f"CSV 파일 '{csv_file}'이(가) 생성되었습니다.")
