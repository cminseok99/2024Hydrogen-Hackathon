import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import requests
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
import threading
import json
import plotly.graph_objs as go
import plotly

# Flask 애플리케이션 초기화
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
db = SQLAlchemy(app)

# 센서 데이터 모델 정의
class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"SensorData(temperature={self.temperature}, humidity={self.humidity})"

# LOF 모델 클래스 정의
class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    def fit(self, X):
        self.clf.fit(X)

    def predict(self, X):
        return self.clf.fit_predict(X)

    def score_samples(self, X):
        return self.clf.negative_outlier_factor_

# 데이터베이스 생성 함수
def create_database():
    with app.app_context():
        db.create_all()

# JSON 파일에서 데이터를 읽어와 데이터베이스에 저장하는 함수
def store_sensor_data():
    with app.app_context():
        # JSON 파일에서 데이터 읽기
        file_path = 'C:/Users/차민석/Desktop/a/sensor_data.json'
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            # 데이터베이스에 데이터 추가
            for entry in data:
                temperature = entry.get('temperature')
                humidity = entry.get('humidity')
                new_data = SensorData(temperature=temperature, humidity=humidity)
                db.session.add(new_data)
                db.session.commit()

            # 레코드 수 제한 (100개로)
            num_records = SensorData.query.count()
            if num_records > 100:
                excess_records = num_records - 100
                oldest_records = SensorData.query.order_by(SensorData.id).limit(excess_records).all()
                for record in oldest_records:
                    db.session.delete(record)
                db.session.commit()

# 데이터 처리 및 시각화를 위한 Flask 라우트 및 함수
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'GET':
        data = SensorData.query.all()
        data_json = [{"temperature": d.temperature, "humidity": d.humidity} for d in data]
        return jsonify(data_json)
    elif request.method == 'POST':
        data = request.json
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        new_data = SensorData(temperature=temperature, humidity=humidity)
        db.session.add(new_data)
        db.session.commit()
        return jsonify({"message": "Data received successfully"})

@app.route('/chart')
def plot():
    data = SensorData.query.all()
    xdata = [d.temperature for d in data]
    ydata = [d.humidity for d in data]

    # Plotly를 사용하여 그래프 생성
    trace = go.Scatter(x=xdata, y=ydata, mode='markers', name='Data')
    layout = go.Layout(title='Data Plot', xaxis=dict(title='Temperature'), yaxis=dict(title='Humidity'))
    fig = go.Figure(data=[trace], layout=layout)

    # 그래프를 JSON 형태로 변환하여 반환
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('chart.html', graphJSON=graphJSON)

# LOF 모델 학습 및 이상치 판별
def train_lof_model():
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

# 데이터베이스 생성 함수 및 LOF 모델 학습 함수를 스레드로 실행
create_database_thread = threading.Thread(target=create_database)
create_database_thread.daemon = True
create_database_thread.start()

train_lof_model_thread = threading.Thread(target=train_lof_model)
train_lof_model_thread.daemon = True
train_lof_model_thread.start()

# Flask 서버 실행
if __name__ == '__main__':
    app.run(debug=True)