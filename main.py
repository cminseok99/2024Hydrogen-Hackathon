import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class GasConcentrationDataset(Dataset):
    def __init__(self, filename, seq_length):
        self.seq_length = seq_length
        self.csv = pd.read_csv(filename)
        
        # 데이터 정규화
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.csv.iloc[:, 1:4].values.astype('float32'))
        self.labels = self.scaler.fit_transform(self.csv.iloc[:, -1].values.reshape(-1, 1)).flatten()
        
    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        data = self.data[idx:idx+self.seq_length]
        label = self.labels[idx+self.seq_length]
        return torch.tensor(data, dtype=torch.float32).permute(1, 0), torch.tensor(label, dtype=torch.float32)


# 데이터 생성 함수
def generate_random_gas_concentration_data(start_date, end_date):
    num_hours = pd.date_range(start=start_date, end=end_date, freq='H').shape[0]
    hydrogen_concentration = np.random.uniform(0.1, 0.5, num_hours)
    
    data = {
        'timestamp': pd.date_range(start=start_date, end=end_date, freq='H'),
        'hydrogen_concentration': hydrogen_concentration,
    }

    df = pd.DataFrame(data)
    csv_filename = 'random_gas_concentration_data.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data generated and saved to {csv_filename}")

# 데이터 로드
start_date = '2024-01-01'
end_date = '2024-01-31'
generate_random_gas_concentration_data(start_date, end_date)

# 데이터셋 및 데이터로더 생성
seq_length = 24
dataset = GasConcentrationDataset('random_gas_concentration_data.csv', seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn

class LSTMModelWithDropout(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(LSTMModelWithDropout, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)  # 드롭아웃 레이어 추가
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)  # 드롭아웃 적용
        out = self.fc(lstm_out[:, -1, :])
        return out


# 모델 초기화
input_size = 24  # 입력 차원 수 (수소 농도, 질소 농도 등)
hidden_size = 100  # 은닉 상태 크기
output_size = 1  # 출력 차원 수 (수소 농도 예측)
model = LSTMModelWithDropout(input_size, hidden_size, output_size)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 100  # 더 많은 에폭으로 학습
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        batch_data = batch_data.permute(1, 0, 2)
        output = model(batch_data)
        loss = criterion(output, batch_labels.unsqueeze(1))  # 레이블의 차원을 맞춤
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 테스트 데이터로 예측
predictions = []
labels = []
with torch.no_grad():
    for batch_data, batch_labels in dataloader:
        output = model(batch_data)
        predictions.extend(output.squeeze().tolist())
        labels.extend(batch_labels.tolist())

# 스케일러 역변환
predictions = dataset.scaler.inverse_transform(np.array(predictions, dtype=np.float32).reshape(-1, 1))
labels = dataset.scaler.inverse_transform(np.array(labels, dtype=np.float32).reshape(-1, 1))

# 예측 결과 그래프 출력
plt.figure(figsize=(12, 6))
plt.plot(labels, label='Actual Hydrogen Concentration')
plt.plot(predictions, label='Predicted Hydrogen Concentration')
plt.title('Hydrogen Concentration Prediction')
plt.xlabel('Time')
plt.ylabel('Hydrogen Concentration')
plt.legend()
plt.show()
