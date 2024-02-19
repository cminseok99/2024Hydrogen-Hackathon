import torch

# 랜덤 데이터 생성
torch.manual_seed(42)
num_samples = 100
num_features = 2

#randn -> 실행될떄마다 랜덤 변수 생성
data = torch.randn(num_samples, num_features)

# 모든 쌍의 거리 계산
distances = torch.cdist(data, data)

print(distances)