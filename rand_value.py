import torch
import pandas as pd

# Generate random data
random_data = torch.randn(1000, 2)

# Convert to a Pandas DataFrame
df = pd.DataFrame(random_data.numpy(), columns=['temperature and humidity', 'press'])

# Save DataFrame to CSV
df.to_csv('random_data.csv', index=False)

print("CSV file saved successfully.")
