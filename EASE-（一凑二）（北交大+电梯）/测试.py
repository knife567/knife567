import numpy as np

# 假设文件路径为 'data/BeiJiao/train/0.npy'
file_path = 'data/BeiJiao/test/1.npy'
data = np.load(file_path)
print(f"Data shape: {data.shape}")
print(f"First few rows of data:\n{data[:5]}")
