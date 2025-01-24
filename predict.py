import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# 读取数据
file_path = r"D:\AQI\AQI_data\14-23.csv"
data = pd.read_csv(file_path)

# 数据预处理
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', errors='coerce')
data['month'] = data['date'].dt.month
data['season'] = (data['month'] % 12 + 3) // 3  # 将月份转化为季节

# 特征和目标列名称
features = ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2', 'hour', 'month', 'season']
target = 'AQI'

# 划分训练集和验证集
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

# 加载归一化器
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# 对测试集特征和目标进行归一化
test_features_scaled = feature_scaler.transform(test_data[features])
test_data.loc[:, features] = test_features_scaled

# 生成时间序列数据
time_steps = 24
def create_sequences(data, time_steps):
    X_main, X_aux, y = [], [], []
    for i in range(len(data) - time_steps):
        X_main.append(data['AQI'].iloc[i:i+time_steps].values)
        X_aux.append(data[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'hour', 'month', 'season']].iloc[i:i+time_steps].values)
        y.append(data['AQI'].iloc[i+time_steps])
    return np.array(X_main), np.array(X_aux), np.array(y)

X_main_val, X_aux_val, y_val = create_sequences(test_data, time_steps)

# 加载模型
model = load_model("CPO.h5")
output_path = r"D:\AQI\predict\CPO.csv"

# 使用模型进行预测
predicted_scaled = model.predict([X_main_val, X_aux_val])

# 反归一化预测结果
predicted = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

# 反归一化实际值
actual = target_scaler.inverse_transform(y_val.reshape(-1, 1))

# 保存预测结果为文件
results = pd.DataFrame({
    'Actual': actual.flatten(),
    'Predicted': predicted.flatten()
})
results.to_csv(output_path, index=False)

print(f"预测结果已保存到 {output_path}")

# 计算评价指标
r2 = r2_score(actual, predicted)
print(actual)
print(predicted)
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

# 计算IA (Index of Agreement)
def index_of_agreement(actual, predicted):
    numerator = np.sum((actual - predicted) ** 2)
    denominator = np.sum((np.abs(predicted - np.mean(actual)) + np.abs(actual - np.mean(actual))) ** 2)
    return 1 - numerator / denominator

ia = index_of_agreement(actual, predicted)

# 打印评价指标
print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"IA: {ia}")

# 仅绘制前100个点
predicted_100 = predicted[1:101]
actual_100 = actual[:100]

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(actual_100, label="Actual", color='blue')
plt.plot(predicted_100, label="Predicted", color='red')
plt.title("AQI Prediction vs Actual")
plt.xlabel("Time Steps")
plt.ylabel("AQI")
plt.legend()
plt.show()