import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 读取数据
file_path = r"D:\AQI\AQI_data\14-23.csv"
data = pd.read_csv(file_path)

# 数据预处理
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', errors='coerce')
data['month'] = data['date'].dt.month
data['season'] = (data['month'] % 12 + 3) // 3  # 将月份转化为季节

# 划分训练集和验证集
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

# 特征和目标列名称
features = ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2', 'hour', 'month', 'season']
target = 'AQI'

# 初始化归一化器
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 对训练集特征和目标进行归一化
train_features_scaled = feature_scaler.fit_transform(train_data[features])
train_target_scaled = target_scaler.fit_transform(train_data[[target]])

# 对测试集特征和目标进行归一化
test_features_scaled = feature_scaler.transform(test_data[features])
test_target_scaled = target_scaler.transform(test_data[[target]])

# 将数据重新转换为DataFrame
train_data.loc[:, features] = train_features_scaled
train_data.loc[:, [target]] = train_target_scaled
test_data.loc[:, features] = test_features_scaled
test_data.loc[:, [target]] = test_target_scaled

# 保存归一化器
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# 创建时间序列数据
time_steps = 24
def create_sequences(data, time_steps):
    X_main, X_aux, y = [], [], []
    for i in range(len(data) - time_steps):
        X_main.append(data['AQI'].iloc[i:i+time_steps].values)
        X_aux.append(data[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'hour', 'month', 'season']].iloc[i:i+time_steps].values)
        y.append(data['AQI'].iloc[i+time_steps])
    return np.array(X_main), np.array(X_aux), np.array(y)


X_main_train, X_aux_train, y_train = create_sequences(train_data, time_steps)
X_main_val, X_aux_val, y_val = create_sequences(test_data, time_steps)

# 打印数据集的信息
print(f"训练数据 X_main_train 的形状: {X_main_train.shape}")
print(f"训练数据 X_aux_train 的形状: {X_aux_train.shape}")
print(f"训练数据 y_train 的形状: {y_train.shape}")
print(f"验证数据 X_main_val 的形状: {X_main_val.shape}")
print(f"验证数据 X_aux_val 的形状: {X_aux_val.shape}")
print(f"验证数据 y_val 的形状: {y_val.shape}")

# 定义TS-LSTME模型
def build_ts_lstme_model(time_steps, aux_feature_dim, r, dropout_rate, learning_rate):
    # 输入层
    main_input = Input(shape=(time_steps, 1), name='main_input')
    aux_input = Input(shape=(time_steps, aux_feature_dim), name='aux_input')

    # LSTM 层
    x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(main_input)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # 在每个时间步进行预测
    outputs = []
    for i in range(0, time_steps, r):
        if i + r < time_steps:
            temp_x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(x)
        else:
            temp_x = Bidirectional(LSTM(32, return_sequences=False, activation='relu'))(x)
        temp_x = Dense(16, activation='relu')(temp_x)
        temp_x = Dropout(dropout_rate)(temp_x)
        temp_x = Dense(1, activation='linear')(temp_x)  # 预测单步
        temp_x = Flatten()(temp_x)
        outputs.append(temp_x)

    # 合并所有预测输出
    concatenated_outputs = concatenate(outputs, axis=-1)

    # 辅助输入的处理
    aux_dense = Dense(16, activation='relu')(aux_input)
    aux_dense = Dropout(dropout_rate)(aux_dense)
    aux_dense = Flatten()(aux_dense)

    # 最终拼接和输出
    final_concat = concatenate([concatenated_outputs, aux_dense], axis=-1)
    final_output = Dense(1, activation='linear', name='final_output')(final_concat)  # 预测单步的 AQI

    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=[main_input, aux_input], outputs=final_output)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model


# 构建和训练模型
main_feature_dim = 1  # 主要输入的特征维度，即AQI
aux_feature_dim = X_aux_train.shape[2]  # 辅助输入的特征维度
r = 24
dropout_rate = 0.163415157721894
learning_rate = 0.000067069190765451

model = build_ts_lstme_model(time_steps, aux_feature_dim, r, dropout_rate, learning_rate)

# 使用早停防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# history = model.fit([X_main_train, X_aux_train], y_train, epochs=50, batch_size=32, validation_data=([X_main_val, X_aux_val], y_val), callbacks=[early_stopping])
history = model.fit([X_main_train, X_aux_train], y_train, epochs=50, batch_size=32, validation_data=([X_main_val, X_aux_val], y_val))


# 计算IA (Index of Agreement)
def index_of_agreement(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    return 1 - (numerator / denominator)


# 计算训练集的预测值
y_train_pred = model.predict([X_main_train, X_aux_train])
y_train_pred_inv = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
y_train_inv = target_scaler.inverse_transform(y_train.reshape(-1, 1))

# 计算验证集的预测值
y_pred = model.predict([X_main_val, X_aux_val])
y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))
print(y_val_inv)
print(y_pred_inv)

# 计算训练集的评价指标
mae_train = mean_absolute_error(y_train_inv, y_train_pred_inv)
mse_train = mean_squared_error(y_train_inv, y_train_pred_inv)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train_inv, y_train_pred_inv)
ia_train = index_of_agreement(y_train_inv, y_train_pred_inv)

# 计算测试集的评价指标
mae_test = mean_absolute_error(y_val_inv, y_pred_inv)
mse_test = mean_squared_error(y_val_inv, y_pred_inv)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_val_inv, y_pred_inv)
ia_test = index_of_agreement(y_val_inv, y_pred_inv)

# 打印评价指标
print(f"训练集 MSE: {mse_train:.3f}")
print(f"训练集 RMSE: {rmse_train:.3f}")
print(f"训练集 MAE: {mae_train:.3f}")
print(f"训练集 R²: {r2_train:.3f}")
print(f"训练集 IA: {ia_train:.3f}")
print("***************************************************")
print(f"测试集 MSE: {mse_test:.3f}")
print(f"测试集 RMSE: {rmse_test:.3f}")
print(f"测试集 MAE: {mae_test:.3f}")
print(f"测试集 R²: {r2_test:.3f}")
print(f"测试集 IA: {ia_test:.3f}")

# 保存训练和验证损失到 DataFrame
loss_df = pd.DataFrame({
    'epoch': range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

# 添加评价指标到 DataFrame
metrics_df = pd.DataFrame({
    'mae_train': [mae_train],
    'mse_train': [mse_train],
    'rmse_train': [rmse_train],
    'r2_train': [r2_train],
    'ia_train': [ia_train],
    'mae_test': [mae_test],
    'mse_test': [mse_test],
    'rmse_test': [rmse_test],
    'r2_test': [r2_test],
    'ia_test': [ia_test]
})

# 合并损失数据和评价指标数据
full_df = pd.concat([loss_df, metrics_df], axis=1)

# 保存数据到CSV文件
full_df.to_csv("CFOA.csv", index=False)

# 保存模型
model.save("CFOA.h5")

print("训练和验证损失及评价指标已保存到 CSV 文件。")

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制预测值与真实值对比图
plt.figure(figsize=(12, 6))
plt.plot(y_val_inv, label='Actual Values', color='blue')
plt.plot(y_pred_inv, label='Predicted Values', color='red')
plt.title('Comparison of Predicted and Actual Values')
plt.xlabel('Time Steps')
plt.ylabel('AQI')
plt.legend()
plt.grid(True)
plt.show()
plt.close()

# 绘制loss下降图
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.close()
