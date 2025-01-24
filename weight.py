import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from pyswarm import pso

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 全局变量初始化
best_mse = float('inf')
objective_call_count = 0
global_best_params = None
global_R2 = None

# 读取数据
file_path = r"D:\AQI\AQI_data\14-23.csv"
data = pd.read_csv(file_path)

# 数据预处理
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', errors='coerce')
data['month'] = data['date'].dt.month
data['season'] = (data['month'] % 12 + 3) // 3  # 将月份转化为季节

# 计算预测部分数据量
test_size = int(0.2 * len(data))
test_data = data[-test_size:].copy()

# 特征和目标列名称
features = ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2', 'hour', 'month', 'season']
target = 'AQI'

# 创建时间序列数据
time_steps = 24

def create_sequences(data, time_steps):
    X_main, X_aux, y = [], [], []
    for i in range(len(data) - time_steps):
        X_main.append(data['AQI'].iloc[i:i+time_steps].values)
        X_aux.append(data[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'hour', 'month', 'season']].iloc[i:i+time_steps].values)
        y.append(data['AQI'].iloc[i+time_steps])
    return np.array(X_main), np.array(X_aux), np.array(y)

X_main_test, X_aux_test, y_test = create_sequences(test_data, time_steps)

# 打印数据集的信息
print(f"测试数据 X_main_test 的形状: {X_main_test.shape}")
print(f"测试数据 X_aux_test 的形状: {X_aux_test.shape}")
print(f"测试数据 y_test 的形状: {y_test.shape}")

# 加载每个算法的预测结果
def load_predictions(algorithm_name):
    file_path = f"D:\\AQI\\predict\\{algorithm_name}.csv"
    predictions = pd.read_csv(file_path)
    predictions = predictions.iloc[1:]  # 删除第一行数据
    return predictions['Predicted'].values

lea_predictions = load_predictions('LEA')
cpo_predictions = load_predictions('CPO')
ego_predictions = load_predictions('EGO')
bslo_predictions = load_predictions('BSLO')
ivym_predictions = load_predictions('IVYA')
fivm_predictions = load_predictions('FIVM')
fata_predictions = load_predictions('FATA')
cfoa_predictions = load_predictions('CFOA')

def objective_function(weights):
    global best_mse, objective_call_count, global_best_params, global_R2

    # 增加目标函数调用计数
    objective_call_count += 1

    # 确保权重之和为1
    weights = np.abs(weights)
    weights /= np.sum(weights)

    # 加权组合预测结果
    combined_predictions = (
        weights[0] * lea_predictions +
        weights[1] * cpo_predictions +
        weights[2] * ego_predictions +
        weights[3] * bslo_predictions +
        weights[4] * ivym_predictions +
        weights[5] * fivm_predictions +
        weights[6] * fata_predictions +
        weights[7] * cfoa_predictions
    )

    # 计算评价指标
    mse_test = mean_squared_error(y_test, combined_predictions)
    r2 = r2_score(y_test, combined_predictions)

    # 更新最优MSE
    if mse_test < best_mse:
        global_R2 = r2
        best_mse = mse_test
        global_best_params = weights.copy()

    print(f"第 {objective_call_count} 次调用目标函数")
    print(f"当前权重: {weights}")
    print(f"R2: {r2}, MSE: {mse_test}")
    print(f"当前最优解 - weight: {global_best_params}, MSE: {best_mse}, R2: {global_R2}")

    return mse_test


# 设置PSO的边界和初始猜测
lb = [0] * 8
ub = [1] * 8
initial_guess = [1/8] * 8

# 使用PSO优化
best_weights, best_error = pso(objective_function, lb, ub, swarmsize=200, maxiter=500)

# 打印最佳权重和最小误差
print("最佳权重:", best_weights)
print("最小误差:", best_error)

# 使用最佳权重计算最终的加权预测结果
final_predictions = (
    best_weights[0] * lea_predictions +
    best_weights[1] * cpo_predictions +
    best_weights[2] * ego_predictions +
    best_weights[3] * bslo_predictions +
    best_weights[4] * ivym_predictions +
    best_weights[5] * fivm_predictions +
    best_weights[6] * fata_predictions +
    best_weights[7] * cfoa_predictions
)

# 计算评价指标
r2_final = r2_score(y_test, final_predictions)
mse_final = mean_squared_error(y_test, final_predictions)
rmse_final = np.sqrt(mse_final)
mae_final = np.mean(np.abs(final_predictions - y_test))
ia_final = 1 - (np.sum((y_test - final_predictions) ** 2) / np.sum((np.abs(final_predictions - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2))

print(f"Final R2: {r2_final}")
print(f"Final MSE: {mse_final}")
print(f"Final RMSE: {rmse_final}")
print(f"Final MAE: {mae_final}")
print(f"Final IA: {ia_final}")

# 保存最终预测结果
final_predictions_df = pd.DataFrame(final_predictions, columns=['predicted_AQI'])
final_predictions_df.to_csv(r"D:\AQI\predict\Final_Combined_Predictions.csv", index=False)
