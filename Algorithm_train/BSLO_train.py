import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping
import scipy.special

# 全局计数器
function_call_counter = 0

# 全局最优MSE
global_best_mse = np.inf

# 全局最优参数
global_best_params = None

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

def evaluate_model(params):
    global function_call_counter
    global global_best_mse
    global global_best_params

    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")

    params = params.flatten()  # 展平二维数组为一维
    learning_rate, dropout_rate = params[0], params[1]

    model = build_ts_lstme_model(time_steps, aux_feature_dim, r, dropout_rate, learning_rate)

    # 使用早停防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit([X_main_train, X_aux_train], y_train, epochs=50, batch_size=32,
                        validation_data=([X_main_val, X_aux_val], y_val), callbacks=[early_stopping])

    # 预测并反标准化
    y_pred = model.predict([X_main_val, X_aux_val])
    y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))

    # 计算评价指标（使用均方误差）
    mse_test = mean_squared_error(y_val_inv, y_pred_inv)
    r2_test = r2_score(y_val_inv, y_pred_inv)

    print(f"learning_rate:{learning_rate}, dropout_rate:{dropout_rate}")
    print(f"R2: {r2_test}, MSE: {mse_test}")

    # 更新并输出最优MSE和对应的参数
    if mse_test < global_best_mse:
        global_best_mse = mse_test
        global_best_params = (learning_rate, dropout_rate)

    print(f"当前最优解 - learning_rate: {global_best_params[0]}, Dropout: {global_best_params[1]}, MSE: {global_best_mse}")

    return mse_test

def BSLO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    # initialize best Leeches
    Leeches_best_pos = np.zeros(dim)
    Leeches_best_score = float('inf')
    # Convert lb and ub to NumPy arrays
    lb = np.array(lb)
    ub = np.array(ub)
    # Initialize the positions of search agents
    Leeches_Positions = initialization(SearchAgents_no, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)
    Temp_best_fitness = np.zeros(Max_iter)
    fitness = np.zeros(SearchAgents_no)
    # Initialize parameters
    t = 0
    m = 0.8
    a = 0.97
    b = 0.001
    t1 = 20
    t2 = 20
    # Main loop
    while t < Max_iter:
        N1 = int((m + (1 - m) * (t / Max_iter) ** 2) * SearchAgents_no)
        # calculate fitness values
        for i in range(Leeches_Positions.shape[0]):
            # boundary checking
            Leeches_Positions[i, :] = np.clip(Leeches_Positions[i, :], lb, ub)
            # Calculate objective function for each search agent
            fitness[i] = fobj(Leeches_Positions[i, :])
            # Update best Leeches
            if fitness[i] <= Leeches_best_score:
                Leeches_best_score = fitness[i]
                Leeches_best_pos = Leeches_Positions[i, :].copy()

        Prey_Position = Leeches_best_pos
        # Re-tracking strategy
        Temp_best_fitness[t] = Leeches_best_score
        if t > t1:
            if Temp_best_fitness[t] == Temp_best_fitness[t - t2]:
                for i in range(Leeches_Positions.shape[0]):
                    if fitness[i] == Leeches_best_score:
                        Leeches_Positions[i, :] = np.random.rand(dim) * (ub - lb) + lb

        if np.random.rand() < 0.5:
            s = 8 - 1 * (-(t / Max_iter) ** 2 + 1)
        else:
            s = 8 - 7 * (-(t / Max_iter) ** 2 + 1)

        beta = -0.5 * (t / Max_iter) ** 6 + (t / Max_iter) ** 4 + 1.5
        LV = 0.5 * levy(SearchAgents_no, dim, beta)

        # Generate random integers
        minValue = 1  # minimum integer value
        maxValue = int(SearchAgents_no * (1 + t / Max_iter))  # maximum integer value
        k2 = np.random.randint(minValue, maxValue, (SearchAgents_no, dim))
        k = np.random.randint(minValue, dim, (SearchAgents_no, dim))

        for i in range(N1):
            for j in range(Leeches_Positions.shape[1]):
                r1 = 2 * np.random.rand() - 1
                r2 = 2 * np.random.rand() - 1
                r3 = 2 * np.random.rand() - 1
                PD = s * (1 - (t / Max_iter)) * r1
                if abs(PD) >= 1:
                    # Exploration of directional leeches
                    b = 0.001
                    W1 = (1 - t / Max_iter) * b * LV[i, j]
                    L1 = r2 * abs(Prey_Position[j] - Leeches_Positions[i, j]) * PD * (1 - k2[i, j] / SearchAgents_no)
                    L2 = abs(Prey_Position[j] - Leeches_Positions[i, k[i, j]]) * PD * (
                                1 - (r2 ** 2) * (k2[i, j] / SearchAgents_no))
                    if np.random.rand() < a:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, j] - L1
                        else:
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, j] + L1
                    else:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, k[i, j]] - L2
                        else:
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, k[i, j]] + L2
                else:
                    # Exploitation of directional leeches
                    if t >= 0.1 * Max_iter:
                        b = 0.00001
                    W1 = (1 - t / Max_iter) * b * LV[i, j]
                    L3 = abs(Prey_Position[j] - Leeches_Positions[i, j]) * PD * (1 - r3 * k2[i, j] / SearchAgents_no)
                    L4 = abs(Prey_Position[j] - Leeches_Positions[i, k[i, j]]) * PD * (
                                1 - r3 * k2[i, j] / SearchAgents_no)
                    if np.random.rand() < a:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] - L3
                        else:
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] + L3
                    else:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] - L4
                        else:
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] + L4

        # Search strategy of directionless Leeches
        for i in range(N1, Leeches_Positions.shape[0]):
            for j in range(Leeches_Positions.shape[1]):
                if np.min(lb) >= 0:
                    LV[i, j] = abs(LV[i, j])
                if np.random.rand() > 0.5:
                    Leeches_Positions[i, j] = (t / Max_iter) * LV[i, j] * Leeches_Positions[i, j] * abs(
                        Prey_Position[j] - Leeches_Positions[i, j])
                else:
                    Leeches_Positions[i, j] = (t / Max_iter) * LV[i, j] * Prey_Position[j] * abs(
                        Prey_Position[j] - Leeches_Positions[i, j])

        t += 1
        Convergence_curve[t - 1] = Leeches_best_score

    return Leeches_best_score, Leeches_best_pos, Convergence_curve


def levy(n, m, beta):
    num = scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2)  # used for Numerator
    den = scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  # used for Denominator
    sigma_u = (num / den) ** (1 / beta)  # Standard deviation

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    z = u / (np.abs(v) ** (1 / beta))
    return z


def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = np.size(ub)
    X = np.zeros((SearchAgents_no, dim))
    if Boundary_no == 1:
        X = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    elif Boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return X


def objective_function(params):
    # 参数范围
    lb = np.array([1e-6, 0.1])  # 变量下界
    ub = np.array([1e-4, 0.4])  # 变量上界

    if np.any(params < lb) or np.any(params > ub):
        print("参数越界，跳过这次训练")
        return np.inf  # 返回一个很大的值作为惩罚

    return evaluate_model(params)


main_feature_dim = 1  # 主要输入的特征维度，即AQI
aux_feature_dim = X_aux_train.shape[2]  # 辅助输入的特征维度
r = 24
SearchAgents_no = 150
Max_iter = 200
lb = np.array([1e-6, 0.1])  # 变量下界
ub = np.array([1e-4, 0.4])  # 变量上界
dim = 2

Leeches_best_score, Leeches_best_pos, Convergence_curve = BSLO(SearchAgents_no, Max_iter, lb, ub, dim,
                                                               objective_function)

# 打印结果
print("最优解 (学习率和Dropout率):", Leeches_best_pos)
print("最优适应度 (MSE):", Leeches_best_score)

# Plot Convergence curve
plt.plot(Convergence_curve, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.title('Convergence Curve')
plt.grid(True)
plt.show()

