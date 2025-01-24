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
from scipy.integrate import cumulative_trapezoid


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
        X_aux.append(data[['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'hour', 'month', 'season']].iloc[i:i+time_steps].values)
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


def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub)
    Positions = np.zeros((SearchAgents_no, dim))

    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions


def FATA(fobj, lb, ub, dim, N, MaxFEs):
    worstInte = 0  # Parameters of Eq.(4)
    bestInte = np.inf  # Parameters of Eq.(4)
    noP = N
    arf = 0.2  # Eq. (15) reflectance=0.2
    gBest = np.zeros(dim)
    cg_curve = []
    gBestScore = np.inf  # Change this to -inf for maximization problems
    Flight = initialization(noP, dim, ub, lb)  # Initialize the set of random solutions
    fitness = np.full(noP, np.inf)
    it = 1
    FEs = 0
    lb = np.full(dim, lb)  # lower boundary
    ub = np.full(dim, ub)  # upper boundary

    # Main loop
    while FEs < MaxFEs:
        for i in range(Flight.shape[0]):
            Flag4ub = Flight[i, :] > ub
            Flag4lb = Flight[i, :] < lb
            Flight[i, :] = Flight[i, :] * (~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
            FEs += 1
            fitness[i] = fobj(Flight[i, :])
            # Make greedy selections
            if gBestScore > fitness[i]:
                gBestScore = fitness[i]
                gBest = Flight[i, :].copy()

        Order = np.sort(fitness)
        worstFitness = Order[N - 1]
        bestFitness = Order[0]

        # The mirage light filtering principle
        Integral = cumulative_trapezoid(Order, initial=0)
        if Integral[N - 1] > worstInte:
            worstInte = Integral[N - 1]
        if Integral[N - 1] < bestInte:
            bestInte = Integral[N - 1]
        IP = (Integral[N - 1] - worstInte) / (
                    bestInte - worstInte + np.finfo(float).eps)  # Eq.(4) population quality factor

        # Calculation Para1 and Para2
        a = np.tan(-(FEs / MaxFEs) + 1)
        b = 1 / np.tan(-(FEs / MaxFEs) + 1)

        for i in range(Flight.shape[0]):
            Para1 = a * np.random.rand(dim) - a * np.random.rand(dim)  # Parameters of Eq.(10)
            Para2 = b * np.random.rand(dim) - b * np.random.rand(dim)  # Parameters of Eq.(13)
            p = (fitness[i] - worstFitness) / (gBestScore - worstFitness + np.finfo(
                float).eps)  # Parameters of Eq.(5) individual quality factor

            if np.random.rand() > IP:
                Flight[i, :] = (ub - lb) * np.random.rand(dim) + lb
            else:
                for j in range(dim):
                    num = np.random.randint(N)
                    if np.random.rand() < p:
                        Flight[i, j] = gBest[j] + Flight[i, j] * Para1[j]  # Light refraction (first phase) Eq.(8)
                    else:
                        Flight[i, j] = Flight[num, j] + Para2[j] * Flight[
                            i, j]  # Light refraction (second phase) Eq.(11)
                        Flight[i, j] = (0.5 * (arf + 1) * (lb[j] + ub[j]) - arf * Flight[
                            i, j])  # Light total internal reflection Eq.(14)

        cg_curve.append(gBestScore)
        it += 1
        bestPos = gBest.copy()

    return bestPos, gBestScore, cg_curve


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
SearchAgents_no = 130
Max_iter = 200
lb = np.array([1e-6, 0.1])  # 变量下界
ub = np.array([1e-4, 0.4])  # 变量上界
dim = 2

Leeches_best_pos, Leeches_best_score, Convergence_curve = FATA(objective_function, lb, ub, dim, SearchAgents_no, Max_iter)

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

