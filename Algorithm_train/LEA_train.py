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
features = ['AQI', 'PM2.5', 'PM10', 'NO2','CO',  'O3', 'SO2', 'hour', 'month', 'season']
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


def Initialization(N, dim, ub, lb):
    Boundary = len(ub)  # 变量边界的数量
    new_lb = lb
    new_ub = ub
    x = np.zeros((N, dim))  # 初始化种群

    # 如果所有变量的边界都相等，并且用户为ub和lb输入了一个单独的数字
    if Boundary == 1:
        x = np.random.rand(N, dim) * (ub - lb) + lb
        new_lb = np.array([lb] * dim)
        new_ub = np.array([ub] * dim)

    # 如果每个变量有不同的lb和ub
    if Boundary > 1:
        x = np.zeros((N, dim))
        for i in range(dim):
            ubi = ub[i]
            lbi = lb[i]
            x[:, i] = np.random.rand(N) * (ubi - lbi) + lbi

    return x, new_lb, new_ub

def LEA(fun, dim, lb, ub, N, MaxFEs):
    # Initialization
    CV = np.zeros(MaxFEs + 1)  # Convergence curves
    X, lb, ub = Initialization(N, dim, ub, lb)  # Initialize the population
    FE = 0  # Number of function evaluations
    H = np.zeros(N)  # The happiness degrees
    H_G = np.inf  # The best value

    for i in range(N):
        H[i] = fun(X[i])
        FE += 1
        if H_G > H[i]:
            H_G = H[i]
            G = X[i].copy()  # The best solution
        CV[FE] = H_G

    # Parameter settings
    h_max = 0.7
    h_min = 0
    lambda_c = 0.5
    lambda_p = 0.5

    # Main loop
    while FE < MaxFEs:
        h = (1 - FE / MaxFEs) * (h_max - h_min) + h_min  # Eq. (17)

        # Encounter
        r = np.random.permutation(N)
        A = X[r[:N // 2]]
        B = X[r[N // 2:]]
        H_A = H[r[:N // 2]]
        H_B = H[r[N // 2:]]

        # Stimulus phase
        c = Gap_P(H_A, H_B)  # Eq. (6)
        mu = np.sum(np.sqrt(np.sum((X - G) ** 2, axis=1)) / N) / dim + np.finfo(float).eps  # Eq. (10)

        for i in range(N // 2):
            if c[i] < lambda_c:
                # Value phase
                for j in range(dim):
                    phi1 = G[j] * A[i, j]
                    phi2 = G[j] ** 2 + A[i, j] * B[i, j]
                    phi3 = G[j] * B[i, j]

                    rho_A = np.sqrt((phi2 - phi1) ** 2)
                    rho_B = np.sqrt((phi2 - phi3) ** 2)

                    A[i, j] = np.random.rand() * A[i, j] + np.random.randn() * rho_A
                    B[i, j] = np.random.rand() * B[i, j] + np.random.randn() * rho_B

                FE += 1
                if FE > MaxFEs:
                    break
                A[i], H_A[i], G, H_G, CV = Update_A_mod(A[i], CV, FE, G, H_G, ub, lb, fun)

                FE += 1
                if FE > MaxFEs:
                    break

                B[i], H_B[i], G, H_G, CV = Update_B_mod(B[i], CV, FE, G, H_G, ub, lb, fun)

                p = (np.random.rand() + 0.5) * c[i] * np.sum(np.sqrt((A[i] - B[i]) ** 2)) / (dim * mu)  # Eq. (15)
                if p < lambda_p:
                    # Role phase
                    xi = A[i] * B[i]
                    xi = (xi - np.min(xi)) / (np.max(xi) - np.min(xi) + np.finfo(float).eps) + h
                    for j in range(dim):
                        A[i, j] = G[j] + np.random.randn() * mu * xi[j]
                        B[i, j] = G[j] + np.random.randn() * mu * xi[j]
                else:
                    # Reflection operation
                    for j in range(dim):
                        sA = (3 * np.random.rand() - 1.5) * (A[i, j] / (B[i, j] + np.finfo(float).eps))
                        sB = (3 * np.random.rand() - 1.5) * (B[i, j] / (A[i, j] + np.finfo(float).eps))
                        z = np.random.randint(dim)
                        k = np.random.randint(dim)
                        delta = 0.5 * (A[i, z] / (ub[z] - lb[z]) + B[i, k] / (ub[k] - lb[k]))
                        A[i, j] = G[j] + sA * mu * delta
                        B[i, j] = G[j] + sB * mu * delta
            else:
                # Reflection operation
                for j in range(dim):
                    sA = (3 * np.random.rand() - 1.5) * (A[i, j] / (B[i, j] + np.finfo(float).eps))
                    sB = (3 * np.random.rand() - 1.5) * (B[i, j] / (A[i, j] + np.finfo(float).eps))
                    z = np.random.randint(dim)
                    k = np.random.randint(dim)
                    delta = 0.5 * (A[i, z] / (ub[z] - lb[z]) + B[i, k] / (ub[k] - lb[k]))
                    A[i, j] = G[j] + sA * mu * delta
                    B[i, j] = G[j] + sB * mu * delta

            FE += 1
            if FE > MaxFEs:
                break

            A[i], H_A[i], G, H_G, CV = Update_A_ordinary(A[i], CV, FE, G, H_G, ub, lb, fun)

            FE += 1
            if FE > MaxFEs:
                break

            B[i], H_B[i], G, H_G, CV = Update_B_ordinary(B[i], CV, FE, G, H_G, ub, lb, fun)

        X = np.concatenate((A, B), axis=0)
        H = np.concatenate((H_A, H_B))

    return G, H_G, CV

# Eq. (6)
def Gap_P(f1, f2):
    p = (0.5 + np.random.rand(len(f1))) * (f1 - f2) ** 2
    p = p / (np.max(p) + np.min(p) + np.finfo(float).eps)
    return p

# Eq. (19)
def Update_A_mod(Ax, CV, FE, G, hG, ub, lb, fun):
    AubE = Ax > ub
    AlbE = Ax < lb

    Ax[AubE] = np.mod(Ax[AubE], ub[AubE] + np.finfo(float).eps) / (ub[AubE] + np.finfo(float).eps) * (ub[AubE] - lb[AubE]) + lb[AubE]
    Ax[AlbE] = np.mod(Ax[AlbE], lb[AlbE] + np.finfo(float).eps) / (lb[AlbE] + np.finfo(float).eps) * (ub[AlbE] - lb[AlbE]) + lb[AlbE]

    Ah = fun(Ax)

    if hG > Ah:
        hG = Ah
        G = Ax

    CV[FE] = hG
    return Ax, Ah, G, hG, CV

def Update_B_mod(Bx, CV, FE, G, hG, ub, lb, fun):
    BubE = Bx > ub
    BlbE = Bx < lb

    Bx[BubE] = np.mod(Bx[BubE], ub[BubE] + np.finfo(float).eps) / (ub[BubE] + np.finfo(float).eps) * (ub[BubE] - lb[BubE]) + lb[BubE]
    Bx[BlbE] = np.mod(Bx[BlbE], lb[BlbE] + np.finfo(float).eps) / (lb[BlbE] + np.finfo(float).eps) * (ub[BlbE] - lb[BlbE]) + lb[BlbE]

    Bh = fun(Bx)

    if hG > Bh:
        hG = Bh
        G = Bx

    CV[FE] = hG
    return Bx, Bh, G, hG, CV

# Eq. (19)
def Update_A_ordinary(Ax, CV, FE, G, hG, ub, lb, fun):
    AubE = np.squeeze(Ax > ub)
    AlbE = np.squeeze(Ax < lb)

    if np.any(AubE):
        Ax[AubE] = ub[AubE]
    if np.any(AlbE):
        Ax[AlbE] = lb[AlbE]

    Ah = fun(Ax)

    if hG > Ah:
        hG = Ah
        G = Ax.copy()

    CV[FE - 1] = hG

    return Ax, Ah, G, hG, CV


# Eq. (20)
def Update_B_ordinary(Bx, CV, FE, G, hG, ub, lb, fun):
    BubE = np.squeeze(Bx > ub)
    BlbE = np.squeeze(Bx < lb)

    if np.any(BubE):
        Bx[BubE] = ub[BubE]
    if np.any(BlbE):
        Bx[BlbE] = lb[BlbE]

    Bh = fun(Bx)

    if hG > Bh:
        hG = Bh
        G = Bx.copy()

    CV[FE - 1] = hG

    return Bx, Bh, G, hG, CV


def objective_function(params):
    # 参数范围
    lb = np.array([1e-6, 0.1])  # 变量下界
    ub = np.array([1e-4, 0.4])  # 变量上界

    if np.any(params < lb) or np.any(params > ub):
        print("参数越界，跳过这次训练")
        return np.inf  # 返回一个很大的值作为惩罚

    return evaluate_model(params)


# 定义LEA需要的参数
main_feature_dim = 1  # 主要输入的特征维度，即AQI
aux_feature_dim = X_aux_train.shape[2]  # 辅助输入的特征维度
r = 24
dim = 2  # 变量维度（学习率和Dropout两个超参数）
lb = np.array([1e-6, 0.1])  # 变量下界
ub = np.array([1e-4, 0.4])  # 变量上界
N = 120  # 种群大小
MaxFEs = 200  # 最大函数评估次数

# 调用LEA函数进行优化
best_solution, best_fitness, convergence_curve = LEA(objective_function, dim, lb, ub, N, MaxFEs)

# 打印结果
print("最优解 (学习率和Dropout率):", best_solution)
print("最优适应度 (MSE):", best_fitness)
