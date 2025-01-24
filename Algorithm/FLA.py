import numpy as np
import time

# BEO算法函数
def BEO(fhd, dim, N, Max_iteration, lb, ub):
    ub = np.array(ub) * np.ones(dim)
    lb = np.array(lb) * np.ones(dim)
    R = N
    T = Max_iteration

    # 初始化
    WAR_curve = np.zeros(T)
    x = initialization(R, dim, ub, lb)
    Fitness1 = np.zeros(R)
    for i in range(R):
        Fitness1[i] = fhd(x[:, i])

    F_x = np.sort(Fitness1)
    index = np.argsort(Fitness1)
    A_Pos = x[:, index[0]]
    A_Score = F_x[0]

    for t in range(T):
        for j in range(R):
            # 更新策略的具体实现可以根据需求调整
            x[:, j] = update_position(x[:, j], A_Pos, lb, ub)
            Fitness1[j] = fhd(x[:, j])

        F_x = np.sort(Fitness1)
        index = np.argsort(Fitness1)
        A_Pos = x[:, index[0]]
        A_Score = F_x[0]

        WAR_curve[t] = A_Score

    runtime = time.time()
    return A_Score, A_Pos, WAR_curve, T, runtime


def initialization(N, dim, ub, lb):
    return np.random.rand(dim, N) * (ub - lb)[:, None] + lb[:, None]


def update_position(pos, best_pos, lb, ub):
    # 示例更新策略
    new_pos = pos + np.random.uniform(-1, 1, pos.shape) * (best_pos - pos)
    return np.clip(new_pos, lb, ub)




# 定义虚拟目标函数
def rastrigin(x):
    return np.sum(x ** 2)

# 参数设置
dim = 10
N = 50
Max_iteration = 10000
lb = -5.12
ub = 5.12

# 调用BEO算法
A_Score, A_Pos, WAR_curve, IT, runtime = BEO(rastrigin, dim, N, Max_iteration, lb, ub)

print(f"最佳得分: {A_Score}")
print(f"最佳位置: {A_Pos}")
print(f"迭代次数: {IT}")
print(f"运行时间: {runtime}")
