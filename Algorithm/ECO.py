import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def ECO(N, Max_iter, lb, ub, dim, fobj):
    H = 0.5  # Learning habit boundary
    G1 = 0.2  # Proportion of primary school (Stage 1)
    G2 = 0.1  # Proportion of middle school (Stage 2)

    G1Number = round(N * G1)  # Number of G1
    G2Number = round(N * G2)  # Number of G2

    if np.isscalar(ub):
        ub = np.full(dim, ub)
        lb = np.full(dim, lb)

    # Initialization
    X0 = initializationLogistic(N, dim, ub, lb)
    X = X0.copy()

    # Compute initial fitness values
    fitness = np.array([fobj(X[i, :]) for i in range(N)])
    sorted_indices = np.argsort(fitness)
    fitness = fitness[sorted_indices]
    GBestF = fitness[0]  # Global best fitness value
    AveF = np.mean(fitness)
    X = X[sorted_indices, :]

    curve = np.zeros(Max_iter)
    avg_fitness_curve = np.zeros(Max_iter)
    GBestX = X[0, :].copy()  # Global best position
    GWorstX = X[-1, :].copy()  # Global worst position
    X_new = X.copy()
    search_history = np.zeros((N, Max_iter, dim))
    fitness_history = np.zeros((N, Max_iter))

    # Start search
    for i in range(Max_iter):
        avg_fitness_curve[i] = AveF
        R1 = np.random.rand()
        R2 = np.random.rand()
        P = 4 * np.random.randn() * (1 - i / Max_iter)
        E = (np.pi * i) / (P * Max_iter)
        w = 0.1 * np.log(2 - (i / Max_iter))

        for j in range(N):
            # Stage 1: Primary school competition
            if i % 3 == 0:
                if j < G1Number:  # Primary school Site Selection Strategies
                    X_new[j, :] = X[j, :] + w * (np.mean(X[j, :]) - X[j, :]) * Levy(dim)
                else:  # Competitive Strategies for Students (Stage 1)
                    X_new[j, :] = X[j, :] + w * (close(X[j, :], 1, X, G1Number) - X[j, :]) * np.random.randn()
            # Stage 2: Middle school competition
            elif i % 3 == 1:
                if j < G2Number:  # Middle school Site Selection Strategies
                    X_new[j, :] = X[j, :] + (GBestX - np.mean(X, axis=0)) * np.exp(i / Max_iter - 1) * Levy(dim)
                else:  # Competitive Strategies for Students (Stage 2)
                    if R1 < H:
                        X_new[j, :] = X[j, :] - w * close(X[j, :], 2, X, G2Number) - P * (
                                    E * w * close(X[j, :], 2, X, G2Number) - X[j, :])
                    else:
                        X_new[j, :] = X[j, :] - w * close(X[j, :], 2, X, G2Number) - P * (
                                    w * close(X[j, :], 2, X, G2Number) - X[j, :])
            # Stage 3: High school competition
            else:
                if j < G2Number:  # High school Site Selection Strategies
                    X_new[j, :] = X[j, :] + (GBestX - X[j, :]) * np.random.randn() - (
                                GBestX - X[j, :]) * np.random.randn()
                else:  # Competitive Strategies for Students (Stage 3)
                    if R2 < H:
                        X_new[j, :] = GBestX - P * (E * GBestX - X[j, :])
                    else:
                        X_new[j, :] = GBestX - P * (GBestX - X[j, :])

            # Boundary control
            X_new[j, :] = np.clip(X_new[j, :], lb, ub)

            # Finding the best location so far
            fitness_new = fobj(X_new[j, :])
            if fitness_new > fitness[j]:
                fitness_new = fitness[j]
                X_new[j, :] = X[j, :].copy()

            if fitness_new < GBestF:
                GBestF = fitness_new
                GBestX = X_new[j, :].copy()

        X = X_new.copy()
        fitness = np.array([fobj(X[k, :]) for k in range(N)])
        curve[i] = GBestF
        Best_pos = GBestX.copy()
        Best_score = curve[i]
        search_history[:, i, :] = X
        fitness_history[:, i] = fitness

        # Sorting and updating
        sorted_indices = np.argsort(fitness)
        fitness = fitness[sorted_indices]
        X = X[sorted_indices, :]

    return avg_fitness_curve, Best_pos, Best_score, curve, search_history, fitness_history


def Levy(d):
    beta = 1.5
    sigma = (scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step


def close(t, G, X, GNumber):
    m = X[0, :].copy()
    if G == 1:
        for s in range(GNumber):
            school = X[s, :].copy()
            if np.sum(np.abs(m - t)) > np.sum(np.abs(school - t)):
                m = school
    else:
        for s in range(GNumber):
            school = X[s, :].copy()
            if np.sum(np.abs(m - t)) > np.sum(np.abs(school - t)):
                m = school
    return m


def initializationLogistic(pop, dim, ub, lb):
    Boundary_no = len(ub)
    Positions = np.zeros((pop, dim))

    for i in range(pop):
        for j in range(dim):
            x0 = np.random.rand()
            a = 4
            x = a * x0 * (1 - x0)
            if Boundary_no == 1:
                Positions[i, j] = (ub - lb) * x + lb
                Positions[i, j] = np.clip(Positions[i, j], lb, ub)
            else:
                Positions[i, j] = (ub[j] - lb[j]) * x + lb[j]
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
            x0 = x
    return Positions

# 全局计数器
function_call_counter = 0
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    return np.sum(x ** 2)


# 示例调用
N = 50
Max_iter = 4
lb = [-9, -1]
ub = [9, 1]
dim = 2

avg_fitness_curve, Best_pos, Best_score, curve, search_history, fitness_history = ECO(N, Max_iter, lb, ub, dim,
                                                                                      objective_function)
print("最优值：", Best_score)
print("最优位置：", Best_pos)

# 绘制适应度曲线
plt.plot(curve, label='Best Fitness Curve')
plt.plot(avg_fitness_curve, label='Average Fitness Curve')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()
plt.show()