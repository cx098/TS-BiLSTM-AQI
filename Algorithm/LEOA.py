import numpy as np
from scipy.special import gamma


def levy(dim):
    beta = 3 / 2
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, dim) * sigma
    v = np.random.randn(1, dim)
    step = u / np.abs(v) ** (1 / beta)
    return 0.01 * step


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=0))


def initialization(SearchAgents_no, dim, ub, lb):
    return np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb


def gauss_mutation(X, sigma=0.1):
    return X + np.random.normal(0, sigma, X.shape)


def lea(SearchAgents_no, Max_iteration, lb, ub, dim, fobj):
    print('Optimizing your problem')
    cg_curve = np.zeros(Max_iteration)

    if np.isscalar(ub):
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    r = (ub - lb) / 10
    Delta_max = (ub - lb) / 10

    Food_fitness = np.inf
    Food_pos = np.zeros(dim)

    Enemy_fitness = -np.inf
    Enemy_pos = np.zeros(dim)

    X = initialization(SearchAgents_no, dim, ub, lb)
    Fitness = np.zeros(SearchAgents_no)

    DeltaX = initialization(SearchAgents_no, dim, ub, lb)

    for iter in range(Max_iteration):
        r = (ub - lb) / 4 + ((ub - lb) * (iter / Max_iteration) * 2)
        w = 0.9 - iter * ((0.9 - 0.4) / Max_iteration)
        my_c = 0.1 - iter * ((0.1 - 0) / (Max_iteration / 2))
        if my_c < 0:
            my_c = 0

        s = 2 * np.random.rand() * my_c
        a = 2 * np.random.rand() * my_c
        c = 2 * np.random.rand() * my_c
        f = 2 * np.random.rand()
        e = my_c

        for i in range(SearchAgents_no):
            Fitness[i] = fobj(X[i])
            if Fitness[i] < Food_fitness:
                Food_fitness = Fitness[i]
                Food_pos = X[i]
            if Fitness[i] > Enemy_fitness and np.all(X[i] < ub) and np.all(X[i] > lb):
                Enemy_fitness = Fitness[i]
                Enemy_pos = X[i]

        for i in range(SearchAgents_no):
            index = 0
            neighbours_no = 0
            Neighbours_DeltaX = []
            Neighbours_X = []

            for j in range(SearchAgents_no):
                Dist2Enemy = euclidean_distance(X[i], X[j])
                if np.all(Dist2Enemy <= r) and np.all(Dist2Enemy != 0):
                    neighbours_no += 1
                    Neighbours_DeltaX.append(DeltaX[j])
                    Neighbours_X.append(X[j])

            S = np.zeros(dim)
            if neighbours_no > 1:
                for k in range(neighbours_no):
                    S += (Neighbours_X[k] - X[i])
                S = -S
            else:
                S = np.zeros(dim)

            if neighbours_no > 1:
                A = np.sum(Neighbours_DeltaX, axis=0) / neighbours_no
            else:
                A = DeltaX[i]

            if neighbours_no > 1:
                C_temp = np.sum(Neighbours_X, axis=0) / neighbours_no
            else:
                C_temp = X[i]

            C = C_temp - X[i]
            Dist2Food = euclidean_distance(X[i], Food_pos)
            if np.all(Dist2Food <= r):
                F = Food_pos - X[i]
            else:
                F = 0

            Dist2Enemy = euclidean_distance(X[i], Enemy_pos)
            if np.all(Dist2Enemy <= r):
                Enemy = Enemy_pos + X[i]
            else:
                Enemy = np.zeros(dim)

            for tt in range(dim):
                if X[i, tt] > ub[tt]:
                    X[i, tt] = lb[tt]
                    DeltaX[i, tt] = np.random.rand()
                if X[i, tt] < lb[tt]:
                    X[i, tt] = ub[tt]
                    DeltaX[i, tt] = np.random.rand()

            if np.any(Dist2Food > r):
                if neighbours_no > 1:
                    for j in range(dim):
                        DeltaX[i, j] = (w * DeltaX[i, j] + np.random.rand() * A[j] +
                                        np.random.rand() * C[j] + np.random.rand() * S[j])
                        DeltaX[i, j] = np.clip(DeltaX[i, j], -Delta_max[j], Delta_max[j])
                        X[i, j] += DeltaX[i, j]
                else:
                    X[i] += levy(dim)[0] * X[i]
                    DeltaX[i] = 0
            else:
                for j in range(dim):
                    DeltaX[i, j] = (a * A[j] + c * C[j] + s * S[j] + f * F[j] + e * Enemy[j] +
                                    w * DeltaX[i, j])
                    DeltaX[i, j] = np.clip(DeltaX[i, j], -Delta_max[j], Delta_max[j])
                    X[i, j] += DeltaX[i, j]

            X[i] = np.clip(X[i], lb, ub)
            X[i] = gauss_mutation(X[i])  # Apply Gaussian mutation

        Best_score = Food_fitness
        Best_pos = Food_pos
        cg_curve[iter] = Best_score

    return Best_score, Best_pos, cg_curve


# 全局计数器
function_call_counter = 0
# 定义要优化的目标函数
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    return np.sum(x**2)


dim = 2  # 变量维度
lb = np.array([-9, -1])  # 两个变量的下界
ub = np.array([9, 1])    # 两个变量的上界
N = 50  # 种群大小
MaxFEs = 4  # 最大函数评估次数


# 调用LEA函数进行优化
best_fitness, best_solution, convergence_curve = lea(N, MaxFEs, lb, ub, dim, objective_function)

# 打印结果
print("最优解:", best_solution)
print("最优适应度:", best_fitness)

