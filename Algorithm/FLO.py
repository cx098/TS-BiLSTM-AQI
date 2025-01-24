import numpy as np


def flo(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    lowerbound = np.ones(dimension) * lowerbound  # Lower limit for variables
    upperbound = np.ones(dimension) * upperbound  # Upper limit for variables

    # INITIALIZATION
    X = lowerbound + np.random.rand(SearchAgents, dimension) * (upperbound - lowerbound)  # Initial population
    fit = np.array([fitness(ind) for ind in X])

    for t in range(Max_iterations):
        # Update: BEST proposed solution
        Fbest = np.min(fit)
        blocation = np.argmin(fit)

        if t == 0:
            xbest = X[blocation, :]
            fbest = Fbest
        elif Fbest < fbest:
            fbest = Fbest
            xbest = X[blocation, :]

        for i in range(SearchAgents):
            # Phase 1: Hunting strategy (exploration)
            prey_position = np.where(fit < fit[i])[0]
            if prey_position.size == 0:
                selected_prey = xbest
            else:
                if np.random.rand() < 0.5:
                    selected_prey = xbest
                else:
                    k = np.random.choice(prey_position)
                    selected_prey = X[k]

            I = round(1 + np.random.rand())
            X_new_P1 = X[i, :] + np.random.rand() * (selected_prey - I * X[i, :])
            X_new_P1 = np.maximum(X_new_P1, lowerbound)
            X_new_P1 = np.minimum(X_new_P1, upperbound)

            fit_new_P1 = fitness(X_new_P1)
            if fit_new_P1 < fit[i]:
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1

            # Phase 2: Moving up the tree (exploitation)
            X_new_P2 = X[i, :] + (1 - 2 * np.random.rand(dimension)) * ((upperbound - lowerbound) / (t + 1))
            X_new_P2 = np.maximum(X_new_P2, lowerbound / (t + 1))
            X_new_P2 = np.minimum(X_new_P2, upperbound / (t + 1))

            fit_new_P2 = fitness(X_new_P2)
            if fit_new_P2 < fit[i]:
                X[i, :] = X_new_P2
                fit[i] = fit_new_P2

        best_so_far = fbest
        average = np.mean(fit)

    Best_score = fbest
    Best_pos = xbest
    FLO_curve = best_so_far

    return Best_score, Best_pos, FLO_curve


# Example of an objective function
def sphere_function(x):
    return np.sum(x ** 2)


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
MaxFEs = 1  # 最大函数评估次数


# 调用LEA函数进行优化
best_fitness, best_solution, convergence_curve = flo(N, MaxFEs, lb, ub, dim, objective_function)

# 打印结果
print("最优解:", best_solution)
print("最优适应度:", best_fitness)
