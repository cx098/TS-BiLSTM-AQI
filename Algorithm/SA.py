import numpy as np


def SA(Max_EFs, lb, ub, dim, fobj, initial_temp=1000, cooling_rate=0.9, min_temp=1e-10):
    # 初始化解
    current_solution = initialization(1, dim, ub, lb)[0, :]
    current_fitness = fobj(current_solution)

    # 初始化最优解
    best_solution = current_solution.copy()
    best_fitness = current_fitness

    # 记录温度
    temp = initial_temp
    cg_curve = np.zeros(Max_EFs)

    # 模拟退火迭代
    for EFs in range(Max_EFs):
        # 生成邻域解
        new_solution = neighbor_solution(current_solution, lb, ub)
        new_fitness = fobj(new_solution)

        # 计算能量差
        delta_f = new_fitness - current_fitness

        # 如果新解更优，或根据概率接受较差解
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / temp):
            current_solution = new_solution
            current_fitness = new_fitness

        # 更新全局最优解
        if current_fitness < best_fitness:
            best_solution = current_solution.copy()
            best_fitness = current_fitness

        # 记录当前最优适应度
        cg_curve[EFs] = best_fitness

        # 更新温度
        temp = temp * cooling_rate
        if temp < min_temp:
            break

    return best_fitness, best_solution, cg_curve


def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub)
    Positions = np.zeros((SearchAgents_no, dim))

    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    elif Boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions


def neighbor_solution(current_solution, lb, ub):
    # 在当前位置附近生成邻域解
    dim = len(current_solution)
    new_solution = current_solution + np.random.uniform(-1, 1, dim)

    # 保证邻域解在边界内
    new_solution = np.clip(new_solution, lb, ub)
    return new_solution


# 全局计数器
function_call_counter = 0


def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    return np.sum(x ** 2)


Max_EFs = 100
lb = [-9, -1]
ub = [9, 1]
dim = 2
Best_score, Best_pos, cg_curve = SA(Max_EFs, lb, ub, dim, objective_function)

print("最优值：", Best_score)
print("最优位置：", Best_pos)


