import numpy as np

def GA(SearchAgents_no, Max_EFs, lb, ub, dim, fobj):
    # 初始化种群
    Population = initialization(SearchAgents_no, dim, ub, lb)

    # 初始化适应度
    fitness_scores = np.zeros(SearchAgents_no)
    for i in range(SearchAgents_no):
        fitness_scores[i] = fobj(Population[i, :])

    # 记录全局最优
    gBest_pos = np.zeros(dim)
    gBest_score = np.inf

    # 遗传算法迭代
    cg_curve = np.zeros(Max_EFs)
    for EFs in range(Max_EFs):
        # 选择父代
        parents = selection(Population, fitness_scores)

        # 生成子代 (交叉 + 变异)
        offspring = crossover(parents, dim)
        offspring = mutation(offspring, lb, ub)

        # 评估子代适应度
        for i in range(SearchAgents_no):
            fitness = fobj(offspring[i, :])
            if fitness < fitness_scores[i]:  # 替换适应度较差的个体
                Population[i, :] = offspring[i, :]
                fitness_scores[i] = fitness

        # 更新全局最优
        min_fitness_idx = np.argmin(fitness_scores)
        if fitness_scores[min_fitness_idx] < gBest_score:
            gBest_score = fitness_scores[min_fitness_idx]
            gBest_pos = Population[min_fitness_idx, :]

        # 保存当前最优适应度值
        cg_curve[EFs] = gBest_score

    return gBest_score, gBest_pos, cg_curve


def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub)
    Population = np.zeros((SearchAgents_no, dim))

    if Boundary_no == 1:
        Population = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    elif Boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Population[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Population


def selection(Population, fitness_scores):
    # 轮盘赌选择
    fitness_inverse = 1.0 / (fitness_scores + 1e-10)  # 避免除零
    selection_probs = fitness_inverse / np.sum(fitness_inverse)
    selected_indices = np.random.choice(range(Population.shape[0]), size=Population.shape[0], p=selection_probs)
    return Population[selected_indices, :]


def crossover(parents, dim):
    offspring = np.zeros(parents.shape)

    if dim == 2:
        # 如果维度是2，使用单点交叉
        cross_point = 1  # 只能在第1个基因位交叉
    else:
        # 如果维度大于2，使用随机交叉点
        cross_point = np.random.randint(1, dim - 1)  # 随机选择交叉点

    # 交叉操作
    for i in range(0, parents.shape[0], 2):
        parent1 = parents[i, :]
        parent2 = parents[i + 1, :]

        # 交叉
        offspring[i, :cross_point] = parent1[:cross_point]
        offspring[i, cross_point:] = parent2[cross_point:]
        offspring[i + 1, :cross_point] = parent2[:cross_point]
        offspring[i + 1, cross_point:] = parent1[cross_point:]

    return offspring


def mutation(offspring, lb, ub, mutation_rate=0.05):
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[i, j] = np.random.uniform(lb[j], ub[j])
    return offspring


# 全局计数器
function_call_counter = 0
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    return np.sum(x ** 2)


SearchAgents_no = 80
Max_EFs = 10
lb = [-9, -1]
ub = [9, 1]
dim = 2
Best_score, Best_pos, cg_curve = GA(SearchAgents_no, Max_EFs, lb, ub, dim, objective_function)

print("最优值：", Best_score)
print("最优位置：", Best_pos)
