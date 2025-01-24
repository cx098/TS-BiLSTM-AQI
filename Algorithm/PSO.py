import numpy as np

def PSO(SearchAgents_no, Max_EFs, lb, ub, dim, fobj):
    # PSO参数
    w = 0.5  # 惯性权重
    c1 = 1.5  # 个体学习因子
    c2 = 1.5  # 群体学习因子

    # 初始化粒子位置和速度
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    Velocities = np.zeros((SearchAgents_no, dim))

    # 初始化个体和全局最优
    pBest_pos = Positions.copy()
    pBest_score = np.inf * np.ones(SearchAgents_no)
    gBest_pos = np.zeros(dim)
    gBest_score = np.inf

    cg_curve = np.zeros(Max_EFs)

    # PSO迭代
    for EFs in range(Max_EFs):
        for i in range(SearchAgents_no):
            # 计算当前粒子的适应度
            fitness = fobj(Positions[i, :])

            # 更新个体最优
            if fitness < pBest_score[i]:
                pBest_score[i] = fitness
                pBest_pos[i, :] = Positions[i, :]

            # 更新全局最优
            if fitness < gBest_score:
                gBest_score = fitness
                gBest_pos = Positions[i, :]

        # 更新粒子速度和位置
        for i in range(SearchAgents_no):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            Velocities[i, :] = (
                w * Velocities[i, :]
                + c1 * r1 * (pBest_pos[i, :] - Positions[i, :])
                + c2 * r2 * (gBest_pos - Positions[i, :])
            )

            Positions[i, :] = Positions[i, :] + Velocities[i, :]

            # 保证粒子位置在边界内
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)

        # 保存当前全局最优适应度值
        cg_curve[EFs] = gBest_score

    return gBest_score, gBest_pos, cg_curve


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
Best_score, Best_pos, cg_curve = PSO(SearchAgents_no, Max_EFs, lb, ub, dim, objective_function)

print("最优值：", Best_score)
print("最优位置：", Best_pos)
