import numpy as np

def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = np.size(ub)  # number of boundaries

    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions

def EGO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    Max_iter += 1
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    ParticleFitness = np.zeros(SearchAgents_no)

    Convergence_curve = np.zeros(Max_iter)
    Trajectories = np.zeros((SearchAgents_no, Max_iter))
    fitness_history = np.zeros((SearchAgents_no, Max_iter))
    position_history = np.zeros((SearchAgents_no, Max_iter, dim))

    for i in range(SearchAgents_no):
        ParticleFitness[i] = fobj(Positions[i, :])
        fitness_history[i, 0] = ParticleFitness[i]
        position_history[i, 0, :] = Positions[i, :]
        Trajectories[:, 0] = Positions[:, 0]

    sorted_indexes = np.argsort(ParticleFitness)
    Sorted_position = Positions[sorted_indexes, :]
    Grouper_pos = Sorted_position[0, :]
    Grouper_score = ParticleFitness[sorted_indexes[0]]

    random_position = np.random.randint(0, SearchAgents_no)
    Eel_pos = Positions[random_position, :]

    t = 0  # Loop counter

    while t < Max_iter - 1:
        a = 2 - 2 * (t / Max_iter)
        starvation_rate = 100 * (t / Max_iter)

        for i in range(SearchAgents_no):
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = (a - 2) * r1 + 2
            r4 = 100 * r2

            C1 = 2 * a * r1 - a
            C2 = 2 * r1
            b = a * r2

            for j in range(dim):
                rand_leader_index = np.random.randint(SearchAgents_no)
                X_rand = Positions[rand_leader_index, :]
                D_X_rand = abs(C2 * Positions[i, j] - X_rand[j])
                Positions[i, j] = X_rand[j] + C1 * D_X_rand

            new_fit = fobj(Positions[i, :])
            if new_fit < Grouper_score:
                Grouper_score = new_fit
                Grouper_pos = Positions[i, :]

            if r4 <= starvation_rate:
                Eel_pos = abs(C2 * Grouper_pos)
            else:
                random_position = np.random.randint(0, SearchAgents_no)
                Eel_pos = C2 * Positions[random_position, :]

            for j in range(dim):
                p = np.random.rand()
                distance2eel = abs(Positions[i, j] - C2 * Eel_pos[j])
                X1 = C1 * distance2eel * np.exp(b * r3) * np.sin(r3 * 2 * np.pi) + Eel_pos[j]

                distance2grouper = abs(C2 * Grouper_pos[j] - Positions[i, j])
                X2 = Grouper_pos[j] + C1 * distance2grouper

                if p < 0.5:
                    Positions[i, j] = (0.8 * X1 + 0.2 * X2) / 2
                else:
                    Positions[i, j] = (0.2 * X1 + 0.8 * X2) / 2

                    # Add Gaussian mutation
                if np.random.rand() < 0.1:  # Mutation probability
                    Positions[i, j] += np.random.normal(0, 1)

        for j in range(SearchAgents_no):
            Tp = Positions[j, :] > ub
            Tm = Positions[j, :] < lb
            Positions[j, :] = Positions[j, :] * (~(Tp + Tm)) + ub * Tp + lb * Tm

            ParticleFitness[j] = fobj(Positions[j, :])
            fitness_history[j, t + 1] = ParticleFitness[j]
            position_history[j, t + 1, :] = Positions[j, :]
            Trajectories[:, t + 1] = Positions[:, 0]

            if ParticleFitness[j] < Grouper_score:
                Grouper_pos = Positions[j, :]
                Grouper_score = ParticleFitness[j]

        t += 1
        Convergence_curve[t] = Grouper_score

    return Grouper_score, Grouper_pos, Convergence_curve, Trajectories, fitness_history, position_history


function_call_counter = 0
# 定义要优化的目标函数
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    print(x)

    # 参数范围
    lb = np.array([1e-6, 0.1])  # 变量下界
    ub = np.array([1e-4, 0.4])  # 变量上界

    # 检查参数是否在范围内
    if np.any(x < lb) or np.any(x > ub):
        print("参数越界，跳过这次训练")
        return np.inf  # 返回一个很大的值作为惩罚

    return np.sum(x ** 2)


# Parameters
SearchAgents_no = 50
Max_iter = 1
lb = np.array([1e-6, 0.1])  # 变量下界
ub = np.array([1e-4, 0.4])  # 变量上界
dim = 2

# Running the EGO
Grouper_score, Grouper_pos, Convergence_curve, Trajectories, fitness_history, position_history = EGO(
    SearchAgents_no, Max_iter, lb, ub, dim, objective_function)

print(f"Grouper Score: {Grouper_score}")
print(f"Grouper Position: {Grouper_pos}")
