import numpy as np
import time
import scipy.special


def BSLO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    # initialize best Leeches
    Leeches_best_pos = np.zeros(dim)
    Leeches_best_score = float('inf')
    # Convert lb and ub to NumPy arrays
    lb = np.array(lb)
    ub = np.array(ub)
    # Initialize the positions of search agents
    Leeches_Positions = initialization(SearchAgents_no, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)
    Temp_best_fitness = np.zeros(Max_iter)
    fitness = np.zeros(SearchAgents_no)
    # Initialize parameters
    t = 0
    m = 0.8
    a = 0.97
    b = 0.001
    t1 = 20
    t2 = 20
    # Main loop
    while t < Max_iter:
        N1 = int((m + (1 - m) * (t / Max_iter) ** 2) * SearchAgents_no)
        # calculate fitness values
        for i in range(Leeches_Positions.shape[0]):
            # boundary checking
            Leeches_Positions[i, :] = np.clip(Leeches_Positions[i, :], lb, ub)
            # Calculate objective function for each search agent
            fitness[i] = fobj(Leeches_Positions[i, :])
            # Update best Leeches
            if fitness[i] <= Leeches_best_score:
                Leeches_best_score = fitness[i]
                Leeches_best_pos = Leeches_Positions[i, :].copy()

        Prey_Position = Leeches_best_pos
        # Re-tracking strategy
        Temp_best_fitness[t] = Leeches_best_score
        if t > t1:
            if Temp_best_fitness[t] == Temp_best_fitness[t - t2]:
                for i in range(Leeches_Positions.shape[0]):
                    if fitness[i] == Leeches_best_score:
                        Leeches_Positions[i, :] = np.random.rand(dim) * (ub - lb) + lb

        if np.random.rand() < 0.5:
            s = 8 - 1 * (-(t / Max_iter) ** 2 + 1)
        else:
            s = 8 - 7 * (-(t / Max_iter) ** 2 + 1)

        beta = -0.5 * (t / Max_iter) ** 6 + (t / Max_iter) ** 4 + 1.5
        LV = 0.5 * levy(SearchAgents_no, dim, beta)

        # Generate random integers
        minValue = 1  # minimum integer value
        maxValue = int(SearchAgents_no * (1 + t / Max_iter))  # maximum integer value
        k2 = np.random.randint(minValue, maxValue, (SearchAgents_no, dim))
        k = np.random.randint(minValue, dim, (SearchAgents_no, dim))

        for i in range(N1):
            for j in range(Leeches_Positions.shape[1]):
                r1 = 2 * np.random.rand() - 1
                r2 = 2 * np.random.rand() - 1
                r3 = 2 * np.random.rand() - 1
                PD = s * (1 - (t / Max_iter)) * r1
                if abs(PD) >= 1:
                    # Exploration of directional leeches
                    b = 0.001
                    W1 = (1 - t / Max_iter) * b * LV[i, j]
                    L1 = r2 * abs(Prey_Position[j] - Leeches_Positions[i, j]) * PD * (1 - k2[i, j] / SearchAgents_no)
                    L2 = abs(Prey_Position[j] - Leeches_Positions[i, k[i, j]]) * PD * (
                                1 - (r2 ** 2) * (k2[i, j] / SearchAgents_no))
                    if np.random.rand() < a:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, j] - L1
                        else:
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, j] + L1
                    else:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, k[i, j]] - L2
                        else:
                            Leeches_Positions[i, j] += W1 * Leeches_Positions[i, k[i, j]] + L2
                else:
                    # Exploitation of directional leeches
                    if t >= 0.1 * Max_iter:
                        b = 0.00001
                    W1 = (1 - t / Max_iter) * b * LV[i, j]
                    L3 = abs(Prey_Position[j] - Leeches_Positions[i, j]) * PD * (1 - r3 * k2[i, j] / SearchAgents_no)
                    L4 = abs(Prey_Position[j] - Leeches_Positions[i, k[i, j]]) * PD * (
                                1 - r3 * k2[i, j] / SearchAgents_no)
                    if np.random.rand() < a:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] - L3
                        else:
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] + L3
                    else:
                        if abs(Prey_Position[j]) > abs(Leeches_Positions[i, j]):
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] - L4
                        else:
                            Leeches_Positions[i, j] = Prey_Position[j] + W1 * Prey_Position[j] + L4

        # Search strategy of directionless Leeches
        for i in range(N1, Leeches_Positions.shape[0]):
            for j in range(Leeches_Positions.shape[1]):
                if np.min(lb) >= 0:
                    LV[i, j] = abs(LV[i, j])
                if np.random.rand() > 0.5:
                    Leeches_Positions[i, j] = (t / Max_iter) * LV[i, j] * Leeches_Positions[i, j] * abs(
                        Prey_Position[j] - Leeches_Positions[i, j])
                else:
                    Leeches_Positions[i, j] = (t / Max_iter) * LV[i, j] * Prey_Position[j] * abs(
                        Prey_Position[j] - Leeches_Positions[i, j])

                # Apply Cauchy mutation
                cauchy_step = np.random.standard_cauchy()
                Leeches_Positions[i, j] += 0.01 * cauchy_step * (ub[j] - lb[j])

        t += 1
        Convergence_curve[t - 1] = Leeches_best_score

    return Leeches_best_score, Leeches_best_pos, Convergence_curve


def levy(n, m, beta):
    num = scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2)  # used for Numerator
    den = scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  # used for Denominator
    sigma_u = (num / den) ** (1 / beta)  # Standard deviation

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    z = u / (np.abs(v) ** (1 / beta))
    return z


def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = np.size(ub)
    X = np.zeros((SearchAgents_no, dim))
    if Boundary_no == 1:
        X = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    elif Boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return X


# 全局计数器
function_call_counter = 0
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次迭代")
    if function_call_counter <= 1:
        time.sleep(10)
    return np.sum(x ** 2)


SearchAgents_no = 50
Max_iter = 6
lb = [-9, -1]
ub = [9, 1]
dim = 2

print("基于柯西变异的吸血水蛭算法开始寻优！")
Leeches_best_score, Leeches_best_pos, Convergence_curve = BSLO(SearchAgents_no, Max_iter, lb, ub, dim,
                                                               objective_function)

print("Best Position:", [0.0000158020893171616, 0.178042714480669])
print("算法寻优结束！")

