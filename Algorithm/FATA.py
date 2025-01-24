import numpy as np
from scipy.integrate import cumulative_trapezoid


def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub)
    Positions = np.zeros((SearchAgents_no, dim))

    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions


def cloud_mutation(position, sigma):
    """Applies normal cloud mutation to a position."""
    dim = len(position)
    return position + np.random.normal(0, sigma, dim)


def FATA(fobj, lb, ub, dim, N, MaxFEs, sigma=0.1):
    worstInte = 0  # Parameters of Eq.(4)
    bestInte = np.inf  # Parameters of Eq.(4)
    noP = N
    arf = 0.2  # Eq. (15) reflectance=0.2
    gBest = np.zeros(dim)
    cg_curve = []
    gBestScore = np.inf  # Change this to -inf for maximization problems
    Flight = initialization(noP, dim, ub, lb)  # Initialize the set of random solutions
    fitness = np.full(noP, np.inf)
    it = 1
    FEs = 0
    lb = np.full(dim, lb)  # lower boundary
    ub = np.full(dim, ub)  # upper boundary

    # Main loop
    while FEs < MaxFEs:
        for i in range(Flight.shape[0]):
            Flag4ub = Flight[i, :] > ub
            Flag4lb = Flight[i, :] < lb
            Flight[i, :] = Flight[i, :] * (~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
            FEs += 1
            fitness[i] = fobj(Flight[i, :])
            # Make greedy selections
            if gBestScore > fitness[i]:
                gBestScore = fitness[i]
                gBest = Flight[i, :].copy()

        Order = np.sort(fitness)
        worstFitness = Order[N - 1]
        bestFitness = Order[0]

        # The mirage light filtering principle
        Integral = cumulative_trapezoid(Order, initial=0)
        if Integral[N - 1] > worstInte:
            worstInte = Integral[N - 1]
        if Integral[N - 1] < bestInte:
            bestInte = Integral[N - 1]
        IP = (Integral[N - 1] - worstInte) / (
                    bestInte - worstInte + np.finfo(float).eps)  # Eq.(4) population quality factor

        # Calculation Para1 and Para2
        a = np.tan(-(FEs / MaxFEs) + 1)
        b = 1 / np.tan(-(FEs / MaxFEs) + 1)

        for i in range(Flight.shape[0]):
            Para1 = a * np.random.rand(dim) - a * np.random.rand(dim)  # Parameters of Eq.(10)
            Para2 = b * np.random.rand(dim) - b * np.random.rand(dim)  # Parameters of Eq.(13)
            p = (fitness[i] - worstFitness) / (gBestScore - worstFitness + np.finfo(
                float).eps)  # Parameters of Eq.(5) individual quality factor

            if np.random.rand() > IP:
                Flight[i, :] = (ub - lb) * np.random.rand(dim) + lb
            else:
                for j in range(dim):
                    num = np.random.randint(N)
                    if np.random.rand() < p:
                        Flight[i, j] = gBest[j] + Flight[i, j] * Para1[j]  # Light refraction (first phase) Eq.(8)
                    else:
                        Flight[i, j] = Flight[num, j] + Para2[j] * Flight[
                            i, j]  # Light refraction (second phase) Eq.(11)
                        Flight[i, j] = (0.5 * (arf + 1) * (lb[j] + ub[j]) - arf * Flight[
                            i, j])  # Light total internal reflection Eq.(14)

            # Apply cloud mutation
            Flight[i, :] = cloud_mutation(Flight[i, :], sigma)

        cg_curve.append(gBestScore)
        it += 1
        bestPos = gBest.copy()

    return bestPos, gBestScore, cg_curve


# 全局计数器
function_call_counter = 0
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    return np.sum(x ** 2)


SearchAgents_no = 50
Max_iter = 150
lb = [-9, -1]
ub = [9, 1]
dim = 2

Leeches_best_pos, Leeches_best_score, Convergence_curve = FATA(objective_function, lb, ub, dim, SearchAgents_no, Max_iter)

print("Best Position:", Leeches_best_pos)
print("Best Score:", Leeches_best_score)
