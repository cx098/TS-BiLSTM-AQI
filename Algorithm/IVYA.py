import numpy as np
import matplotlib.pyplot as plt

def IVY(N, Max_iteration, lb, ub, dim, fobj):
    def CostFunction(x):
        return fobj(x)

    VarMin = np.array(lb)       # Variables Lower Bound
    VarMax = np.array(ub)       # Variables Upper Bound

    # IVYA Parameters
    MaxIt = Max_iteration        # Maximum Number of Iterations
    nPop = N                     # Population Size
    VarSize = (1, dim)           # Decision Variables Matrix Size

    # Initialization

    # Empty Plant Structure
    class Plant:
        def __init__(self):
            self.Position = None
            self.Cost = None
            self.GV = None

    pop = [Plant() for _ in range(nPop)]    # Initial Population Array

    for plant in pop:
        # Initialize Position
        plant.Position = np.random.uniform(VarMin, VarMax, VarSize)
        # GV initialization
        plant.GV = plant.Position / (VarMax - VarMin)

        # Evaluation
        plant.Cost = CostFunction(plant.Position)

    # Initialize Best Cost History
    BestCosts = np.zeros(MaxIt)
    Convergence_curve = np.zeros(MaxIt)

    # Ivy Main Loop

    for it in range(MaxIt):
        # Get Best and Worst Cost Values
        Costs = [plant.Cost for plant in pop]
        BestCost = np.min(Costs)
        WorstCost = np.max(Costs)

        # Determine the best individual
        BestPlant = pop[np.argmin(Costs)]

        # Initialize new Population
        newpop = []

        for i in range(len(pop)):
            ii = i + 1
            if i == len(pop) - 1:
                ii = 0

            newsol = Plant()

            beta_1 = 1 + (np.random.rand() / 2)  # beta value in line 8 of "Algorithm 1 (Fig.2 in paper)"

            if pop[i].Cost < beta_1 * pop[0].Cost:
                # Eq.(5)-(6) with elite differential mutation
                if np.random.rand() < 0.5:
                    newsol.Position = BestPlant.Position + np.random.rand() * (pop[ii].Position - pop[i].Position)
                else:
                    newsol.Position = pop[i].Position + np.abs(np.random.randn(*VarSize)) * (pop[ii].Position - pop[i].Position) + np.random.randn(*VarSize) * pop[i].GV
            else:
                # Eq.(7)
                newsol.Position = pop[0].Position * (np.random.rand() + np.random.randn(*VarSize) * pop[i].GV)

            # Eq.(3)
            pop[i].GV = pop[i].GV * ((np.random.rand() ** 2) * np.random.randn(1, dim))

            # Ensure newsol.Position is within bounds
            newsol.Position = np.maximum(newsol.Position, VarMin)
            newsol.Position = np.minimum(newsol.Position, VarMax)
            # Eq.(8)
            newsol.GV = newsol.Position / (VarMax - VarMin)

            # Evaluate new population
            newsol.Cost = CostFunction(newsol.Position)

            newpop.append(newsol)

        # Merge Populations
        pop.extend(newpop)

        # Sort Population
        pop.sort(key=lambda plant: plant.Cost)

        # Competitive Exclusion (Delete Extra Members)
        if len(pop) > nPop:
            pop = pop[:nPop]

        # Store Best Solution Ever Found
        BestSol = pop[0]
        BestCosts[it] = BestSol.Cost
        Convergence_curve[it] = BestSol.Cost

    # Results
    Destination_fitness = pop[0].Cost
    Destination_position = pop[0].Position

    return Destination_fitness, Destination_position, Convergence_curve


# 全局计数器
function_call_counter = 0
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    print(x)
    return np.sum(x ** 2)


N = 50
Max_iteration = 3
lb = [-9, -1]
ub = [9, 1]
dim = 2

# Call the IVY function
Destination_fitness, Destination_position, Convergence_curve = IVY(N, Max_iteration, lb, ub, dim, objective_function)

# Print results
print(f'Best Position: {Destination_position}')
print(f'Best Fitness: {Destination_fitness}')

# Plot Convergence curve
plt.plot(Convergence_curve, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.title('Convergence Curve')
plt.grid(True)
plt.show()
