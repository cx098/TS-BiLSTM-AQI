import numpy as np
from scipy.stats import t

def FVIM(pNum, maxIter, lb, ub, dim, costFunc):
    # 初始化解
    fstPos = np.zeros(dim)
    fstVal = np.inf
    sndPos = np.zeros(dim)
    sndVal = np.inf
    thrdPos = np.zeros(dim)
    thrdVal = np.inf
    frthPos = np.zeros(dim)
    frthVal = np.inf
    bstVal = np.inf
    alpha = 1.5

    fitHist = np.zeros((pNum, maxIter))
    posHist = np.zeros((pNum, maxIter, dim))
    traj = np.zeros((pNum, maxIter))

    # 初始化粒子位置
    meanPos = initialization(pNum, dim, ub, lb)
    convCurve = np.zeros(maxIter)
    iter = 0

    # 主循环
    while iter < maxIter:
        iter += 1
        for i in range(meanPos.shape[0]):
            # 处理边界条件
            ubFlag = meanPos[i, :] > ub
            lbFlag = meanPos[i, :] < lb
            meanPos[i, :] = meanPos[i, :] * ~(ubFlag + lbFlag) + ub * ubFlag + lb * lbFlag

            # 评估适应度
            fit = costFunc(meanPos[i, :])
            fstPos, sndPos, thrdPos, frthPos, fstVal, sndVal, thrdVal, frthVal = updateSolutionHierarchy(
                fit, meanPos[i, :], fstPos, sndPos, thrdPos, frthPos, fstVal, sndVal, thrdVal, frthVal)

            # 存储适应度和位置
            fitHist[i, iter - 1] = fit
            posHist[i, iter - 1, :] = meanPos[i, :]
            traj[:, iter - 1] = meanPos[:, 0]

        # 更新位置
        alpha -= 0.004
        for i in range(pNum):
            for j in range(dim):
                r3 = np.random.rand()
                X1 = updatePosition(fstPos[j], meanPos[i, j], alpha, r3)
                r3 = np.random.rand()
                X2 = updatePosition(sndPos[j], meanPos[i, j], alpha, r3)
                r3 = np.random.rand()
                X3 = updatePosition(thrdPos[j], meanPos[i, j], alpha, r3)
                r3 = np.random.rand()
                X4 = updatePosition(frthPos[j], meanPos[i, j], alpha, r3)
                meanPos[i, j] = (X1 + X2 + X3 + X4) / 4

            # 加入t分布扰动变异
            meanPos[i, :] += t_dist_mutation(dim)

        bstVal = updateBestValue(bstVal, fstVal, sndVal, thrdVal, frthVal)
        convCurve[iter - 1] = bstVal

    return bstVal, fstPos, sndPos, thrdPos, frthPos, convCurve, traj, fitHist, posHist

def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub) if isinstance(ub, (list, np.ndarray)) else 1
    Positions = np.zeros((SearchAgents_no, dim))

    # 如果所有变量的边界相等，并且用户为 ub 和 lb 输入单个数字
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    # 如果每个变量有不同的 lb 和 ub
    else:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions

def updateSolutionHierarchy(fitness, position, fstPos, sndPos, thrdPos, frthPos, fstVal, sndVal, thrdVal, frthVal):
    if fitness < fstVal:
        frthVal, thrdVal, sndVal, fstVal = thrdVal, sndVal, fstVal, fitness
        frthPos, thrdPos, sndPos, fstPos = thrdPos, sndPos, fstPos, position
    elif fitness < sndVal:
        frthVal, thrdVal, sndVal = thrdVal, sndVal, fitness
        frthPos, thrdPos, sndPos = thrdPos, sndPos, position
    elif fitness < thrdVal:
        frthVal, thrdVal = thrdVal, fitness
        frthPos, thrdPos = thrdPos, position
    elif fitness < frthVal:
        frthVal = fitness
        frthPos = position

    return fstPos, sndPos, thrdPos, frthPos, fstVal, sndVal, thrdVal, frthVal

def updatePosition(best, current, a, r3):
    if r3 < 0.5:
        X = best + (a * 2 * np.random.rand() - a) * abs(np.random.rand() * best - current)
    else:
        X = best - (a * 2 * np.random.rand() - a) * abs(np.random.rand() * best - current)
    return X

def updateBestValue(bstVal, fstVal, sndVal, thrdVal, frthVal):
    return min([bstVal, fstVal, sndVal, thrdVal, frthVal])

def t_dist_mutation(dim, df=1.5):
    return t.rvs(df, size=dim)

function_call_counter = 0
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")
    return np.sum(x ** 2)

# 示例调用
pNum = 50
maxIter = 4
lb = [-9, -1]
ub = [9, 1]
dim = 2
bstVal, fstPos, sndPos, thrdPos, frthPos, convCurve, traj, fitHist, posHist = FVIM(pNum, maxIter, lb, ub, dim, objective_function)

print("最优值：", bstVal)
print("最优位置：", fstPos)
