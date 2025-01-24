import numpy as np

def CFOA(SearchAgents_no, Max_EFs, lb, ub, dim, fobj):
    # 随机差分变异缩放因子
    F = 0.8
    # 初始化参数
    Fisher = initialization(SearchAgents_no, dim, ub, lb)
    newFisher = Fisher.copy()
    EFs = 0
    Best_score = np.inf
    Best_pos = np.zeros(dim)
    cg_curve = np.zeros(Max_EFs)
    fit = np.inf * np.ones(SearchAgents_no)
    newfit = fit.copy()

    # 主循环
    while EFs < Max_EFs:
        for i in range(SearchAgents_no):
            Flag4ub = newFisher[i, :] > ub
            Flag4lb = newFisher[i, :] < lb
            newFisher[i, :] = (newFisher[i, :] * ~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
            newfit[i] = fobj(newFisher[i, :])
            if newfit[i] <= fit[i]:
                fit[i] = newfit[i]
                Fisher[i, :] = newFisher[i, :]
            if newfit[i] <= Best_score:
                Best_pos = Fisher[i, :]
                Best_score = fit[i]
            EFs += 1
            cg_curve[EFs - 1] = Best_score
            if EFs >= Max_EFs:
                break

        try:
            if EFs < Max_EFs / 2:
                alpha = ((1 - 3 * EFs / (2 * Max_EFs)) ** (3 * EFs / (2 * Max_EFs)))
                p = np.random.rand()
                pos = np.random.permutation(SearchAgents_no)
                i = 0
                while i < SearchAgents_no:
                    per = np.random.randint(3, 5)  # 随机确定组大小
                    if p < alpha or i + per - 1 >= SearchAgents_no:
                        r = np.random.randint(SearchAgents_no)
                        while r == i:
                            r = np.random.randint(SearchAgents_no)
                        Exp = ((fit[pos[i]] - fit[pos[r]]) / (max(fit) - Best_score))
                        rs = np.random.rand(dim) * 2 - 1
                        rs = np.linalg.norm(Fisher[r, :] - Fisher[i, :]) * np.random.rand() * (1 - EFs / Max_EFs) * rs / np.linalg.norm(rs)
                        newFisher[pos[i], :] = Fisher[pos[i], :] + (Fisher[pos[r], :] - Fisher[pos[i], :]) * Exp + (abs(Exp) ** 0.5) * rs
                        i += 1
                    else:
                        aim = np.sum(fit[pos[i:i + per]] / np.sum(fit[pos[i:i + per]]) * Fisher[pos[i:i + per], :], axis=0)
                        newFisher[pos[i:i + per], :] = Fisher[pos[i:i + per], :] + np.random.rand(per, 1) * (aim - Fisher[pos[i:i + per], :]) + (1 - 2 * EFs / Max_EFs) * (np.random.rand(per, dim) * 2 - 1)
                        i += per
            else:
                sigma = (2 * (1 - EFs / Max_EFs) / ((1 - EFs / Max_EFs) ** 2 + 1)) ** 0.5
                for i in range(SearchAgents_no):
                    W = abs(Best_pos - np.mean(Fisher, axis=0)) * (np.random.randint(1, 4) / 3) * sigma
                    newFisher[i, :] = Best_pos + np.random.normal(0, W, dim)

                    # 随机差分变异策略
                    indices = np.random.choice(SearchAgents_no, 3, replace=False)
                    mutant_vector = Fisher[indices[0], :] + F * (Fisher[indices[1], :] - Fisher[indices[2], :])
                    mutant_vector = np.clip(mutant_vector, lb, ub)  # 保证变异后仍在边界内

                    # 用变异后的个体代替当前个体
                    if fobj(mutant_vector) < fobj(Fisher[i, :]):
                        newFisher[i, :] = mutant_vector

        except ValueError as e:
            print(f"发生错误：{e}，跳过此轮计算。")
            continue

    return Best_score, Best_pos, cg_curve


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
Max_EFs = 200
lb = [-9, -1]
ub = [9, 1]
dim = 2
Best_score, Best_pos, cg_curve = CFOA(SearchAgents_no, Max_EFs, lb, ub, dim, objective_function)

print("最优值：", Best_score)
print("最优位置：", Best_pos)
