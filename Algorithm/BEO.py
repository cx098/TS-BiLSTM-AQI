import numpy as np
import time
import matplotlib.pyplot as plt

def BEO(fhd, dim, N, Max_iteration, lb, ub, *args):
    # 参数设置
    ub = np.array(ub) * np.ones(dim)
    lb = np.array(lb) * np.ones(dim)
    R = N
    T = Max_iteration
    np.random.seed(int(time.time() % 2 ** 31))

    # 初始化
    WAR_curve = np.zeros(T)
    x = initialization(R, dim, ub, lb)
    Fitness1 = np.array([fhd(x[:, i], *args) for i in range(R)])
    index = np.argsort(Fitness1)
    A_Pos = x[:, index[0]]
    A_Score = Fitness1[index[0]]
    t = 1
    L = 0
    A = A_Score
    IT = 0
    start_time = time.time()

    # 开始迭代更新
    while t < T:
        # 跟踪行为
        ub1 = ub.copy()
        lb1 = lb.copy()
        max_index = np.argmax(np.abs(A_Pos))
        lb1[max_index] = (ub[max_index] + lb[max_index]) / 2
        xnew1 = x.copy()
        xnew21 = np.zeros_like(x)
        xnew22 = np.zeros_like(x)

        for j in range(R):
            r1 = np.random.rand(dim)
            D = max(np.linalg.norm(ub - A_Pos), np.linalg.norm(A_Pos - lb))
            alpha = np.exp(-np.linalg.norm(x[:, j] - A_Pos) / D)
            r2 = np.random.rand()
            r3 = tent_map(np.random.rand(), 0.5)
            x_r = (np.exp(-0.2 * (t / T))) * (lb1 + r1 * (ub1 - lb1))
            rx = np.random.randint(R)
            x_k = x[:, rx]
            xnew21[:, j] = x_r + alpha * r2 * (r3 * A_Pos - x_k)
            xnew22[:, j] = lb + ub - xnew21[:, j]

        xnew = np.hstack((xnew1, xnew21, xnew22))
        xnew = np.clip(xnew, lb[:, np.newaxis], ub[:, np.newaxis])

        f = np.array([fhd(xnew[:, i], *args) for i in range(3 * R)])
        indexf = np.argsort(f)
        A_Pos1 = xnew[:, indexf[0]]
        A_Score1 = f[indexf[0]]

        if A_Score1 < A_Score:
            A_Score = A_Score1
            A_Pos = A_Pos1

        x = xnew[:, indexf[:R]]
        Fitness1 = f[indexf[:R]]

        # 警告机制
        x0 = np.zeros((dim, R))
        for i in range(dim):
            if A_Pos[i] >= 0.8 * ub[i] or A_Pos[i] <= 0.8 * lb[i]:
                center = (ub + lb) / 2
                distance = np.linalg.norm(A_Pos[i] - center[i])
                max_distance = np.linalg.norm(ub[i])
                min_distance = 0.8 * np.linalg.norm(ub[i])
                distance_normalized = (distance - min_distance) / (max_distance - min_distance)
                lambda_value = distance_normalized * (5 - 15) + 15
                distances = np.sqrt((x[i, :] - A_Pos[i]) ** 2)
                sorted_indices1 = np.argsort(distances)
                xnew3 = np.zeros((dim, R))
                poissrn_values = np.random.poisson(lambda_value, R)

                for j in range(R):
                    xnew3[i, j] = A_Pos[i] + (x[i, sorted_indices1[j]] - A_Pos[i]) * (1 + poissrn_values[j])

                lambda_normalized = (lambda_value - 5) / (15 - 5)
                bate = np.sin(lambda_normalized * np.pi / 2) * (0.953 - 0.89) + 0.89
                xnew3[i, :] = xnew3[i, :] + bate * (A_Pos[i] - xnew3[i, :])
                x0[i, :] = xnew3[i, :]

        # 盘旋行为
        xnew4 = np.zeros((dim, R))
        for j in range(R):
            rotation_angle = np.random.rand() * 2 * np.pi
            rotation_matrix = np.eye(dim)
            for ri in range(dim - 1):
                rotation_matrix[ri, ri] = np.cos(rotation_angle)
                rotation_matrix[ri, ri + 1] = -np.sin(rotation_angle)
                rotation_matrix[ri + 1, ri] = np.sin(rotation_angle)
                rotation_matrix[ri + 1, ri + 1] = np.cos(rotation_angle)
            xnew4[:, j] = A_Pos + rotation_matrix @ (x[:, j] - A_Pos)

        xnew4 = np.clip(xnew4, lb[:, np.newaxis], ub[:, np.newaxis])
        for i in range(R):
            f = fhd(xnew4[:, i], *args)
            if f < Fitness1[i] or f < np.mean(Fitness1):
                Fitness1[i] = f
                x[:, i] = xnew4[:, i]

        index = np.argsort(Fitness1)
        A_Pos1 = x[:, index[0]]
        A_Score1 = Fitness1[index[0]]

        if A_Score1 < A_Score:
            A_Score = A_Score1
            A_Pos = A_Pos1

        # 捕获行为
        d1 = np.sqrt(np.sum((x - np.mean(x, axis=1, keepdims=True)) ** 2, axis=0))
        md1 = (np.median(d1) + np.mean(d1)) / 2
        xnew50 = 2 * x - A_Pos[:, np.newaxis] + np.random.randn(dim, R) * md1
        d2 = np.sqrt(np.sum((xnew50 - np.mean(xnew50, axis=1, keepdims=True)) ** 2, axis=0))
        md15 = (np.median(d2) + np.mean(d2)) / 2
        md2 = 1 - md1 / md15
        xnew5 = xnew50 + md2 * (A_Pos[:, np.newaxis] - xnew50)

        xnew = xnew5.copy()
        for i in range(dim):
            if np.sum((x0[i, :]) ** 2) > 0:
                xnew[i, :] = x0[i, :]

        xnew = np.clip(xnew, lb[:, np.newaxis], ub[:, np.newaxis])
        for i in range(R):
            f = fhd(xnew[:, i], *args)
            if f < Fitness1[i]:
                Fitness1[i] = f
                x[:, i] = xnew[:, i]

        index = np.argsort(Fitness1)
        A_Pos1 = x[:, index[0]]
        A_Score1 = Fitness1[index[0]]

        if A_Score1 < A_Score:
            A_Score = A_Score1
            A_Pos = A_Pos1

        # 迁移行为
        if t >= 0.1 * T or L > 0.1 * T / 2:
            for j in range(R):
                Fitness1[j] = fhd(x[:, j], *args)
                z = 1 / (2 * np.exp(-(A_Score / (Fitness1[j] + np.finfo(float).eps))))
                s0 = -np.ones(dim) + 2 * np.random.rand(dim)
                xnew7 = A_Pos[:, np.newaxis] + z * s0[:, np.newaxis] * (
                            x[:, j][:, np.newaxis] - (0.4 + 0.6 * tent_map(np.random.rand(dim), 0.5))[:,
                                                     np.newaxis] * A_Pos[:, np.newaxis])

            xnew7 = np.clip(xnew7, lb[:, np.newaxis], ub[:, np.newaxis])
            for i in range(R):
                f = fhd(xnew7[:, i], *args)
                if f < Fitness1[i]:
                    Fitness1[i] = f
                    x[:, i] = xnew7[:, i]

            index = np.argsort(Fitness1)
            A_Pos1 = x[:, index[0]]
            A_Score1 = Fitness1[index[0]]

            if A_Score1 < A_Score:
                A_Score = A_Score1
                A_Pos = A_Pos1

        # 求爱行为
        if t >= 0.3 * T or L > 0.3 * T / 2:
            f = 1 / (1 * (1 + np.exp(-(14 * t / T - 9)))) + 0.3  # 0.3-1.3
            xnew8 = np.zeros((dim, R))
            for j in range(R):
                if j % 2 == 0:
                    xnew8[:, j] = A_Pos[:, np.newaxis] + f * (
                                x[:, j][:, np.newaxis] - (np.random.rand() * 0.4 + 0.6)[:, np.newaxis] * A_Pos[:,
                                                                                                         np.newaxis])
                else:
                    xnew8[:, j] = x[:, j][:, np.newaxis] + f * (
                                x[:, j][:, np.newaxis] - (np.random.rand() * 0.4 + 0.6)[:, np.newaxis] * A_Pos[:,
                                                                                                         np.newaxis])

            xnew8 = np.clip(xnew8, lb[:, np.newaxis], ub[:, np.newaxis])
            for i in range(R):
                f = fhd(xnew8[:, i], *args)
                if f < Fitness1[i]:
                    Fitness1[i] = f
                    x[:, i] = xnew8[:, i]

            index = np.argsort(Fitness1)
            A_Pos1 = x[:, index[0]]
            A_Score1 = Fitness1[index[0]]

            if A_Score1 < A_Score:
                A_Score = A_Score1
                A_Pos = A_Pos1

        if A_Score == A:
            L += 1
        else:
            L = 0

        A = A_Score
        WAR_curve[t] = A_Score
        IT += 1
        t += 1

    end_time = time.time()
    runtime = end_time - start_time

    # 绘制优化过程曲线
    plt.plot(WAR_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Best score obtained so far')
    plt.title('WAR Curve')
    plt.show()

    return A_Score, A_Pos, WAR_curve, IT, runtime


# 初始化函数
def initialization(R, dim, ub, lb):
    return np.random.rand(dim, R) * (ub - lb)[:, np.newaxis] + lb[:, np.newaxis]

# Tent映射函数
def tent_map(x, mu):
    return mu * x if x < 0.5 else mu * (1 - x)


def objective_function(x):
    return np.sum(x ** 2)

# 设置参数
dim = 2  # 维度
N = 20  # 种群大小
Max_iteration = 100  # 最大迭代次数
lb = [-5.12] * dim  # 下界
ub = [5.12] * dim  # 上界

# 调用BEO函数
A_Score, A_Pos, WAR_curve, IT, runtime = BEO(objective_function, dim, N, Max_iteration, lb, ub)

# 输出结果
print("最优值：", A_Score)
print("最优位置：", A_Pos)
print("运行时间：", runtime, "秒")
