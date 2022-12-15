import pandas as pd                                         # 用于导入数据
import pylab as plt                                         # 用于绘图
import numpy as np                                          # 用于矩阵、向量运算
plt.rcParams['font.sans-serif'] = ['SimHei']               # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False                 # 用来正常显示负号


def gaussian_kernel(x, i, sigma):                           # 高斯核函数
    gauss = np.exp(- (x-x[i]) ** 2 / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)
    return gauss


def kernel_smoothing(x, y, h):                              # 核平滑（相当于0阶局部多项式核回归）
    n = len(x)
    y_pred = np.zeros(n)
    for i in range(n):
        w = gaussian_kernel(x, i, h)
        y_pred[i] = np.sum(y * w) / np.sum(w)
    return y_pred


def weighted_least_squares(y, x, p, W):                     # 基于局部多项式回归模型的加权最小二乘法
    n = len(x)
    design_matrix = np.asmatrix(np.ones(n)).T
    for i in range(p):
        arr = np.asmatrix(np.power(x, i+1)).T
        design_matrix = np.hstack((design_matrix, arr))
    coef = np.linalg.pinv(design_matrix.T * W * design_matrix) * design_matrix.T * W * np.asmatrix(y).T   # 最小二乘求参
    return np.array(coef)


def local_kernel_regression(y, x, p, width):                # 局部线性回归和局部2阶多项式回归
    y_pred = np.zeros(len(x))
    for i in range(len(x)):
        local = []
        for j in range(len(x)):
            if np.abs(x[j]-x[i]) <= width:                             # 取邻域
                local.append(j)
        local_y = y[local]
        local_x = x[local]
        weight = gaussian_kernel(x, i, width)                          # 利用高斯核函数求权重
        local_weight = weight[local]
        W = np.diag(local_weight)
        coef = weighted_least_squares(local_y, local_x, p, W)          # 计算系数
        if p == 1:                                                     # 计算回归拟合值
            y_pred[i] = coef[1] * x[i] + coef[0]
        if p == 2:
            y_pred[i] = coef[2] * x[i] * x[i] + coef[1] * x[i] + coef[0]
    return y_pred


def cal_rmse(y, y_hat):                                       # 计算均方根误差RMSE
    y = np.array(y).reshape(1, len(y))
    y_hat = np.array(y_hat).reshape(1, len(y_hat))
    mse = np.mean((y - y_hat) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def cubic_spline(x, y, knots):                                      # 三次样条回归
    x = np.matrix(x).T
    y = np.matrix(y).T
    n = x.shape[0]
    design_matrix = np.matrix(np.ones(n)).T                         # 设计矩阵，初始时只有一列1
    for i in range(3):                                              # 将1-3次项加入设计矩阵
        design_matrix = np.hstack((design_matrix, np.power(x, i+1)))
    m = len(knots)
    for i in range(m):
        x_plus = np.matrix(x)
        for j in range(n):
            if x_plus[j] > knots[i]:
                x_plus[j] = x_plus[j] - knots[i]
            else:
                x_plus[j] = 0
        design_matrix = np.hstack((design_matrix, np.power(x_plus, 3)))    # 将(x-a_k)_+加入设计矩阵
    coeff = (design_matrix.T * design_matrix).I * design_matrix.T * y      # 最小二乘法计算回归参数
    y_pred = design_matrix * coeff                                         # 计算回归值y_pred
    return y_pred


def width_plot(band_width, x, y):                          # 绘制不同窗宽下不同回归模型的RMSE
    rmse_kernel = []                                       # 分别存储各回归模型的RMSE
    rmse_linear = []
    rmse_ply = []
    plt.rcParams['font.sans-serif'] = ['SimHei']           # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False             # 用来正常显示负号
    for h in band_width:                                   # 分别计算不同窗宽下不同回归模型的RMSE
        kernel_hat = kernel_smoothing(x, y, h)
        linear_pred = local_kernel_regression(y, x, 1, h)
        poly_pred = local_kernel_regression(y, x, 2, h)
        RMSE_kernel = cal_rmse(y, kernel_hat)
        RMSE_linear = cal_rmse(y, linear_pred)
        RMSE_poly = cal_rmse(y, poly_pred)
        rmse_kernel.append(RMSE_kernel)
        rmse_linear.append(RMSE_linear)
        rmse_ply.append(RMSE_poly)
    plt.figure()
    plt.plot(band_width, rmse_kernel, color='blue', label='核平滑')         # 绘制曲线
    plt.plot(band_width, rmse_linear, color='green', label='局部线性回归')
    plt.plot(band_width, rmse_ply, color='yellow', label='局部2阶多项式回归')
    plt.grid(linestyle='-.')                          # 添加网格线
    plt.legend()                                      # 添加图例
    plt.xlabel('窗宽')                                 # 设置x轴和y轴标题
    plt.ylabel('RMSE')
    plt.title('窗宽h与RMSE')                           # 设置图像标题


data = pd.read_csv("D:\\C++ and Python\\.vscode\\数据科学导引\\mcycle.csv")                      # 导入数据集mcycle
data = data.drop('Index', axis=1)                     # 删除Index列
times = data["times"]                                 # 提取times和accel
accel = data["accel"]
print("各属性缺失值个数")                                # 检查数据有无缺失值
print(data.isnull().sum())

bandwidth = np.linspace(1, 10, 20)                    # 设置窗宽在1-10之间等间距取20个值
smooth_acc = kernel_smoothing(times, accel, 1.5)      # 取核回归窗宽为1.5

linear_pred = local_kernel_regression(accel, times, 1, 3)   # 取局部线性回归窗宽为3

poly_pred = local_kernel_regression(accel, times, 2, 6)     # 取局部2阶多项式回归窗宽为6

width_plot(bandwidth, times, accel)                         # 绘制窗宽与RMSE的关系

knot_set = np.array([15, 21, 31, 32])                       # 根据散点图设置四个结点15,21,31,32
spline_pred = cubic_spline(times, accel, knot_set)          # 三次样条回归

# 计算四种回归模型的RMSE
rmse_kernel = cal_rmse(accel, smooth_acc)
print("\n核平滑法")
print("均方根误差RMSE=", rmse_kernel)

rmse_linear = cal_rmse(accel, linear_pred)
print("\n局部线性回归法")
print("均方根误差RMSE=", rmse_linear)

rmse_ply = cal_rmse(accel, poly_pred)
print("\n局部2阶多项式回归法")
print("均方根误差RMSE=", rmse_ply)

rmse_sp = cal_rmse(accel, spline_pred)
print("\n三次样条回归法")
print("均方根误差RMSE=", rmse_sp)

# 回归结果可视化
plt.figure()
plt.scatter(times, accel, alpha=0.2, label='原数据点')       # 绘制times和accel两个变量之间的散点图
plt.plot(times, spline_pred, color='red', label='三次样条回归曲线')      # 绘制回归曲线
plt.plot(times, poly_pred, color='yellow', label='局部2阶多项式回归曲线')
plt.plot(times, smooth_acc, color='blue', label='核平滑曲线')
plt.plot(times, linear_pred, color='green', label='局部线性回归曲线')
plt.xlabel('time')                                         # 设置x轴和y轴标题
plt.ylabel('acceleration')
plt.title("mcycle plotting")                               # 添加图像标题
plt.grid(linestyle='-.')                                   # 添加网格线
plt.legend()                                               # 添加图例
plt.show()
