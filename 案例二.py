import pandas as pd                                              # 用于导入数据和one-hot编码
import matplotlib.pyplot as plt                                  # 用于绘图
import seaborn as sns                                            # 用于绘制相关系数矩阵热图
from sklearn.model_selection import KFold                        # 用于k折交叉验证
from pyearth import Earth                                        # 用于MARS模型
import warnings                                                  # 用于屏蔽运行过程中可能出现的警告
import numpy as np                                               # 用于矩阵、向量运算

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']                     # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False                       # 用来正常显示负号


def cal_rmse(y, y_hat):                                          # 计算均方根误差RMSE
    y = np.array(y).reshape(1, len(y))
    y_hat = np.array(y_hat).reshape(1, len(y_hat))
    mse = np.mean((y - y_hat) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def R_square(y, y_hat):                                          # 计算判定系数R^2
    y = np.array(y).reshape(1, len(y))
    y_hat = np.array(y_hat).reshape(1, len(y_hat))
    SStot = np.sum((y - np.mean(y)) ** 2)
    SSres = np.sum((y - y_hat) ** 2)
    r2 = 1 - SSres/SStot
    return r2


def cal_loss(X, y, w):                                           # 计算残差平方和
    y_loss = (y - X * w).T * (y - X * w)
    return y_loss


def std_residual(y, y_hat):                                      # 计算标准残差
    rmse = cal_rmse(y, y_hat)
    z_res = (y-y_hat)/rmse
    return z_res


def dummy_encode(dataset):                                       # 哑变量编码
    dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'])
    dataset = dataset.drop(['sex_female'], axis=1)
    dataset = dataset.drop(['smoker_no'], axis=1)
    dataset = dataset.drop(['region_northeast'], axis=1)
    return dataset


def least_squares(X, y):                                         # 最小二乘回归
    X = np.array(X)
    X = np.column_stack((np.ones(np.shape(X)[0]), X))
    X = np.matrix(X)
    y = np.matrix(y)
    if np.linalg.det(X.T * X) == 0:
        print("\nX.T * X为奇异矩阵，回归失败\n")
        return
    coef = (X.T * X).I * X.T * y
    return coef


def lasso_regression(X, y, lam, threshold, max_iter=100):       # 利用坐标下降法实现LASSO回归，threshold为阈值
    m, n = X.shape
    coeff = np.matrix(np.zeros((n, 1)))
    r = cal_loss(X, y, coeff)
    for it in range(max_iter):
        for k in range(n):
            z_k = np.sum(X[:, k].T * X[:, k])
            p_k = 0
            for i in range(m):
                p_k += X[i, k] * (y[i, 0] - sum([X[i, j] * coeff[j, 0] for j in range(n) if j != k]))
            if p_k < -lam / 2:
                w_k = (p_k + lam / 2) / z_k
            elif p_k > lam / 2:
                w_k = (p_k - lam / 2) / z_k
            else:
                w_k = 0
            coeff[k, 0] = w_k
        r_prime = cal_loss(X, y, coeff)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime
        if delta < threshold:
            break
    return coeff


def lasso_trace(X, y, ntest=25):                                # 计算正则化路径中每个lambda对应回归系数
    X = np.column_stack((np.ones(np.shape(X)[0]), X))
    n = X.shape[1]
    ws = np.zeros((ntest, n))
    for i in range(ntest):
        w = lasso_regression(X, y, lam=np.exp(i-10), threshold=0.1)
        ws[i, :] = w.T
    return ws


def KFold_CV(X, y, lambs, n=5):                                 # k折交叉验证，用于求LASSO回归的最佳参数
    K_Fold = KFold(n_splits=n, random_state=1000, shuffle=True)
    minRMSE = 0xffffffff
    best_lamb = lambs[0]
    for lamb in lambs:
        loss = 0
        for train_index, test_index in K_Fold.split(X):
            train_x = X[train_index]
            train_y = y[train_index]
            test_x = X[test_index]
            test_y = y[test_index]
            train_x = np.column_stack((np.ones(np.shape(train_x)[0]), train_x))
            test_x = np.column_stack((np.ones(np.shape(test_x)[0]), test_x))
            coeffs = lasso_regression(train_x, train_y, lamb, threshold=5000000)
            y_hat = test_x * coeffs
            loss = loss + cal_rmse(test_y, y_hat)
        if loss / n < minRMSE:
            minRMSE = loss / n
            best_lamb = lamb
    return best_lamb


def z_score(data):                                           # 对输入数据进行z-score标准化
    data_norm = data.copy()
    for j in range(data_norm.shape[1]):
        column = data_norm[:, j]
        mu = np.mean(column)
        sigma = np.std(column)
        for i in range(data_norm.shape[0]):
            data_norm[i][j] = (data_norm[i][j]-mu)/sigma
    return data_norm


def gaussian_kernel(X, i, h):                                     # 多元高斯核函数
    gauss = np.exp(-(np.linalg.norm(X-X[i], ord=1, axis=1) ** 2) / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h)
    return gauss


def weighted_least_squares(X, y, W):                              # 加权最小二乘
    n = X.shape[0]
    design_matrix = np.matrix(np.ones(n)).T
    coeff = (design_matrix.T * W * design_matrix).I * (design_matrix.T * W * y)
    return coeff


def kernel_smoothing(X, y, h):                                    # 多元核平滑
    n = X.shape[0]
    m = X.shape[1]
    design_matrix = np.matrix(np.ones(n)).T
    y = np.array(y)
    X = np.matrix(X)
    y_pred = np.zeros(n)
    for i in range(n):
        local = []
        for j in range(n):
            tmp = (X[j]-X[i]).reshape(m, 1)
            if np.linalg.norm(tmp, ord=1, axis=0) <= h:
                local.append(j)
        local_matrix = design_matrix[local]
        local_y = y[local]
        weight = gaussian_kernel(X, i, h)
        local_weight = weight[local]
        W = np.diag(local_weight)
        coeff = weighted_least_squares(local_matrix, local_y, W)    # 采用最小二乘得到参数
        y_pred[i] = coeff[0]
    return y_pred


def MARS_model(X, y):                                               # 多元自适应样条回归
    model = Earth()
    model.fit(X, y)
    y_hat = model.predict(X)
    return y_hat


ins = pd.read_csv("insurance.csv")                             # 导入数据，检查是否有缺失值
print("各属性缺失值个数")
print(ins.isnull().sum())

ins = dummy_encode(ins)                                        # 对数据集进行哑变量编码
ins_X = ins.drop(['charges'], axis=1)                          # ins_X为解释变量
charges = ins['charges']                                       # charges为目标特征
charges = np.array(charges).reshape((np.size(charges), 1))
ins_X = np.array(ins_X)

# 考虑到年龄与医疗费用之间的线性关系不是很显著，增添一个年龄的二次项age_square
age = ins_X[:, 0]
age_square = age * age
age_square = age_square.reshape(len(age_square), 1)
ins_X = np.hstack((ins_X, age_square))

# 考虑到过度肥胖可能会使医疗费用增加，此处增加一个is_fat指标，若bmi高于30，则其值为1，否则为0
n = np.shape(ins_X)[0]
is_fat = np.zeros((n, 1))
for i in range(n):
    if ins_X[i][1] >= 30:
        is_fat[i] = 1
    else:
        continue
ins_X = np.hstack((ins_X, is_fat))

# 为体现肥胖指标is_fat和吸烟指标smoker_yes的相互作用，可以将is_fat*smoker_yes也作为自变量加入X中
smoker_yes = ins_X[:, 4]
smoker_yes = smoker_yes.reshape(len(smoker_yes), 1)
fat_smoker = is_fat * smoker_yes
ins_X = np.hstack((ins_X, fat_smoker))

# 普通最小二乘回归
coeff = least_squares(ins_X, charges)
ins_X = np.column_stack((np.ones(np.shape(ins_X)[0]), ins_X))
charges_hat = ins_X * coeff
rmse = cal_rmse(charges, charges_hat)
r2 = R_square(charges, charges_hat)
print("\n加入平方项和交互项的最小二乘回归")
print("均方根误差RMSE=", rmse)
print("R-square=", r2, '\n')
index = np.arange(1, ins_X.shape[0]+1)
residual_1 = std_residual(charges, charges_hat)

# 对正则化系数候选集进行交叉验证，选取给定阈值下的最佳正则化系数进行LASSO回归
ins_X = np.delete(ins_X, 0, axis=1)
lamdas = np.arange(31000, 36000, 1000)
best_lamda = KFold_CV(ins_X, charges, lamdas)
ins_X = np.column_stack((np.ones(np.shape(ins_X)[0]), ins_X))
coeff = lasso_regression(ins_X, charges, best_lamda, threshold=5000000)
charges_hat = ins_X * coeff
rmse = cal_rmse(charges, charges_hat)
r2 = R_square(charges, charges_hat)
print(coeff)
print("LASSO回归（正则化系数：{}）".format(best_lamda))
print("均方根误差RMSE=", rmse)
print("R-square=", r2, '\n')

# 绘制LASSO正则化路径，进行特征选择
ins_X = np.delete(ins_X, 0, axis=1)
ntest = 25
ins_X_norm = z_score(ins_X)                      # 对解释变量和目标特征进行标准化，便于比较回归系数大小
charges_norm = z_score(charges)

coeffs_lasso = lasso_trace(ins_X_norm, charges_norm, ntest)

plt.figure()
lambdas = [i-10 for i in range(ntest)]
plt.plot(lambdas, coeffs_lasso)
plt.xlabel('log(lambda)')                                        # 设置x轴和y轴标题
plt.ylabel('回归系数')
plt.title("LASSO回归正则化路径")
plt.legend(['Intercept', 'age', 'bmi', 'children',
            'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast',
            'region_southwest', 'age_square', 'is_fat', 'fat_smoker'])
plt.show()

# 基于LASSO回归的正则化路径，保留age,bmi,children,smoker_yes,age_square,fat_smoker六个特征作为主要特征
main_feature = np.vstack((ins_X[:, 0].T, ins_X[:, 1].T, ins_X[:, 2].T,
                          ins_X[:, 4].T, ins_X[:, 8].T, ins_X[:, 10].T))
main_feature = main_feature.T

# 绘制上述六个主要特征与目标特征之间的相关系数矩阵
part_features = np.hstack((main_feature, charges))
corr_matrix = pd.DataFrame(part_features, columns=['age', 'bmi', 'children', 'smoker_yes', 'age_square',
                                                   'fat_smoker', 'charges']).corr()
sns.heatmap(np.round(corr_matrix, 3), mask=np.zeros_like(corr_matrix, dtype=bool),
            cmap=sns.diverging_palette(240, 10, as_cmap=True), square=True, annot=True)
plt.title('LASSO特征选择后相关系数矩阵')
plt.show()

# 使用上述六个特征构建回归模型
coeff_main = least_squares(main_feature, charges)
main_feature = np.column_stack((np.ones(np.shape(main_feature)[0]), main_feature))
charges_hat_main = main_feature * coeff_main
rmse_main = cal_rmse(charges, charges_hat_main)
r2_main = R_square(charges, charges_hat_main)
print("\nLASSO特征选择后的最小二乘回归")
print("均方根误差RMSE=", rmse_main)
print("R-square=", r2_main)

# 多元核平滑
h = 1
y_kernel = kernel_smoothing(ins_X_norm, charges_norm, h)       # 采用z-score标准化后的解释变量进行预测
print("\n核平滑方法")
print("均方根误差RMSE=", cal_rmse(charges_norm, y_kernel))
print("R-square=", R_square(charges_norm, y_kernel))

# MARS（多元自适应样条回归）
mars_pred = MARS_model(ins_X, charges)
print("\nMARS（多元自适应样条回归）")
print("均方根误差RMSE=", cal_rmse(charges, mars_pred))

# 绘制线性回归模型标准残差图，检验残差是否满足正态性假定
plt.figure()
plt.scatter(list(index), list(residual_1), alpha=0.3, color='green', label='线性回归')
plt.ylabel('标准化残差')
plt.title('线性回归标准化残差图')
plt.legend()
plt.show()


