import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 载入数据，返回一个二维的numpy数组，
# 第一维是 x轴数据，第二维是 y轴数据
def loadtData(file_path):
    X = np.array([])
    Y = np.array([])
    for i in open(file_path):
        # 根据逗号的位置取出数据
        x = i[0:i.index(',') - 1]
        y = i[i.index(',') + 1:len(i) - 1]
        # 读出的数据是字符串类型，需要转换为浮点类型
        X = np.append(X, float(x))
        Y = np.append(Y, float(y))
    return np.array([X, Y])


# 将数据可视化,使用散点图
def plotData(X, y, x_label, y_label):
    plt.scatter(X, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# 计算代价函数
def computeCost(X, y, theta):
    m = len(y)
    result = np.dot(X, theta)
    result = result - y.reshape(97, 1)
    result = np.square(result)
    result = np.sum(result, axis=0)
    result = result / (2.0 * float(m))
    return result


# 梯度下降
def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(0, iterations):
        theta_0 = theta[0] - alpha * computePartialDerivative(X, y, theta, 0)
        theta_1 = theta[1] - alpha * computePartialDerivative(X, y, theta, 1)
        theta[0] = theta_0
        theta[1] = theta_1
        print("iterations : ", i, " cost : ", computeCost(X, y, theta))
    return theta


# 计算偏导数
def computePartialDerivative(X, y, theta, num):
    m = len(y)
    result = 0
    for i in range(0, m):
        result += (theta[0] * X[i][0] + theta[1] * X[i][1] - y[i]) * X[i][num]
    result /= m
    return result


# ======================== 绘图 ====================================
print("loading data ex1data1.txt...")
ex1data = loadtData('ex1data1.txt')
X = ex1data[0]
y = ex1data[1]
print(X)
print(y)
print()

x_label = "Population of City in 10,000s"  # 设置横纵坐标的标题
y_label = "Profit in $10,000s"
# 绘图
plotData(X, y, x_label, y_label)

# ======================= 代价函数 和 梯度下降 ======================
m = len(y)  # 样本数量
X = np.c_[np.ones(m), X]  # 为X增加一行 值为1
theta = np.zeros((2, 1), float)  # 初始化参数theta

# 一些梯度下降的设置
iterations = 1500  # 迭代次数
alpha = 0.01  # 学习率
print("Testing the coust function ...\n")
# 计算并显示初始的代价值
J = computeCost(X, y, theta)

print('With theta = [0 ; 0] ');
print("Cost computed = %f \n" % J)
print('Expected cost value (approx) 32.07\n');

# 继续测试代价函数
theta[0] = -1
theta[1] = 2
J = computeCost(X, y, theta);
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n' % J);
print('Expected cost value (approx) 54.24\n');

print('\nRunning Gradient Descent ...\n')
# 运行梯度下降
theta = gradientDescent(X, y, theta, alpha, iterations);

# 将theta的值打印到屏幕上
print('Theta found by gradient descent:\n');
print('theta_0 : %f \ntheta_1 : %f' % (theta[0], theta[1]));
print('Expected theta values (approx)\n');
print(' -3.6303\n  1.1664\n\n');

# 绘制线性拟合的图
plt.plot(X[:, 1], np.dot(X, theta))
plt.scatter(X[:, 1], y, c='r')
plt.show()

# 绘制三维的图像
fig = plt.figure()
axes3d = Axes3D(fig)
# 指定参数的区间
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
# 存储代价函数值的变量初始化
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
# 为代价函数的变量赋值
for i in range(0, len(theta0_vals)):
    for j in range(0, len(theta1_vals)):
        t = np.zeros((2, 1))
        t[0] = theta0_vals[i]
        t[1] = theta1_vals[j]
        J_vals[i, j] = computeCost(X, y, t)
# 下面这句代码不可少，matplotlib还不熟悉，后面填坑
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)  # 必须加上这段代码
axes3d.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='rainbow')
plt.show()

