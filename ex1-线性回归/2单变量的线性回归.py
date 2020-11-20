#  参考：https://blog.csdn.net/SugerOO/article/details/89737776
import pandas as pd  # https://www.cnblogs.com/traditional/p/12514914.html
import matplotlib.pyplot as plt
import numpy as np

# 读入数据
path = "D:/0Study0/学习/机器学习/主要资料/code/ex1-linear regression/ex1data1.txt"
data = pd.read_csv(path, header=None,
                   names=['Population',
                          'Profit'])  # sep=","设置分隔符，默认为逗号     data = pd.read_csv(path), 默认第一行为属性名   DataFrame
m = len(data)
alpha = 1


# print(data.head(1)) 前1行

#  https://blog.csdn.net/weixin_44166997/article/details/88672954?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
# #  散点图
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))  # figsize:图片尺寸大小
# plt.show()
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.mat(np.zeros(theta.shape))  # np.zeros((1,2))
    parameters = int(theta.ravel().shape[1])  # shape[0]和shape[1]分别代表行和列的长度
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost


# 这个部分实现了Ѳ的更新

data.insert(0, 'Ones', 1)  # 插入第0列，属性名为“Ones”，值为1
cols = data.shape[1]  # [0]=97行, [1]=3列
X = data.iloc[:, :-1]  # X是data里的除最后列  A:B,C:D——第A行到第B行，不包括B，第C列到第D列
y = data.iloc[:, cols - 1:cols]  # y是data最后一列
X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat([0, 0])

alpha = 0.01
iters = 1500
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)
predict1 = [1, 3.5] * g.T
print("predict1:", predict1)
predict2 = [1, 7] * g.T
print("predict2:", predict2)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2) #第二象限
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
# if __name__ == '__main__':
#     print("hello")
