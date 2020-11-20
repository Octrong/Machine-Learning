import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def J1(t1,t2):
    s=0.0
    for i in range(m):
        s=s+(t1*x[i]+t2-y[i])*x[i]
    return s/m
def J2(t1,t2):
    s=0.0
    for i in range(m):
        s=s+(t1*x[i]+t2-y[i])
    return s/m
def g(t1,t2):
    for i in range(1500):
        print(i)
        temp1=t1-0.01*J1(t1,t2)
        temp2=t2-0.01*J2(t1,t2)
        t1=temp1
        t2=temp2
    return t1,t2
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
if __name__ == '__main__':
    print("loading data ex1data1.txt...")
    ex1data = loadtData('ex1data1.txt')
    x = ex1data[0]
    print(x[0])
    y = ex1data[1]
    m=len(y)
    t1,t2=g(0,0)
    print(t1,t2)


