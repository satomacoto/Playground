#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Parameter Estimation for Linear Dynamical Systems

x[t] = A*x[t-1] + w[t]
y[t] = C*x[t] + v[t]
w[t] ~ N(0,Q)
v[t] ~ N(0,R)
'''

import numpy as np
import matplotlib.pyplot as plt

def E_step(y, A, C, Q, R, pi_1, V_1):
    # num of observations
    T = len(y)

    # prediction
    x_p = {} # x_p[t-1] := x_t^{t-1}
    V_p = {} # V_p[t-1] := V_t^{t-1}
    K   = {} # K[t-1]   := K_t
    # filter
    x_f = {} # x_f[t-1] := x_t^t
    V_f = {} # V_f[t-1] := V_t^t
    # smoother
    J   = {} # J[t-1]   := J_{t}
    x_s = {} # x_s[t-1] := x_{t}^T
    V_s = {} # V_s[t-1] := V_{t}^T
    V_a = {} # V_a[t-1] := V_{t,t-1}^T
    # response
    x   = {} # x[t-1]   := {¥hat x}_t
    P   = {} # P[t-1]   := P_t
    P_a = {} # P_a[t-1] := P_{t,t-1}

    # kalman filter
    for t in range(T):
        if t == 0: # initialize
            x_p[t] = pi_1
            V_p[t] = V_1
        else:
            x_p[t] = A * x_f[t-1]
            V_p[t] = A * V_f[t-1] * A.T + Q
        K[t] = V_p[t] * C.T * np.linalg.pinv((C * V_p[t] * C.T + R))
        x_f[t] = x_p[t] + K[t] * (y[t] - C * x_p[t])
        V_f[t] = V_p[t] - K[t] * C * V_p[t]

    # kalman smoother
    x_s[T-1] = x_f[T-1]
    V_s[T-1] = V_f[T-1]
    for t in range(T-1, 0, -1):
        J[t-1] = V_f[t-1] * A.T * np.linalg.pinv(V_p[t])
        x_s[t-1] = x_f[t-1] + J[t-1] * (x_s[t] - A * x_f[t-1])
        V_s[t-1] = V_f[t-1] + J[t-1] * (V_s[t] - V_p[t]) * J[t-1].T

    I = np.mat(np.eye(*A.shape))
    V_a[T-1] = (I - K[T-1] * C) * A * V_f[T-2]
    for t in range(T-1, 1, -1):
        V_a[t-1] = V_f[t-1] * J[t-2].T + J[t-1] * (V_a[t] - A * V_f[t-1]) * J[t-2].T

    # set response
    for t in range(T):
        x[t] = x_s[t]
        P[t] = V_s[t] + x_s[t] * x_s[t].T
        if t == 0: continue
        P_a[t] = V_a[t] + x_s[t] * x_s[t-1].T

    return x, P, P_a

def M_step(y, x, P, P_a):
    # num of observations
    T = len(y)
    # Output matrix
    C = sum([y[t]*x[t].T for t in range(T)]) * np.linalg.pinv(sum([P[t] for t in range(T)]))
    # Output noise covariance
    R = sum([y[t]*y[t].T - C * x[t] * y[t].T for t in range(T)]) / T
    # State dynamics matrix
    A = sum([P_a[t] for t in range(1,T)]) * np.linalg.pinv(sum([ P[t-1] for t in range(1,T)]))
    # State noise covariance
    Q =  (sum([P[t] for t in range(1,T)]) - A * sum([P_a[t] for t in range(1,T)])) / (T - 1)
    #Q = sum( [ (P[t] - A * P_a[t]) for t in range(1,T) ] ) / (T-1)
    # Initail state mean
    pi_1 = x[1]
    # Initial state covariance
    V_1 = P[1] - x[1] * x[1].T

    return A, C, Q, R, pi_1, V_1

if __name__ == '__main__':
    # テストデータ生成
    dt = 0.1
    x = np.mat([[0.0],
                [0.0]])
    A = np.mat([[1.0,dt],
                [0.0,1.0]])
    C = np.mat([[1.0,0]])
    Q = np.mat([[dt**4/4,dt**3/2],
                [dt**3/2,dt**2]])
    R = np.mat([[1.0]])
    print "A\n%s\nC\n%s\nQ\n%s\nR\n%s" % (A, C, Q, R)

    X = [] # 状態
    Y = [] # 観測
    K = 500 # サンプル数
    for i in range(K):
        x = A * x + np.mat(np.random.multivariate_normal((0,0),Q)).T
        y = C * x + np.mat(np.random.normal(0,1)).T
        X.append(x)
        Y.append(y)


    # 推定
    # 初期値をランダムに振る
    A = np.mat(np.random.rand(2,2))
    C = np.mat(np.random.rand(1,2))
    Q = np.mat(np.random.rand(2,2))
    Q = (Q + Q.T) / 2
    R = np.mat(np.random.rand(1,1))
    pi_1 = np.mat(np.random.rand(2,1))
    V_1 = np.mat(np.random.rand(2,2))
    V_1 = (V_1 + V_1.T) / 2

    N = 100 # EM回数
    e = E_step(Y, A, C, Q, R, pi_1, V_1)
    for i in range(100):
        print i
        m = M_step(Y, *e)
        e = E_step(Y, *m)


    # 結果表示
    print "A\n%s\nC\n%s\nQ\n%s\nR\n%s\npi_1\n%s\nV_1\n%s" % m
    x_hat, pi, pa = e
    # テストデータ
    X1 = []
    X2 = []
    # 推定結果
    X3 = []
    X4 = []
    for x in X:
        X1.append(x[0,0])
        X2.append(x[1,0])
    for i in x_hat:
        X3.append(x_hat[i][0,0])
        X4.append(x_hat[i][1,0])
    plt.plot(X1, 'r-')
    plt.plot(X2, 'b-')
    plt.plot(X3, 'm-')
    plt.plot(X4, 'c-')
    plt.show()

