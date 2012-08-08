#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Parameter Estimation for Linear Dynamical Systems

x[t] = A*x[t-1] + w[t]
y[t] = C*x[t] + v[t]
w[t] ~ N(0,Q)
v[t] ~ N(0,R)
'''

from pprint import pprint
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
    x   = {} # x[t-1]   := {\hat x}_t
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
        K[t] = V_p[t] * C.T * (C * V_p[t] * C.T + R).I        
        x_f[t] = x_p[t] + K[t] * (y[t] - C * x_p[t])
        V_f[t] = V_p[t] - K[t] * C * V_p[t]
    
    # kalman smoother
    x_s[T-1] = x_f[T-1]
    V_s[T-1] = V_f[T-1]
    for t in range(T-1, 0, -1):
        J[t-1] = V_f[t-1] * A.T * V_p[t].I
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
    C = sum([ y[t]*x[t].T for t in range(T)]) * sum([ P[t] for t in range(T)]).I
    # Output noise covariance
    R = sum([ y[t]*y[t].T - C * x[t] * y[t].T for t in range(T)]) / T
    # State dynamics matrix
    A = sum([ P_a[t] for t in range(1,T)]) * sum([ P[t-1] for t in range(1,T)]).I
    # State noise covariance
    Q =  ( sum([ P[t] for t in range(1,T)]) - A * sum([ P_a[t] for t in range(1,T)]) ) / (T-1)
    #Q = sum( [ (P[t] - A * P_a[t]) for t in range(1,T) ] ) / (T-1)
    # Initail state mean
    pi_1 = x[1]
    # Initial state covariance
    V_1 = P[1] - x[1] * x[1].T
    
    return A, C, Q, R, pi_1, V_1

def make_dataset():
    # sample problem
    dt = 0.1
    x = np.mat([[1.0],
                [1.0]])
    A = np.mat([[1.0,dt],
                [0.0,1.0]])
    C = np.mat([[1.0,0]])
    Q = np.mat([[dt**4/4,dt**3/2],
                [dt**3/2,dt**2]])
    R = np.mat([[1.0]])
    
    f = open('parameter_estimation.txt','w')
    for i in range(2000):
        x = A * x + np.mat(np.random.multivariate_normal((0,0),Q)).T
        f.write('%f %f ' % (x[0,0],x[1,0]))
        y = C * x + np.mat(np.random.normal(0,1)).T
        f.write(' %f\n' % y[0,0])
    f.close()


def kf(y,A,C,Q,R,pi_1,V_1):

    T = len(y)
    
    # prediction
    x_p = {} # x_p[t-1] := x_t^{t-1}
    V_p = {} # V_p[t-1] := V_t^{t-1}
    K   = {} # K[t-1]   := K_t
    # filter
    x_f = {} # x_f[t-1] := x_t^t
    V_f = {} # V_f[t-1] := V_t^t

    # kalman filter
    for t in range(T):
        if t == 0: # initialize
            x_p[t] = pi_1
            V_p[t] = V_1
        else:
            x_p[t] = A * x_f[t-1]
            V_p[t] = A * V_f[t-1] * A.T + Q
        K[t] = V_p[t] * C.T * (C * V_p[t] * C.T + R).I
        x_f[t] = x_p[t] + K[t] * (y[t] - C * x_p[t])
        V_f[t] = V_p[t] - K[t] * C * V_p[t]
    return x_f,V_f


def test1():
    x = np.mat([[0.0]])
    A = np.mat([[1.0]])
    C = np.mat([[1.0]])
    
    print x,A,C
    
    X = []
    Y = []
    T = 1000
    for i in range(T):
        x = A * x + np.mat(np.random.normal(0,1)).T
        y = C * x + np.mat(np.random.normal(0,1)).T
        X.append(x)
        Y.append(y)
    
    A = np.mat(np.identity(1))
    C = np.mat(np.random.rand(1,1))
    Q = np.mat(np.random.rand(1,1))
    R = np.mat(np.random.rand(1,1))
    pi_1 = np.mat(np.random.rand(1,1))
    V_1 = np.mat(np.random.rand(1,1))

    e = E_step(Y,A,C,Q,R,pi_1,V_1)
    for i in range(T):
        m = M_step(Y,*e)
        e = E_step(Y,*m)
    print "A=%s\nC=%s\nQ=%s\nR=%s\npi_1=%s\nV_1=%s" % m

def test2():
    X1 = []
    X2 = []
    Y = []
    for line in open('parameter_estimation.txt'):
        x1,x2,y = line.split()
        X1.append(float(x1))
        X2.append(float(x2))
        Y.append(np.mat(float(y)))
    
    A = np.mat(np.identity(2))
    C = np.mat(np.random.rand(1,2))
    Q = np.mat(np.random.rand(2,2))
    R = np.mat(np.random.rand(1,1))
    pi_1 = np.mat(np.random.rand(2,1))
    V_1 = np.mat(np.random.rand(2,2))

    
    e = E_step(Y,A,C,Q,R,pi_1,V_1)
    for i in range(4):
        m = M_step(Y,*e)
        e = E_step(Y,*m)
        print "A=%s\nC=%s\nQ=%s\nR=%s\npi_1=%s\nV_1=%s" % m

    #print "x:%s, P:%s, Pa:%s" % e
    # A, C, Q, R, pi_1, V_1 = m
    # x, P, Pa = e
    print "A=%s\nC=%s\nQ=%s\nR=%s\npi_1=%s\nV_1=%s" % m
    x_, P, Pa = e
    
    X1_ = []
    X2_ = []
    for i in range(2000):
        X1_.append(x_[i][0,0])
        X2_.append(x_[i][1,0])
    
    plt.plot(X1,'b.-')
    plt.plot(X2,'r.-')
    plt.plot([ y[0,0] for y in Y ],'y.-')
    plt.plot(X1_,'g.-')
    plt.plot(X2_,'c.-')
    
    
    plt.draw()
    plt.show()
    
    """
        X1_ = []
    X2_ = []
    x_,V = kf(Y,*m)
    for i in range(2000):
        X1_.append(x_[i][0,0])
        X2_.append(x_[i][1,0])
    plt.plot(X1_,'g.-')
    plt.plot(X2_,'c.-')
    plt.draw()
    plt.show()
    """

if __name__ == '__main__':

    #make_dataset()
    # sample problem
    dt = 0.1
    x = np.mat([[0.0],
                [0.0]])
    A = np.mat([[1.0,dt],
                [0.0,1.0]])
    C = np.mat([[1.0,0]])
    Q = np.mat([[dt**4/4,dt**3/2],
                [dt**3/2,dt**2]])
    R = np.mat([[1.0]])
    
    X = []
    Y = []
    for i in range(300):
        x = A * x + np.mat(np.random.multivariate_normal((0,0),Q)).T
        y = C * x + np.mat(np.random.normal(0,1)).T
        X.append(x)
        Y.append(y)
    
    pi_1 = np.mat([[0.0],
                [0.0]])
    V_1 = np.mat([[0.0,0.0],
                  [0.0,0.0]])
    PI, P, Pa = E_step(Y, A, C, Q, R, pi_1, V_1)
    
    PF, VF = kf(Y, A, C, Q, R, pi_1, V_1)
    
    X1_ = []
    X2_ = []
    X3_ = []
    for i in range(100):
        X1_.append(X[i][0,0])
        X2_.append(PI[i][0,0])
        X3_.append(PF[i][0,0])
#    plt.plot(X1_,'g.-')
#    plt.plot(X2_,'c.-')
#    plt.plot(X3_,'r.-')
#    plt.draw()
#    plt.show()

    e = E_step(Y,A,C,Q,R,pi_1,V_1)
    for i in range(100):
        m = M_step(Y,*e)
        e = E_step(Y,*m)
        if i%10 == 0:
            print "A=%s\nC=%s\nQ=%s\nR=%s\npi_1=%s\nV_1=%s" % m



