#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import numpy as np
import matplotlib.pyplot as plt
 
def main():
    # state x = A * x_ + B * u + w, w~N(0,Q)
    A = np.mat([[1,0],[0,1]])
    B = np.mat([[1,0],[0,1]])
    u = np.mat([[2],[2]])
    Q = np.mat([[1,0],[0,1]])
    # observation Y = C * x + v, v~N(0,R)
    C = np.mat([[1,0],[0,1]])
    R = np.mat([[2,0],[0,2]])
     
    # 初期化
    T = 10 # 観測数
    x = np.mat([[0],[0]]) # 初期位置
    X = [x] # 状態
    Y = [x] # 観測
 
    # 観測データの生成
    for i in range(T):
        x = A * x + B * u + np.random.multivariate_normal([0,0],Q,1).T
        X.append(x)
        y = C * x + np.random.multivariate_normal([0,0],R,1).T
        Y.append(y)
     
    # LKF
    mu = np.mat([[0],[0]])
    Sigma = np.mat([[0,0],[0,0]])
    MF = [mu] # Filter Distribution [t]; p(x_{t}|y_{0:t})
    SF = [Sigma]
    MP = [mu] # Prediction Distribution [t]; p(x_{t}|y_{0:t-1})
    SP = [Sigma]
    for i in range(T):
        # prediction
        mu_ = A * mu + B * u
        Sigma_ = Q + A * Sigma * A.T
        MP.append(mu_)
        SP.append(Sigma_)
        
        # update
        yi = Y[i+1] - C * mu_
        S = C * Sigma_ * C.T + R
        K = Sigma_ * C.T * S.I
        mu = mu_ + K * yi
        Sigma = Sigma_ - K * C * Sigma_        
        MF.append(mu)
        SF.append(Sigma)
    
    # LKS
    mu__ = MF[T]
    Sigma__ = SF[T]
    MS = [mu__] # Smoother Distribution [t]; p(x_{t}|y_{0:T})
    SS = [Sigma__]
    for i in range(T-1, -1, -1):
        J = SF[i] * A * SP[i+1].I
        mu__ = MF[i] + J * (mu__ - MP[i+1])
        Sigma__ = SF[i] + J * (Sigma__ - SP[i+1]) * J.T
        MS = [mu__] + MS
        SS = [Sigma__] + SS
    
    print len(MP), len(SP)
    print len(MF), len(SF)
    print len(MS), len(SS)

    # 描画
    a,b = np.array(np.concatenate(X,axis=1))
    plt.plot(a,b,'rs-')
    a,b = np.array(np.concatenate(Y,axis=1))
    plt.plot(a,b,'g^-')
    a,b = np.array(np.concatenate(MF,axis=1))
    plt.plot(a,b,'bo-')
    a,b = np.array(np.concatenate(MP,axis=1))
    plt.plot(a,b,'co-')
    a,b = np.array(np.concatenate(MS,axis=1))
    plt.plot(a,b,'mo-')
    plt.axis('equal')
    plt.show()
 
if __name__ == '__main__':
    main()