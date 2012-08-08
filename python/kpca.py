#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kernel PCA カーネル主成分分析
http://en.wikipedia.org/wiki/Kernel_principal_component_analysis
"""

import numpy as np
import matplotlib.pyplot as plt


# カーネル k(x,y) 2種類
# 利用しない方はコメントアウト
#k = lambda x, y: (np.dot(x,y) + 1.0)**2
k = lambda x, y: np.exp(-np.linalg.norm(x-y)**2/1.0)

# データ読み込み
# kpca.data
# <x1> <x2> <class>
# 6.040760282116731938e-02 2.003600989193874138e-01 0
A = np.loadtxt('kpca.data')

# データ総数
N = len(A)
# 色
color = ['r.', 'g.', 'b.']

# 元データプロット
#for i in range(len(A)):
#    plt.plot(A[i,0],A[i,1],color[int(A[i,2])])
#plt.draw()
#plt.show()

# kernel matrix
K = np.zeros((N,N))
for i in range(N):
    for j in range(i,N):
        K[i,j] = K[j,i] = k(A[i,0:2], A[j,0:2])
# centered kernel matrix
ones = np.mat(np.ones((N,N))) / N
K = K - ones * K - K * ones + ones * K * ones

print "centered kernel matrix:"
print K

# 第一固有値，第二固有値と対応する固有ベクトル
w,v = np.linalg.eig(K)
ind = np.argsort(w)
x1 = ind[-1] # 第一固有値のインデックス
x2 = ind[-2] # 第二固有値のインデックス

print "1st eigenvalue:"
print w[x1]
print "2nd eigenvalue:"
print w[x2]

# プロット
for i in range(N):
    plt.plot(v[i,x1],v[i,x2],color[int(A[i,2])])
plt.draw()
plt.show()
