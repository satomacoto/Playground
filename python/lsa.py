#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
標準入力から潜在意味分析
http://ja.wikipedia.org/wiki/%E6%BD%9C%E5%9C%A8%E6%84%8F%E5%91%B3%E8%A7%A3%E6%9E%90

Input:
5
3
A
C
B
E
D
A:1 C:1 B:1
A:1 B:2 D:1
C:1 E:2 D:2
A B C
A B B D
C D D E E

Output:
5 # of terms
3 # of documents
5 5 # U shape
3 # s shape
3 3 # Vt shape
A
C
B
E
D
0.0508075063713 -0.529238947665 -0.232282569295 0.68150820069 0.446000127763
0.193084227641 -0.254222994688 -0.770992871266 -0.541490219987 0.102212935209
0.0760886657644 -0.76538010426 0.297087888701 -0.140017980703 -0.548213062971
0.907999004467 0.210676120171 -0.050641148513 0.248094799077 -0.25891663721
0.360396172512 -0.158384575129 0.510675103097 -0.401470857973 0.650431406434
0.478516975911 0.310690291996 0.130141272357
0.0903770463932 0.119343217618 0.988731098881
-0.673769835115 -0.723773662974 0.14894930035
-0.733393617899 0.679638787294 -0.0149973341399
A B C
A B B D
C D D E E
'''

import sys
from scipy.sparse import dok_matrix
from scipy.linalg import svd

# read data
K = int(sys.stdin.readline()) # of terms
N = int(sys.stdin.readline()) # of documents

# terms
terms = []
for i in range(K):
    terms.append(sys.stdin.readline().strip())

# term-document matrix
X = dok_matrix((K,N))
for j in range(N):
    line = sys.stdin.readline()
    for kv in line.strip().split():
        tmp = kv.split(":")
        i = terms.index(":".join(tmp[0:-1]))
        v = float(tmp[-1])
        X[i,j] = v

# singular value decomposition
U,s,Vt = svd(X.todense())

# output
print K
print N
print U.shape[0], U.shape[1]
print s.shape[0]
print Vt.shape[0], Vt.shape[1]
# terms
for term in terms:
    print term
# U
for i in range(U.shape[0]):
    print " ".join([str(v) for v in U[i].tolist()])
# s
print " ".join([str(v) for v in s.tolist()])
# Vt
for i in range(Vt.shape[0]):
    print " ".join([str(v) for v in Vt[i].tolist()])
# 
for line in sys.stdin:
    print line.strip()
