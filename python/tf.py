#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
標準入力からTF-IDFを求める
1行1ドキュメント
半角スペース区切り
分かち書き，ステム済み

Input:
2 # of documents N
A B
A A B C C

Output:
3 # of terms K
2 # of documents N
A
B
C
A:0.5 B:0.5
A:1.0 B:0.5 C:2.0

sample:
> cat documents.txt|python tf.py
'''
import sys, math
from collections import defaultdict

N = int(sys.stdin.readline())

tf = []
K = {} # of term kinds
for i in range(N):
    # read line
    line = sys.stdin.readline()
    # split line
    terms = line.split()
    # init term
    tmp = defaultdict(int)
    for term in terms:
        tmp[term] += 1
        K[term] = 1
    tf.append(tmp)

# output
print len(K) # of terms K
print N # of doucments 
for k in K.iterkeys():
    print k
for tmp in tf:
    print " ".join(["%s:%d" % (k, v) for k, v in tmp.iteritems()])

for line in sys.stdin:
    print line.strip()
