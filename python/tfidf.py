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
'''
import sys, math
from collections import defaultdict

N = int(sys.stdin.readline())

dc = defaultdict(int)
tf = []
for i in range(N):
    # read line
    line = sys.stdin.readline()
    # split line
    terms = line.split()
    # init term
    tmp = defaultdict(float)
    # count terms
    s = len(terms)
    for term in terms:
        tmp[term] += 1.0
    # set term frequency    
    for k, v in tmp.iteritems():
        tmp[k] /= s
    tf.append(tmp)
    # count number of documents where the term appears
    for term in set(terms):
        dc[term] += 1

# output
print len(dc) # of terms K
print N # of doucments 
for k in dc.iterkeys():
    print k
for tmp in tf:
    print " ".join(["%s:%f" % (k, v * math.log(1.0 * N / (dc[k]))) for k, v in tmp.iteritems() if math.log(1.0 * N / (dc[k])) > 0])

for line in sys.stdin:
    print line.strip()
