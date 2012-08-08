#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

N = int(sys.stdin.readline())
res = defaultdict(int)

for i in range(N):
    terms = sorted(set(sys.stdin.readline().strip().split()))
    T = len(terms)
    for i in range(T):
        for j in range(T):
            res[terms[i], terms[j]] += 1

for (s, t), v in res.iteritems():
    print s, t, v
