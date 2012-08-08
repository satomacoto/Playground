#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kernel PCA
カーネル主成分分析
http://en.wikipedia.org/wiki/Kernel_principal_component_analysis
"""

import numpy as np
import matplotlib.pyplot as plt

dt = np.dtype([('data', 'f8', (4,)), ('class', 'S16')])
iris =  np.loadtxt('iris.data', delimiter=',', dtype=dt)

print iris[0]['data']

data = np.array(iris)
data = (data - data.mean(axis=0)) / data.std(axis=0)
C = np.corrcoef(iris.data, rowvar=0)
w, v = np.linalg.eig(C)
print w
print v
