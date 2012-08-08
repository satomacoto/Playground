#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy

x = scipy.io.loadmat("./oilFlow3Class.mat")
for k,v in x.items():
    print "%s.txt: %dx%d" % (k, len(v[0]), len(v))
    numpy.savetxt(k + ".txt", v, fmt='%15.7e')
