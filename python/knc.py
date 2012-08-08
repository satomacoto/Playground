#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pprint import pprint
import numpy as np
import networkx as nx
import pylab as P
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = []
    lines = open('analysis_result2').readlines()
    for i in range(0, len(lines), 3):
        source = lines[i].strip()
        target = lines[i+1].strip()
        value = float(lines[i+2].split()[1])
        data.append((source,target,value))
    f = lambda x: filter(lambda (s,t,v): s == x, data)
    sources = set([ s for s,t,v in data ])

    tr = {}
    for source in sources:
        tr[source] = {}
    for s,t,v in data:
        tr[s][t] = v
    
    # マルコフ過程で遷移生成
    print "マルコフ過程"
    for i in range(10):
        node = 'come in'
        print node
        while 1:
            if node == 'go out':
                break
            s = sum(tr[node].values())
            node = np.random.choice(tr[node].keys(), p=[ v/s for v in tr[node].values()])[0]
            print node
    
    # ネットワーク
    G = nx.DiGraph()
    G.add_weighted_edges_from(data)
    
    # PageRank
    print
    print "PageRank: alpha=0.85"
    pr = nx.pagerank(G,alpha=0.85)
    for v,k in sorted([ (v,k) for k,v in pr.iteritems()]):
        print v,k
    
    # HITS
    print
    print "HITS"
    h, a = nx.hits(G)
    print "hub"
    for v,k in sorted([ (v,k) for k,v in h.iteritems()]):
        print v,k
    print "authority"
    for v,k in sorted([ (v,k) for k,v in a.iteritems()]):
        print v,k

    A = nx.to_numpy_matrix(G)
    w,v = np.linalg.eig(A)
    ev = []

    for i in range(len(w)):
        ev.append((w[i],v[:,i]))
    ev = sorted(ev,key=lambda x:x[0],reverse=True)
    x = np.concatenate((ev[0][1].T,ev[1][1].T)).T
    print x[0,:]
    
    

    """
    P.plot(ev[0][1],ev[1][1],'o')
    P.draw()
    P.show()
    """

"""
[[1.0,dt],
 [0 ,1.0]]

"""
