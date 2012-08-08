#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
connecting dots

"""


import glob

import numpy as np
import scipy as sp

from scipy.sparse import dok_matrix
from numpy import linalg as LA
from scipy import sparse as SP
import networkx as nx

def coherence1(*d):
    n = len(d)
    score = 0
    for i in range(n-1):
        score += len(set(d[i]).intersection(d[i+1]))
    return score

def coherence2(*d):
    """
    dは語のリスト
    """
    n = len(d)
    score = []
    for i in range(n-1):
        score += [len(set(d[i]).intersection(d[i+1]))]
    return min(score)

def make_ddwv():
    T = 2914
    N = 29
    D = sum([ np.load("output/%d-0.15" % i) for i in range(T) ])
    #for i in range(T):
        #D += np.load("output/%d-0.15" % i)
        #x = dok_matrix(np.load("output/%d-0.15" % i))
        #for (j,k),v in x.iteritems():
        #    D[i,j,k] = v
    print D
    
    source = 1
    target = 28
    
    res = []
    for i in range(N):
        for j in range(N):
            if i == j or i == source or i == target or j == source or j == target: continue
            res.append((D[source,i] + D[i,j] + D[j,target],i,j))
    print sorted(res)[-10:]


def Pi(terms,tfidf,w=None,eps=0.15,max_iter=100,tol=1e-08):
    """
    terms は語のリスト
    tfidf は語とtfidf値の辞書のリスト
    """
    # dok_matrixをつくる
    n = len(terms) # 語の種類
    m = len(tfidf) # 文書の数
    # 文書から単語への遷移行列
    A = dok_matrix((m,n))
    for i,d in enumerate(tfidf):
        s = sum(d.values()) # sum of tfidf for each document
        for k,v in d.iteritems():
            j = terms.index(k)
            A[i,j] = v / s
    # 単語から文書への遷移行列
    B = dok_matrix((n,m))
    s = sum(A) # sum of col
    stop = terms.index(w) if w else -1 # 排除する語
    for (i,j),v in A.iteritems():
        if j == stop: continue
        B[j,i] = v / s[0,j]
    # 文書から文書への遷移行列
    C = A * B
    # ランダムウォークの定常状態を求める
    x = one = SP.identity(m)
    for i in range(max_iter):
        x_ = eps * one + (1.0 - eps) * x * C 
        if LA.norm(x_.todense() - x.todense()) < tol:
            break
        x = x_
    return x_
    #x_.todense().dump('hoge')
    #print np.load('hoge')
    
    #print np.sum(x.todense())
    #print np.sum(C.todense(),axis=1)
    # Graphに変換
    #G = nx.from_scipy_sparse_matrix©
    #print nx.pagerank(G,alpha=0.85)
    #print nx.pagerank_numpy(G,alpha=0.85)
    #print nx.pagerank_scipy(G,alpha=0.85)
    #print nx.google_matrix(G,alpha=0.85)

def get_influence(terms,tfidf,eps=0.15):
    """
    influenceを取得
    """
    influence = {}
    p = Pi(terms,tfidf,eps=eps)
    for i,term in enumerate(terms):
        print term
        tmp = Pi(terms,tfidf,w=term,eps=eps)
        (p-tmp).todense().dump("output/" + str(i) + "-" + str(eps))
    return influence

if __name__ == '__main__':

    # 単語を取得
    terms = []
    f = open('terms.txt')
    for line in f:
        terms.append(line.strip())

    # tfを取得
    tf = []
    f = open('tf.txt')
    for line in f:
        tmp = [ x.split(':') for x in line.split() ]
        tmp = dict([ (k, int(v)) for k,v in tmp ])
        tf.append(tmp)
    
    # tfidfを取得
    tfidf = []
    f = open('tfidf.txt')
    for line in f:
        tmp = [ x.split(':') for x in line.split() ]
        tmp = dict([ (k, float(v)) for k,v in tmp ])
        tfidf.append(tmp)
    
    
    # 2種類のcohereceによる連鎖
    n = 29
    source = 1
    target = 28
    c1 = []
    c2 = []
    for i in range(29):
        for j in range(29):
            if j in [source, target, i]: continue
            c1 += [( coherence1(tf[source].keys(), tf[i].keys(), tf[j].keys(), tf[target].keys()), i, j )]
            c2 += [( coherence2(tf[source].keys(), tf[i].keys(), tf[j].keys(), tf[target].keys()), i, j )]
    print sorted(c1,reverse=True)[:10]
    print sorted(c2,reverse=True)[:10]
    
    # 
    #get_influence(terms,tfidf)
    make_ddwv()
