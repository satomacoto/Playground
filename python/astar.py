#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import networkx as nx
import random
from collections import defaultdict
 
def astar(G,start,goal,heuristic_cost_estimate,weight='weight'):
    """
    A*アルゴリズム
     
    G:グラフ(networkx.Graph)
    start,goal:ノード
    heuristic_cost_estimate:ヒューリスティック関数
    weight:エッジの重みのキー
    """
    closedset = [] # 計算済みノード
    openset = [start] # 計算中ノード
    came_from = {} # 親のノード
     
    g_score = defaultdict(float) # スタートノードからの（推定）最小コスト
    h_score = defaultdict(float) # ゴールノードまでの（推定）最小コスト
    f_score = defaultdict(float) # 最短経路の（推定）コスト
     
    # スタートノードの設定
    g_score[start] = 0
    h_score[start] = heuristic_cost_estimate(start,goal)
    f_score[start] = h_score[start]
     
    while openset:
        # 計算中ノードから推定コスト最小のxを取得
        x = min(openset,key=lambda k:f_score[k])
        # ゴールだったら終了
        if x == goal:
            return reconstruct_path(came_from,goal)
         
        openset.remove(x) # 計算中から削除
        closedset.append(x) # 計算済みに追加
         
        # 隣接ノードについて
        for y in G.neighbors(x):
            # 計算済みなら継続
            if y in closedset:
                continue
            # 移動コストを付加
            if weight in G[x][y]:
                tentative_g_score = g_score[x] + G[x][y][weight]
            else:
                tentative_g_score = g_score[x] + 1.0
             
            # 計算中でなければ
            if y not in openset:
                openset.append(y)
                tentative_is_better = True
            # 計算中でコストが下がるなら
            elif tentative_g_score < g_score[y]:
                tentative_is_better = True
            # 計算中でコストが上がるなら
            else:
                tentative_is_better = False
             
            # もし今まで計算した中でコスト最小なら
            if tentative_is_better:
                came_from[y] = x
                g_score[y] = tentative_g_score
                h_score[y] = heuristic_cost_estimate(y,goal)
                f_score[y] = g_score[y] + h_score[y]
    # 到達しない場合はNoneを返す
    return None
 
def reconstruct_path(came_from,current_node):
    """
    結果表示用
    """
    if current_node in came_from:
        return reconstruct_path(came_from,came_from[current_node]) + [current_node]
    else:
        return [current_node]
 
if __name__ == '__main__':
    """
    S-o-o
    | | |
    o-o-o
    | | |
    o-o-G
    """
    G = nx.grid_graph(dim=[3,3])
    for s,t in G.edges():
        weight = random.random()
        G[s][t]['weight'] = weight
        print "%s %s %.2f" % (s,t,weight)
    print
 
    def dist(a, b):
        "直線距離"
        (x1, y1) = a
        (x2, y2) = b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
     
    def monotone(a,b):
        "ダイクストラ法"
        return 0
 
    start = (0,0)
    goal = (2,2)
     
    print astar(G,start,goal,dist)
    print astar(G,start,goal,monotone) # equivalent to running Dijkstra
    print nx.astar_path(G,start,goal,dist) # NexworkX