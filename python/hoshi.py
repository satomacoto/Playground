#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MeCab

def adjnoun(text):
    client = MeCab.Tagger()
    pos = "形容詞,形容動詞,感動詞,副詞,連体詞,名詞,動詞"
    m = client.parseToNode(text)
    res = []
    while m:
        f = m.feature.split(",")
        print " ".join(f)
        if f[0] in pos or pos == "":
            res += [(m.surface, f[0])]
        m = m.next
    return res

if __name__ == '__main__':
    # 形容詞＋名詞を抜き出す
    text = """飛べない豚は唯のブタ．吾輩は猫である．名前はまだない．おいしいごはんが食べたいな．"""
    pairs = adjnoun(text)
    for k,v in pairs:
        print k,v
