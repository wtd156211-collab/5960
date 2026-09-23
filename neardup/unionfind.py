"""基于 array 的并查集，几百万篇文档时内存只有几十 MB。"""

from array import array


class UnionFind:
    def __init__(self, size):
        self.parent = array("I", range(size))
        self.rank = array("B", bytes(size))

    def find(self, x):
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rank = self.rank
        parent = self.parent
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
