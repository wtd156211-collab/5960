"""按文档编号做连通分量的并查集。只存放真正参与候选对的文档。"""


class UnionFind:
    def __init__(self):
        self._parent = {}
        self._size = {}

    def find(self, x):
        parent = self._parent
        if x not in parent:
            parent[x] = x
            self._size[x] = 1
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]

    def components(self):
        groups = {}
        for x in self._parent:
            groups.setdefault(self.find(x), []).append(x)
        return list(groups.values())
