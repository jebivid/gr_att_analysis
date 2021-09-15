class UnionFind(object):
    def __init__(self):
        self.children = {}
        self.leaders = {}

    def union(self, cur_parent, prev_parent):
        assert len(self.leaders[cur_parent]) >= len(self.leaders[prev_parent]) 
        for idx, elem in enumerate(self.leaders[prev_parent]):
            self.children[elem] = cur_parent
            self.leaders[cur_parent].append(elem)
        del self.leaders[prev_parent]
    
    def parent_less(self, elem):
        return self.children[elem] == elem and not elem in self.leaders
