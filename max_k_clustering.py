from union_find import UnionFind


class Graph(object):
    def __init__(self, triplets, nodes):
        self.triplets = triplets
        self.num_nodes = len(nodes)
        self.nodes = nodes
    
    def kruskal(self, k=4):
       def compute_spacing(uf):
           min_space = float('inf')
           for triplet in self.triplets:
               u, v, w = triplet
               if uf.children[u] != uf.children[v]:
                   if w < min_space:
                       min_space = w
           return min_space
                       
       self.triplets.sort(key=lambda x: (x[2], x[0], x[1]))
       mst = []
       idx = 0
       uf = UnionFind()
       for node in self.nodes:
           uf.children[node] = node
           uf.leaders[node] = [node]

       while len(mst) < self.num_nodes - 1 - k:
           u, v = self.triplets[idx][:2]
           cycle = False
           if uf.children[u] == uf.children[v]:
               cycle = True
           elif len(uf.leaders[uf.children[u]]) >= len(uf.leaders[uf.children[v]]):
               uf.union(uf.children[u], uf.children[v])
           else:
               uf.union(uf.children[v], uf.children[u])
           if not cycle:
               mst.append(self.triplets[idx])
           idx += 1 
       return compute_spacing(uf) 

     
if __name__ == "__main__":
    triplets = []
    nodes = {}
    with open ('clustering.txt') as f:
        cnt = 0
        for line in f:
            if cnt == 0:
                cnt = line.strip().split()[0]
            else:
                u, v, w = list(map(int, line.strip().split()))
                triplets.append([u,v,w])
                if u not in nodes:
                    nodes[u] = True
                if v not in nodes:
                    nodes[v] = True
    g = Graph(triplets, nodes)
    max_k = g.kruskal(k=4-1)
    print(max_k)
    #print(sum([x[2] for x in mst]))


