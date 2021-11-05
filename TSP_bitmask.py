from collections import defaultdict


class TSP:
    def __init__(self, dist):
        self.dist = dist
        self.n_nodes = len(dist)

    def min_cost(self):
        S = defaultdict(list)
        subsets = [[]]
        for i in range(2, self.n_nodes+1): 
            subsets += [subset + [i] for subset in subsets]
        for subset in subsets:
            S[len(subset)+1].append(sum([1 << s for s in subset]) + 1)
        ans = {}
        for i in range(2, self.n_nodes+1):
            ans[((1 + (1 << i)), 1 << i)]=dist[i-1][0]
        final_result = []
        for i in range(3, self.n_nodes+1):
            for cntr, subset in enumerate(S[i]):
                if cntr % (len(S[i]) // 100) == 0:
                    print(i, cntr / (len(S[i]) // 100))
                for j in range(2, self.n_nodes + 1):
                    if subset | (1 << j) != subset:
                        continue
                    S_j = subset & ~(1 << j)
                    min_d = float('inf')
                    for k in range(2, self.n_nodes + 1):
                        if k == j or S_j | (1 << k) != S_j:
                            continue
                        if min_d > ans[(S_j, 1 << k)] + dist[k-1][j-1]:
                            min_d = ans[(S_j, 1 << k)] + dist[k-1][j-1]
                    if min_d != float('inf'):
                        ans[(subset, 1 << j)] = min_d
                        if i == self.n_nodes:
                            final_result.append(min_d + dist[j-1][0])
        return min(final_result)

if __name__ == "__main__":
    adj = {}
    points = []
    cnt = 0
    with open ('tsp.txt') as f:
        for line in f:
            if cnt == 0:
                num_cities = int(line.strip())
                cnt = 1
            else:
                points.append(list(map(float, line.split())))
        dist = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
        for i in range(num_cities):
            for j in range(num_cities):
                dist[i][j] = ((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)**0.5
    g = TSP(dist)
    cost = g.min_cost()
    print(cost)
