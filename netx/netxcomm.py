from random import random, shuffle, sample

from .netxfun import *
from .netxg import DiGraph as NetxDiGraph

# ---------------------------------------------------------------------------
# shuffle_of
# remove_loops
# ---------------------------------------------------------------------------

def shuffle_of(l):
    shuffle(l)
    return l
# end


def remove_loops(g: AGraph) -> AGraph:
    toremove = []
    for u, v in g.edges:
        if u == v:
            toremove.append((u, v))
    g.remove_edges_from(toremove)
    return g
# end


# ---------------------------------------------------------------------------
# clusterize_graph
# ---------------------------------------------------------------------------

class SwapEdges:
    def __init__(self, g: AGraph,
                 cdict: Union[Dict[str, List[int]], List[List[int]]],
                 rho: float = 0.4,
                 sigma: float = 0.3,
                 connected: bool = True,
                 ntries: int = 10):

        if isinstance(cdict, dict):
            cdict = list(cdict.values())
        assert isinstance(cdict, list)

        self.g = g
        self.directed = g.is_directed()
        self.connected = connected
        # communities
        self.communities = cdict
        # probability keep edge
        self.rho = rho
        self.sigma = sigma
        # n retries
        self.ntries = ntries
        # n of clusters
        self.nc = len(self.communities)
        # map: node -> cluster
        self.vc: Dict[int, int] = dict()

        # edges in cluster & between clusters
        self.edges: Dict[Tuple[int, int], List[Edge]] = defaultdict(lambda: [])
        # adjacent lists
        self.adj: Dict[int, Set[int]] = defaultdict(lambda: set())

        self.toupdate: AGraph = None
        pass

    def apply(self, toupdate: AGraph):
        self.toupdate: nx.Graph = toupdate
        self.directed = toupdate.is_directed()

        self._populate()
        self._swap_edges()
    # end

    def _populate(self):
        communities = self.communities

        # 1) create the map node -> community
        for c in range(self.nc):
            community = communities[c]
            for n in community:
                self.vc[n] = c

        g = self.g
        toupdate = self.toupdate

        # initialize 'toupdate' with all nodes in g
        for u in g.nodes:
            toupdate.add_node(u, **g.nodes[u])

        # populate the data structures with the graph's edges
        for u, v in self.g.edges:
            cu = self.vc[u]
            cv = self.vc[v]
            self.edges[(cu, cv)].append((u, v))
        pass
    # end

    def _swap_edges(self):
        toadd, toremove = self._select_edges()

        self._add_edges(toadd)
        self._remove_edges(toremove)
    # end

    def _select_edges(self):
        toadd = []
        toremove = []
        culist = shuffle_of(list(range(self.nc)))
        cvlist = shuffle_of(list(range(self.nc)))
        for cu in culist:
            for cv in cvlist:
                if cv == cu: continue
                cut = self.edges[(cu, cv)][:]
                if len(cut) <= 2: continue

                while len(cut) > 2:
                    e1 = cut.pop()
                    e2 = cut.pop()
                    f1 = (e1[0], e2[0])
                    f2 = (e1[1], e2[1])

                    if f1 == f2:
                        continue

                    if random() <= self.rho:
                        toremove += [e1, e2]
                        toadd += [f1, f2]
                # end
            # end
        # end
        return toadd, toremove
    # end

    def _add_edges(self, toadd):
        toupdate = self.toupdate
        count = 0
        for e in toadd:
            if e not in toupdate.edges:
                toupdate.add_edge(*e)
            else:
                count += 1
        # end

        vl = [v for v in toupdate.nodes]
        nv = len(vl)

        # add random edges
        for i in range(count):
            u, v = sample(vl, 2)
            toupdate.add_edge(u, v)
            pass

    def _remove_edges(self, toremove):
        toupdate = self.toupdate
        toupdate.remove_edges_from(toremove)
    # end
# end


class AddRemEdges:
    def __init__(self, g: AGraph,
                 cdict: Union[Dict[str, List[int]], List[List[int]]],
                 rho: float = 0.4,
                 sigma: float = 0.3,
                 connected: bool = True,
                 ntries: int = 10):

        if isinstance(cdict, dict):
            cdict = list(cdict.values())
        assert isinstance(cdict, list)

        self.g = g
        self.directed = g.is_directed()
        # communities
        self.communities = cdict
        # probability keep edge
        self.rho = rho
        self.sigma = sigma
        # n retries
        self.ntries = ntries
        # n of clusters
        self.nc = len(cdict)
        # map: node -> cluster
        self.vc: Dict[int, int] = dict()

        # edges in cluster & between clusters
        # self.edges: Dict[Tuple[int, int], List[Edge]] = defaultdict(lambda: [])
        # adjacent lists
        self.adj: Dict[int, Set[int]] = defaultdict(lambda: set())

        self.toupdate = None
        pass

    def apply(self, toupdate: AGraph):
        self.toupdate: nx.Graph = toupdate
        self.directed = toupdate.is_directed()

        self._populate()
        self._remove_edges()
        self._add_edges()
    # end

    def _populate(self):
        communities = self.communities

        # 1) create the map node -> community
        for c in range(self.nc):
            community = communities[c]
            for n in community:
                self.vc[n] = c

        g = self.g
        toupdate = self.toupdate

        # initialize 'toupdate' with all nodes in g
        for u in g.nodes:
            toupdate.add_node(u, **g.nodes[u])

        # populate the data structures with the graph's edges
        # for u, v in self.g.edges:
        #     cu = self.vc[u]
        #     cv = self.vc[v]
        #     self.edges[(cu, cv)].append((u, v))
    # end

    def _remove_edges(self):
        toremove = []
        for e in self.g.edges:
            u, v = e
            cu = self.vc[u]
            cv = self.vc[v]
            if cu != cv and random() < self.rho:
                toremove.append(e)

        # for cu in range(self.nc):
        #     for cv in range(self.nc):
        #         if cu != cv:
        #             for e in self.edges[(cu, cv)]:
        #                 if random() < self.rho:
        #                     toremove.append(e)
        self.toupdate.remove_edges_from(toremove)
        pass
    # end

    def _add_edges(self):
        toadd = []
        for c in range(self.nc):
            community = self.communities[c]
            for u in community:
                for v in community:
                    if u != v and random() < self.sigma:
                        toadd.append((u, v))
        self.toupdate.add_edges_from(toadd)
        pass
    # end
# end


def force_communities(g: AGraph,
                      cdict: Dict[str, List[int]],
                      rho=0.4,
                      sigma=0.3,
                      mode="addrem",
                      connected=True) -> AGraph:
    """
    Force a cluster structure in the graph

    :param g:
    :param cdict:
    :param rho:
    :param sigma:
    :param mode:
    :return:
    """
    with_communities = g.copy()

    if mode == "swap":
        forcecomm = SwapEdges(g, cdict=cdict, rho=rho, sigma=sigma, connected=connected)
    else:
        forcecomm = AddRemEdges(g, cdict=cdict, rho=rho, sigma=sigma, connected=connected)

    forcecomm.apply(with_communities)
    remove_loops(with_communities)

    return with_communities
# end


# ---------------------------------------------------------------------------
# spectral_communities
# ---------------------------------------------------------------------------

from sklearn.cluster import SpectralClustering


def spectral_communities(g: nx.Graph, n_clusters=8, **kwargs) -> list[list[int]]:
    X = nx.adjacency_matrix(g)

    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **kwargs)
    sc.fit(X)
    labels = sc.labels_

    if "n_clusters" in kwargs:
        nc = kwargs["n_clusters"]
    else:
        nc = 8

    n = len(labels)
    communities = [[] for i in range(nc)]
    for i in range(n):
        c = labels[i]
        communities[c].append(i)

    return communities
# end
