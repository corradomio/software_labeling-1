
# ---------------------------------------------------------------------------
# Graph with similar interface of networx.Graph/DiGraph
# ---------------------------------------------------------------------------
# Used for experiment
#
#   G[u][v] returns the edge attribute dictionary.
#   G[u, v] as G[u][v]  (not supported)
#   n in G tests if node n is in graph G.
#   for n in G: iterates through the graph.
#   for nbr in G[n]: iterates through neighbors.
#   for e in list(G.edges): iterates through edges
#   for v in G.adj[u] | for v in G.succ[u] | for v in G.pred[u]
#
from collections import defaultdict
from typing import Dict, Tuple, List, Union


class Graph:
    """
    A minimal clone of nx.Graph, nx.DiGraph
    In theory it is ha the same interface
    """
    def __init__(self, direct=False, loops=False, **kwargs):
        self._direct = direct
        self._loops = loops
        self._graph = kwargs
        self._nodes: Dict[int, dict] = dict()
        if direct:
            self._edges: Union["dedges", "uedges"] = dedges(loops)
        else:
            self._edges: Union["dedges", "uedges"] = uedges(loops)
        # end
        if "name" not in self._graph:
            self._graph["name"] = "G"
    # end

    def is_directed(self) -> bool:
        return self._direct

    def is_multigraph(self) -> bool:
        return False

    def order(self) -> int:
        """N of nodes"""
        return len(self._nodes)

    def size(self) -> int:
        """N of edges"""
        return len(self._edges)

    @property
    def graph(self) -> dict:
        return self._graph

    @property
    def nodes(self) -> Dict[int, dict]:
        return self._nodes

    @property
    def edges(self) -> Dict[Tuple[int, int], dict]:
        return self._edges

    @property
    def adj(self) -> Dict[int, List[int]]:
        return self._edges.adj

    @property
    def succ(self) -> Dict[int, List[int]]:
        return self._edges.succ

    @property
    def pred(self) -> Dict[int, List[int]]:
        return self._edges.pred

    def add_node(self, n, **kwargs):
        if n not in self._nodes:
            self._nodes[n] = kwargs
            self.adj[n] = []
    # end

    def add_edge(self, u, v, **kwargs):
        if not self._direct and u > v:
            u, v = v, u
        if not self.check_edge(u, v):
            return

        self.add_node(u)
        self.add_node(v)

        if v in self._edges.adj[u]:
            return

        self._edges[(u, v)] = kwargs
    # end

    def check_edge(self, u, v) -> bool:
        # check if u == v (loop)
        # check if (u,v) is an edge already present
        #   (multiple edges)
        return True
    # end

    def has_node(self, n) -> bool:
        return n in self.nodes

    def has_edge(self, e) -> bool:
        return e in self.edges

    def __len__(self):
        return self.order()

    def __contains__(self, n):
        return n in self._nodes

    def __getitem__(self, n):
        return self._edges.adj[n]

    def __iter__(self):
        return iter(self._edges.adj)

    def __repr__(self):
        nv = len(self._nodes)
        ne = len(self._edges)
        name = self.graph["name"] if "name" in self.graph else "G"
        return f"{name}=(|V|={nv}, |E|={ne})"
# end


class uedges(dict):
    """Undirected edges dictionary"""

    def __init__(self, loops=False):
        super().__init__()
        self.loops = loops
        self.adj: Dict[int, List[int]] = defaultdict(lambda: list())

    def __contains__(self, edge):
        u, v = edge
        if u > v: u, v = v, u
        return super().__contains__((u, v))

    def __getitem__(self, edge):
        u, v = edge
        if u > v: u, v = v, u
        return super().__getitem__((u,v))

    def __setitem__(self, edge, value):
        u, v = edge
        if u > v: u, v = v, u
        if not self.loops and u == v:
            return None
        elif super().__contains__((u,v)):
            return super().__setitem__((u, v), value)
        elif u not in self.adj or v not in self.adj[u]:
            self.adj[u].append(v)
            self.adj[v].append(u)
            return super().__setitem__((u,v), value)
        else:
            return None
    # end
# end


class dedges(dict):
    """Directed edges dictionary"""

    def __init__(self, loops=False, **kwargs):
        super().__init__()
        self.loops = loops
        self.succ: Dict[int, List[int]] = defaultdict(lambda: list())
        self.pred: Dict[int, List[int]] = defaultdict(lambda: list())
        self.adj = self.succ

    def __contains__(self, edge):
        u, v = edge
        return super().__contains__((u, v))

    def __getitem__(self, edge):
        u, v = edge
        return super().__getitem__((u, v))

    def __setitem__(self, edge, value):
        u, v = edge
        if not self.loops and u == v:
            return None
        elif super().__contains__((u,v)):
            return super().__setitem__((u, v), value)
        elif u not in self.adj or v not in self.adj[u]:
            self.succ[u].append(v)
            self.pred[v].append(u)
            return super().__setitem__((u,v), value)
        else:
            return None
    # end
# end


