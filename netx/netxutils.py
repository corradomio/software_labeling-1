from typing import Union, List, Dict, Optional
from tabulate import tabulate
import networkx as nx
import numpy as npy

AGraph = Union[nx.Graph, nx.DiGraph]


# ---------------------------------------------------------------------------
# write_vertex_category
# read_vertex_category
# ---------------------------------------------------------------------------

def write_vertex_category(file: str, data: Union[List[int], Dict[str, List[int]]]):
    if isinstance(data, list):
        data = {i: data[i] for i in range(len(data))}

    with open(file, mode="wt") as wrt:
        for k, v in data.items():
            sdata = ",".join(map(str, v))
            wrt.write(f"{k}:{sdata}\n")
    # end
# end


def read_vertex_category(file: str) -> Dict[str, List[int]]:
    data = dict()
    with open(file, mode="rt") as rdr:
        for line in rdr:
            key, sdata = line.split(':')
            data[key] = list(map(int, sdata.split(',')))
    return data
# end


# ---------------------------------------------------------------------------
# load_community_map
# set_node_attribute
# ---------------------------------------------------------------------------

def load_community_map(csvfile: str, header=1) -> Dict[str, str]:
    """
    The file must have the structure:

        category,namespace, ...

    with 1 line of header and '#' as comment lines

    It return the map: namespace -> category
    """
    map = dict()
    with open(csvfile) as rdr:
        for line in rdr:
            line = line.strip()
            if line.startswith("#"):
                continue
            if len(line) == 0:
                continue
            if header > 0:
                header -= 1
                continue
            parts = line.split(",")
            map[parts[1]] = parts[0]
    # end
    return map
# end


def set_node_attribute(g: AGraph,
                       attribute: str,
                       communities: Union[List[int], Dict[str, List[int]], None] = None,
                       value: Optional = None,
                       map: Dict[str, str] = None) -> AGraph:
    assert attribute is not None

    if map is None:
        map = dict()

    if communities is None:
        assert map is not None;
        for n in g.nodes:
            value = g.nodes[n][attribute]
            g.nodes[n][attribute] = value if value not in map else map[value]
        # end
        return g
    # end

    if isinstance(communities, list):
        assert isinstance(communities, list)
        nvalue = value if value not in map else map[value]
        for n in communities:
            g.nodes[n][attribute] = nvalue
    else:
        assert isinstance(communities, dict)
        for value in communities:
            nvalue = value if value not in map else map[value]
            for n in communities[value]:
                g.nodes[n][attribute] = nvalue
    # end
    return g
# end


# ---------------------------------------------------------------------------
# analyze_clusters
# ---------------------------------------------------------------------------

def analyze_communities(g: AGraph,
                        cdict: Union[List[List[int]], Dict[str, List[int]]],
                        show=True,
                        label=None):

    if isinstance(cdict, Dict):
        pnames = sorted(list(cdict.keys()))
        communities = [cdict[c] for c in pnames]
    else:
        communities = cdict

    directed = g.is_directed()

    counts = list(map(len, communities))

    np = len(cdict)
    nv = max(max(cdict[c]) for c in cdict)+1
    vc = npy.zeros(nv, dtype=int)

    matrix = npy.zeros((np, np), dtype=int)

    for c in range(np):
        for v in communities[c]:
            vc[v] = c

    for u, v in g.edges:
        cu = vc[u]
        cv = vc[v]
        if directed:
            matrix[cu, cv] += 1
        else:
            matrix[cu, cv] += 1
            matrix[cv, cu] += 1
    # end

    if label is not None:
        _analyze_communities_save_on_file(label, matrix)

    if not show:
        return counts, matrix

    print("n communities:", len(cdict))

    # pnames = list(cdict.keys())
    headers = [""] + pnames
    data = []
    data.append(["count"] + counts)
    data.append(["-"]*(np + 1))
    for i in range(np):
        row = [pnames[i]] + matrix[i, :].tolist()
        data.append(row)
    # end

    print(tabulate(data, headers=headers))
    print()

    # -----------------------------------------------------------------------

    nv = g.number_of_nodes()
    ne = g.number_of_edges()

    headers = ["name", "value"]
    data = [
        ["directed", g.is_directed()],
        ["nodes", nv],
        ["edges", ne],
        ["density", 2*ne / (nv*nv-nv)],
        ["communities", np],
        ["modularity", nx.algorithms.community.modularity(g, communities)],
        ["coverage", nx.algorithms.community.coverage(g, communities)],
        # ["partition_quality", nx.algorithms.community.partition_quality(g, communities)],
        # ["performance", nx.algorithms.community.performance(g, communities)],
    ]
    print(tabulate(data, headers=headers))

    print(g.number_of_edges(), matrix.sum()/2)

    print()
    print()

    return counts, matrix
# end


def _analyze_communities_save_on_file(label, matrix):
    with open(f"experiments/{label}.m", mode="w") as f:
        def ctos(r):
            return "{" + ",".join(map(str, r)) + "}"

        def rtos(c):
            return ",\n".join(c)

        c = map(ctos, matrix)
        m = rtos(c)
        s = "{" + m + "}\n"
        f.write(s)
    # end
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
