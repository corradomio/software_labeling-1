from .netxfun import *

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# partition_edges_on_clusters
# ---------------------------------------------------------------------------

def partition_edges_on_communities(g: AGraph, cdict: Dict[str, List[int]]) -> EdgeMatrix:
    if isinstance(cdict, dict):
        communities = list(cdict.values())
    else:
        communities = cdict

    # n of communities
    nc = len(communities)
    # node -> cluster
    vc = [0]*g.number_of_nodes()
    for c in range(nc):
        for n in g.nodes:
            vc[n] = c

    edges = [[[] for i in range(nc)] for j in range(nc)]

    for u, v in g.edges:
        cu = vc[u]
        cv = vc[v]

        edges[cu][cv].append((u, v))
    # end

    return edges
# end


# ---------------------------------------------------------------------------
# coarsening_graph
# closure_coarsening_graph
# ---------------------------------------------------------------------------

def coarsening_graph(g: AGraph,
                     partitions: Union[List[List[int]], Dict[str, List[int]]],
                     create_using=None,
                     attribute=None,
                     **kwargs) -> AGraph:
    """
    Create a coarsed graph using the partitions as 'super' nodes and edges between
    partitions i and j if there exist a node in partition i connected to a node in partition j

    :param g: original graph
    :param partitions: partitions (a list of list of nodes)
    :param create_using: graph to populate, or None
    :param attribute: if specified, to the node in the coarsed graph will be assigned
            an attribute with the specified name. The value will be selected based on
            the most represented value in the original nodes. If two attributes are represented
            the same number of times, it will be selected the greater one (if it is a string,
            using the lexicographic order)
    :param kwargs: extra parameters passed to the created graph
    :return: a coarsed graph
    """
    if type(partitions) == list:
        np = len(partitions)
        partitions = {f"p{i:02}": partitions[i] for i in range(np)}

    assert isinstance(partitions, dict)
    assert g.number_of_nodes() == sum(map(len, partitions.values()))

    # in_partition[u] -> partition of u
    def in_partition(nv):
        in_part = [0]*nv
        c = 0
        for partition in partitions.values():
            for u in partition:
                in_part[u] = c
            c += 1
        return in_part

    # create an empty graph
    if create_using is not None:
        coarsed = create_using
    else:
        coarsed = nx.Graph(**kwargs)

    name = g.graph["name"] if "name" in g.graph else "G"
    # if "name" not in coarsed.graph:
    coarsed.graph["name"] = f"coarsed-{name}"

    # force partitions to be a list
    in_part = in_partition(g.number_of_nodes())

    # add nodes
    np = len(partitions)
    for i in range(np):
        coarsed.add_node(i)

    # add edges
    for u, v in list(g.edges):
        cu = in_part[u]
        cv = in_part[v]
        if cu != cv:
            coarsed.add_edge(cu, cv)
    # end

    # assign the attribute value
    if attribute is not None:
        values = list(partitions.keys())
        for i in range(np):
            coarsed.nodes[i][attribute] = values[i]
        # end
    # end
    return coarsed
# end


def closure_coarsening_graph(g: AGraph,
                             create_using=None,
                             attribute=None,
                             **kwargs) -> Tuple[AGraph, Dict[int, List[int]]]:
    """
    Create a coarsed graph using the following protocol

        1) for each node creates the transitive closure
        2) create an edge from closure i and closure j if closure i is a proper superset of closure j
        3) apply a transitive reduction

    :param g: original graph
    :param create_using: graph to populate, or None
    :param attribute: if specified, to the nodes in the coarsed graph will be assigned
            an attribute with the specified name. The value will be selected based on
            the most represented value in the original nodes. If two attribute's
            values are represented the same number of times, it will be selected the
            greater one (if it is a string, using the lexicographic order)
    :param kwargs: extra parameters passed to the created graph
    :return: a coarsed graph, the map 'node->closure'
    """
    assert g.is_directed()

    def closure_of(u: int, closures: Dict[int, Set[int]]) -> Set[int]:
        visited: Set[int] = set()
        tovisit: List[int] = [u]
        while len(tovisit) > 0:
            u: int = tovisit.pop()
            if u in visited:
                continue
            if u in closures:
                visited.update(closures[u])
            else:
                visited.add(u)
                tovisit += list(g.succ[u])
        # end
        return visited
    # end

    # create an empty graph
    if create_using is not None:
        coarsed = create_using
    else:
        coarsed = nx.Graph(**kwargs)
    # end

    name = g.graph["name"] if "name" in g.graph else "G"
    coarsed.graph["name"] = f"closure-coarsed-{name}"

    # compute the closures
    nclosures: Dict[int, Set[int]] = dict()
    for u in g:
        nclosures[u] = closure_of(u, nclosures)
    # end

    # remove duplicate closures
    # keep the closure originated from the node with the lower id
    closures = dict()
    closures.update(nclosures)
    nodes = list(closures.keys())
    n = len(nodes)
    for i in range(0, n-1):
        u = nodes[i]
        # skip deleted nodes
        if u not in closures:
            continue
        cu = closures[u]
        for j in range(i+1, n):
            v = nodes[j]
            # skip deleted nodes
            if v not in closures:
                continue
            cv = closures[v]
            # if the same closure, delete the second one
            if cu == cv:
                del closures[v]
        # end
    # end

    # compute the simplified graph
    # (using the subset of nodes)
    for u in closures:
        cu = closures[u]
        for v in closures:
            cv = closures[v]
            # check for closure(u) > closure(v)
            #   then create the edge u->v
            if u == v:
                continue
            if len(cu) == len(cv):
                continue
            if cv.issubset(cu):
                coarsed.add_edge(u, v)
        # end
    # end/

    # apply the transitive reduction
    reducted = nx.transitive_reduction(coarsed)
    # propagate the graph attributes
    reducted.graph.update(coarsed.graph)

    # assign the attribute value
    if attribute is not None:
        for u in reducted.nodes:
            selected = closures[u]
            # value = _select_attribute(attribute, g.nodes, closure)
            counts = count_attribute(g, attribute, selected)
            value = counts.most_common()[0][0]
            reducted.nodes[u][attribute] = value
        # end
    # return the coarsed graph and the list of
    return reducted, nclosures
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
