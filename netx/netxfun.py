from collections import defaultdict, Counter
from typing import Optional, Set, Any, Dict

import networkx as nx

from .netxtypes import *


# ---------------------------------------------------------------------------
# select_nodes_by_attribute
# ---------------------------------------------------------------------------

def degree_of(g: AGraph, n: Union[int, List[int], None] = None, mode: Union[None, bool, str] = None) \
        -> Union[int, List[int]]:
    """
    Degree of the node based on the graph type and which degree to
    retrieve

    :param g: selected graph
    :param n: selected node
    :param mode: if to retrieve the total/input/output degree

        None:  degree        (also 'deg',    'degee')
        False: output degree (also 'outdeg', 'outdegree'
        True:  input  degree (also 'indeg',  'indegree')

    :return: the node's degree
    """
    if n is None:
        n = list(g.nodes)

    directed = g.is_directed()

    if type(n) == int:
        if not directed:
            return len(g.adj[n])
        elif mode in [None, 'deg', 'degree']:
            return len(g.pred[n]) + len(g.succ[n])
        elif mode in [True, 'indeg', 'indegree']:
            return len(g.pred[n])
        elif mode in [False, 'outdeg', 'outdegree']:
            return len(g.succ[n])
        else:
            raise ValueError("Unsupported mode {mode}")
    else:
        if not directed:
            degof = lambda g, n: len(g.adj[n])
        elif mode in [None, 'deg', 'degree']:
            degof = lambda g, n: len(g.pred[n]) + len(g.succ[n])
        elif mode in [True, 'indeg', 'indegree']:
            degof = lambda g, n: len(g.pred[n])
        elif mode in [False, 'outdeg', 'outdegree']:
            degof = lambda g, n: len(g.succ[n])
        else:
            raise ValueError("Unsupported mode {mode}")

        selected = n
        degrees = []
        for n in selected:
            degrees.append(degof(g, n))
        return degrees
    # end
# end


def values_of(g: AGraph, attribute: str, selected: Optional[List[int]] = None) -> Set[Any]:
    """
    Retrieve all possible values of the selected attribute

    :param g: graph to analyze
    :param attribute: attribute to analyze
    :param selected: if to analyze a subset of nodes
    :return: list of attribute's values
    """
    if selected is None:
        selected = g.nodes

    values = set()
    for n in selected:
        values.add(g.nodes[n][attribute])
    return values
# end


def communities_on_attribute(g: AGraph, attribute: str) -> Dict[str, List[int]]:
    """
    Create a node partition based on the node's attribute values

    :param g: graph
    :param attribute: attribute to analyze
    :return: map value -> nodes with the attribute's value
    """
    communities: Dict[str, List[int]] = dict()

    for n in g:
        value = g.nodes[n][attribute]
        if value not in communities:
            communities[value] = []
        communities[value].append(n)
    # end
    return communities
# end


def partition_nodes_on_degree(g: AGraph, mode=None) -> Dict[int, List[int]]:
    """
    Create a dictionary where the key is the node degree and the value the list
    of nodes with the same degree

    :param g: graph to analyze
    :param mode: if to retrieve the total/input/output degree

        None:  degree        (also 'deg',    'degee')
        False: output degree (also 'outdeg', 'outdegree'
        True:  input  degree (also 'indeg',  'indegree')

    """
    dnodes = defaultdict(lambda: [])
    for n in g.nodes:
        d = degree_of(g, n, mode=mode)
        dnodes[d].append(n)
    return dnodes
# end


def count_attribute(g: AGraph, attribute: str, selected: Optional[List[int]]=None) -> Counter[Any, int]:
    """
    Count the number of occurrences of the attribute's values

    :param g: graph to analyze
    :param attribute: selected attribute
    :param selected: if to scan a node's subset
    :return: map 'value->n of nodes with the value'
    """
    if selected is None:
        selected = g.nodes
    counts = Counter(g.nodes[n][attribute] for n in selected)
    return counts
# end


def select_nodes_by_degree(g: AGraph, degree: Union[int, Tuple[int, int]] = 0,
                           mode: Union[None, bool, int, str] = None) \
        -> List[int]:
    """
    Select the nodes based on the degree

    :param g: graph to analyze
    :param degree: degree of the nodes. If it is a tuple, select nodes with degree min <= degree(n) < max
    :param mode: how to select the nodes:

        None:  degree        (also 'deg',    'degee')
        False: output degree (also 'outdeg', 'outdegree'
        True:  input  degree (also 'indeg',  'indegree')

    :return: list of nodes
    """
    directed = g.is_directed()
    mind, maxd = degree if type(degree) != int else degree, degree+1

    # how to select the degree
    if not directed:
        degof = lambda g, n: len(g.adj[n])
    elif mode in {0, None, "degree", "deg"}:
        degof = lambda g, n: len(g.pred[n]) + len(g.succ[n])
    elif mode in {-1, True, "indegree", "indeg"}:
        degof = lambda g, n: len(g.pred[n])
    elif mode in {1, False, "outdegree", "outdeg"}:
        degof = lambda g, n: len(g.succ[n])
    else:
        raise ValueError(f"Unsupported mode {mode}")

    selected = []
    for n in g:
        if mind <= degof(g, n) < maxd:
            selected.append(n)
    # end
    return selected
# end


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------

def is_connected(g, undirected=True):
    if undirected and g.is_directed():
        g = g.to_undirected()
    return len(list(nx.connected_components(g))) == 1
# end


# ---------------------------------------------------------------------------
# directed_configuration_model
# ---------------------------------------------------------------------------

def directed_configuration_model(idegs, odegs, create_using=None, connected=False, ntries=10) \
    -> nx.DiGraph:
    dcm = None
    for i in range(ntries):
        dcm = nx.generators.directed_configuration_model(idegs, odegs, create_using=create_using)
        if is_connected(dcm):
            break
    return dcm
# end


def configuration_model(degs, create_using=None, connected=False, ntries=10) \
    -> nx.Graph:
    dcm = None
    for i in range(ntries):
        dcm = nx.generators.configuration_model(degs, create_using=create_using)
        if is_connected(dcm):
            break
    return dcm
# end


# ---------------------------------------------------------------------------
# miscellanie
# ---------------------------------------------------------------------------

def print_graph_stats(g: AGraph):
    n = g.number_of_nodes()
    m = len(g.edges)
    if not g.is_directed():
        print(f"G={{V: {n}, E: {m}}}")
    else:
        print(f"G={{V: {n}, E: {m}}}, directed")
# end

