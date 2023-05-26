import networkx as nx
from typing import Tuple, Union, List

Edge = Tuple[int, int]
AGraph = Union[nx.Graph, nx.DiGraph]
EdgeMatrix = List[List[List[Edge]]]
