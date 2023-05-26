import networkx as nx


#
# networkx Graph & Digraph without loops
#

class DiGraph(nx.DiGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        if u_of_edge != v_of_edge:
            super().add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        for edge in ebunch_to_add:
            u_of_edge = edge[0]
            v_of_edge = edge[1]
            self.add_edge(u_of_edge, v_of_edge, **attr)

    def remove_edge(self, u, v):
        if (u, v) in self.edges:
            super().remove_edge(u, v)
# end


class Graph(nx.Graph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        if u_of_edge != v_of_edge:
            super().add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        for edge in ebunch_to_add:
            u_of_edge = edge[0]
            v_of_edge = edge[1]
            self.add_edge(u_of_edge, v_of_edge, **attr)

    def remove_edge(self, u, v):
        if (u,v) in self.edges:
            super().remove_edge(u, v)
# end
