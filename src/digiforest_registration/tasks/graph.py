import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, points, node_prefix: str = "node"):
        self.graph = nx.Graph()
        # A dictionary with nodes as keys and 2d positions as values
        self.pos = {}
        self.node_prefix = node_prefix
        # adding nodes
        for i in range(len(points)):
            self.graph.add_node(self.node_name(i))
            self.pos[self.node_name(i)] = points[i][0:2]

        # adding edges
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                self.graph.add_edge(
                    self.node_name(i),
                    self.node_name(j),
                    distance=np.linalg.norm(points[i][0:2] - points[j][0:2]),
                    angle=self.get_angle(points[i][0:2], points[j][0:2]),
                )

    def get_angle(self, p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def display_graph(self, display_weights: bool = False, display_edges: bool = True):
        if display_edges:
            nx.draw(
                self.graph,
                pos=self.pos,
                with_labels=True,
                node_color="skyblue",
                font_weight="bold",
            )
        else:
            nx.draw_networkx_nodes(
                self.graph, pos=self.pos, node_color="skyblue", node_size=100
            )
            nx.draw_networkx_labels(self.graph, pos=self.pos, font_weight="bold")

        # Draw edge labels
        if display_weights:
            edge_labels = {
                (u, v): f'{d["distance"]}' for u, v, d in self.graph.edges(data=True)
            }
            pos = nx.circular_layout(self.graph)
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.gca().invert_yaxis()
        plt.show()

    def node_name(self, i: int):
        return self.node_prefix + "_" + str(i)

    def node_id(self, node_name: str):
        return int(node_name.split("_")[1])


class CorrespondenceGraph:
    def __init__(self, graph1: Graph, graph2: Graph):
        self.graph = nx.cartesian_product(graph1.graph, graph2.graph)
        self.g1 = graph1
        self.g2 = graph2
        self.graph.remove_edges_from(list(self.graph.edges()))

        # Creating the edges
        for node1 in self.graph.nodes():
            for node2 in self.graph.nodes():
                if node1[0] != node2[0] and node1[1] != node2[1]:  # nodes are different
                    edge1 = graph1.graph.get_edge_data(node1[0], node2[0])
                    edge2 = graph2.graph.get_edge_data(node1[1], node2[1])
                    if self.compare_edge(edge1, edge2, use_angle=True):
                        self.graph.add_edge(node1, node2, weight=0)

        print(
            "Number of nodes in the correspondence graph: ",
            self.graph.number_of_nodes(),
        )
        print(
            "Number of edges in the correspondence graph: ",
            self.graph.number_of_edges(),
        )

    def compare_edge(self, edge1, edge2, use_angle=False) -> bool:
        if edge1 is None or edge2 is None:
            return False

        w1 = edge1["distance"]
        w2 = edge2["distance"]
        if abs(w1 - w2) / w2 > 0.1:
            return False

        if use_angle:
            a1 = edge1["angle"]
            a2 = edge2["angle"]
            if abs(a1 - a2) > 0.1:
                return False
        return True

    def maximum_clique(self):
        max_clique_size = 0
        # max_clique = nx.algorithms.approximation.max_clique(self.graph)
        for clique in nx.enumerate_all_cliques(self.graph):
            # for clique in nx.find_cliques(self.graph):
            if len(clique) > max_clique_size:
                max_clique_size = len(clique)
                max_clique = clique

        # Print the maximum cliques
        print(len(max_clique), max_clique)

        # Create the edges
        edges = []
        for c in max_clique:
            edges.append((self.g1.pos[c[0]], self.g2.pos[c[1]]))
        return edges

    def display_graph(self, display_weights: bool = False):
        nx.draw(
            self.graph,
            with_labels=True,
            node_color="skyblue",
            font_weight="bold",
            edge_color="white",
        )

        # Draw edge labels
        if display_weights:
            edge_labels = {
                (u, v): f'{d["distance"]}' for u, v, d in self.graph.edges(data=True)
            }
            pos = nx.circular_layout(self.graph)
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.gca().invert_yaxis()
        plt.show()
