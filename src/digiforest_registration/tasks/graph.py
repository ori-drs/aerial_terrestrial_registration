import numpy as np
import math
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
        # angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        # if angle < 0:
        #     angle += np.pi
        # return angle
        # angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        # if p2[0] - p1[0] < 0:
        #     angle = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        # if angle < 0:
        #     angle += np.pi
        # dx = p2[0] - p1[0]
        # dy = p2[1] - p1[1]
        # if dx == 0:
        #     return 0
        # angle = np.arctan(dy / dx)
        x = p2[0] - p1[0]
        y = p2[1] - p1[1]
        angle = np.arccos(x / math.sqrt(x * x + y * y))
        inverted_angle = np.arccos(-x / math.sqrt(x * x + y * y))
        angle = max(angle, inverted_angle)
        return angle

    def _test_angle(self):
        # TODO to be move elsewhere
        # all comparison should be true

        # reason why arctan(y/x) isn't good
        print(
            self.get_angle([108, 50], [100, 96]),
            print(self.get_angle([339, 175], [340, 217])),
        )

        # reason why arctan2 isn't good
        print(
            self.get_angle([0, 0], [-30, 1]), print(self.get_angle([0, 0], [-30, -1]))
        )

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
    def __init__(self, graph1: Graph, graph2: Graph, logger):
        self.graph = nx.cartesian_product(graph1.graph, graph2.graph)
        self.g1 = graph1
        self.g2 = graph2
        self.graph.remove_edges_from(list(self.graph.edges()))
        self.distance_threshold = 0.2
        self.angle_threshold = 0.25
        self.logger = logger

        # Creating the edges
        for node1 in self.graph.nodes():
            for node2 in self.graph.nodes():
                if node1[0] != node2[0] and node1[1] != node2[1]:  # nodes are different
                    edge1 = graph1.graph.get_edge_data(node1[0], node2[0])
                    edge2 = graph2.graph.get_edge_data(node1[1], node2[1])
                    if self.compare_edge(edge1, edge2, use_angle=True):
                        self.graph.add_edge(node1, node2, weight=0)

        self.logger.debug(
            f"Number of nodes in the correspondence graph: {self.graph.number_of_nodes()}"
        )
        self.logger.debug(
            f"Number of edges in the correspondence graph: {self.graph.number_of_edges()}"
        )

    def compare_edge(self, edge1, edge2, use_angle=False, debug=False) -> bool:
        """
        Return true if the two edges are similar"""
        if edge1 is None or edge2 is None:
            return False

        w1 = edge1["distance"]
        w2 = edge2["distance"]

        if debug:
            print(w1, w2, abs(w1 - w2) / w2)
        if abs(w1 - w2) / w2 > self.distance_threshold:
            return False

        if use_angle:
            a1 = edge1["angle"]
            a2 = edge2["angle"]
            if debug:
                print(a1, a2, abs(a1 - a2))
            if abs(a1 - a2) > self.angle_threshold:
                return False
        return True

    def maximum_clique(self) -> list:
        """
        Compute the maximum cliques.
        Returns the edges of the maximum cliques.
        """
        max_clique_size = 0
        max_cliques = []
        # max_clique = nx.algorithms.approximation.max_clique(self.graph)
        # for clique in nx.enumerate_all_cliques(self.graph):
        for clique in nx.find_cliques(self.graph):  # seems to be the fastest method
            if len(clique) > max_clique_size:
                max_clique_size = len(clique)
                max_cliques.clear()

            if len(clique) == max_clique_size:
                max_cliques.append(clique)

        if len(max_cliques) == 0:
            return []

        # Print the maximum cliques
        self.logger.debug(f"Length of the maximum clique {len(max_cliques[0])}")
        self.logger.debug(f"Number of maximum cliques: {len(max_cliques)}")

        # Create the edges
        edges = []
        for i in range(len(max_cliques)):
            edges.append([])  # edges is a list of list
            for c in max_cliques[i]:
                edges[i].append((self.g1.pos[c[0]], self.g2.pos[c[1]]))
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
