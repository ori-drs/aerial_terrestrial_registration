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
                    weight=np.linalg.norm(points[i] - points[j]),
                )

        # k_neighbors = len(points)
        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(points)
        # kd_tree = o3d.geometry.KDTreeFlann(cloud)

        # for i in range(len(points)):
        #     [k, idx, _] = kd_tree.search_knn_vector_3d(cloud.points[i], k_neighbors)
        #     for idx_j in idx:
        #         if idx_j != i:
        #             self.graph.add_edge(self.node_name(i), self.node_name(idx_j), weight=np.linalg.norm(points[i] - points[idx_j]))

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
                (u, v): f'{d["weight"]}' for u, v, d in self.graph.edges(data=True)
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
                    if self.compare_edge(edge1, edge2):
                        self.graph.add_edge(node1, node2, weight=0)

    def compare_edge(self, edge1, edge2) -> bool:
        if edge1 is None or edge2 is None:
            return False

        w1 = edge1["weight"]
        w2 = edge2["weight"]
        if abs(w1 - w2) / w2 > 0.1:
            return False
        return True

    def maximum_clique(self):
        max_clique_size = 0
        for clique in nx.enumerate_all_cliques(self.graph):
            if len(clique) > max_clique_size:
                max_clique_size = len(clique)
                max_clique = clique

        # Print the maximum cliques
        print(max_clique)

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
                (u, v): f'{d["weight"]}' for u, v, d in self.graph.edges(data=True)
            }
            pos = nx.circular_layout(self.graph)
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.gca().invert_yaxis()
        plt.show()
