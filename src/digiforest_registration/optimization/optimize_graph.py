from digiforest_registration.optimization.pose_graph import PoseGraph
import gtsam
import numpy as np
import copy


class PoseGraphOptimization:
    def __init__(self, pose_graph: PoseGraph):
        self.pose_graph = pose_graph
        self.factor_graph = None
        self.initialize_factor_graph()

    def initialize_factor_graph(self):
        # Create a factor graph container and add factors to it
        factor_graph = gtsam.NonlinearFactorGraph()

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6))
        factor_graph.add(
            gtsam.PriorFactorPose3(
                self.pose_graph.root_id,
                self.pose_graph.get_node_pose(self.pose_graph.root_id),
                prior_noise,
            )
        )

        for e in self.pose_graph.edges:
            if e["type"] == "center":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                factor_graph.add(
                    gtsam.BetweenFactorPose3(
                        e["parent_id"], e["child_id"], e["pose"], noise
                    )
                )
            elif e["type"] == "aerial":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                factor_graph.add(
                    gtsam.PriorFactorPose3(e["parent_id"], e["pose"], noise)
                )

        self.factor_graph = factor_graph

    def visualize(self, estimate):
        self.factor_graph.saveGraph("/tmp/test.dot", estimate)

        from graphviz import Source

        s = Source.from_file("/tmp/test.dot")
        s.view()

    def optimize(self):
        optimized_pose_graph = copy.deepcopy(self.pose_graph)

        initial_estimate = gtsam.Values()
        for id, node in self.pose_graph.nodes.items():
            initial_estimate.insert(id, node["pose"])

        self.visualize(initial_estimate)

        # Setup optimizer
        parameters = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(
            self.factor_graph, initial_estimate, parameters
        )

        # Optimize
        print("Optimizing factor graph...")
        result = optimizer.optimize()
        for id, value in optimized_pose_graph.nodes.items():
            optimized_pose_graph.set_node_pose(id, result.atPose3(id))
