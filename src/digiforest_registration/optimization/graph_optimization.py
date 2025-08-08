from digiforest_registration.optimization.pose_graph import PoseGraph
import digiforest_registration.optimization.visualization as vis
import gtsam
import open3d as o3d
import logging


class PoseGraphOptimization:
    def __init__(self, pose_graph: PoseGraph, debug, show_clouds=False, logger=None):
        self.pose_graph = pose_graph
        self.show_clouds = show_clouds
        self.factor_graph = None
        self.debug = debug
        self.logger = logger or logging.getLogger(
            __name__ + ".null"
        )  # no-op logger if logger is not provided
        self.initialize_factor_graph()

    def initialize_factor_graph(self):
        # Create a factor graph container and add factors to it
        factor_graph = gtsam.NonlinearFactorGraph()

        # add the other edges
        for e in self.pose_graph.edges:
            if e["type"] == "in-between":
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
            elif e["type"] == "loop-closure":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                robust_model = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(1.0), noise
                )
                factor_graph.add(
                    gtsam.BetweenFactorPose3(
                        e["child_id"], e["parent_id"], e["pose"], robust_model
                    )
                )

        self.factor_graph = factor_graph

    def visualize(self, estimate):
        self.factor_graph.saveGraph("/tmp/test.dot", estimate)

        from graphviz import Source

        s = Source.from_file("/tmp/test.dot")
        s.view()

    def optimize(self):
        # Display initial node poses
        for id, node in self.pose_graph.nodes.items():
            self.logger.info(f"Node {id}:")
            self.logger.info(f"  pose: {node['pose']}")

        if self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=False,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Initial Pose Graph",
            )

        if self.show_clouds and self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=True,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Initial Pose Graph with Clouds",
            )

        initial_estimate = gtsam.Values()
        for id, node in self.pose_graph.nodes.items():
            initial_estimate.insert(id, node["pose"])

        if self.debug:
            self.visualize(initial_estimate)

        # Setup optimizer
        parameters = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(
            self.factor_graph, initial_estimate, parameters
        )

        # Optimize
        self.logger.info("Optimizing factor graph...")
        result = optimizer.optimize()

        for id, _ in self.pose_graph.nodes.items():
            self.pose_graph.set_node_pose(id, result.atPose3(id))

        # Display results
        for id, node in self.pose_graph.nodes.items():
            self.logger.info(f"Node {id}: pose: {node['pose']}")

        if self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=False,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Optimized Pose Graph",
            )

        if self.show_clouds and self.debug:
            geometries = vis.graph_to_geometries(
                self.pose_graph,
                show_frames=True,
                show_edges=True,
                show_nodes=True,
                show_clouds=True,
                show_coordinate_frame=True,
                odometry_color=vis.GRAY,
                loop_color=vis.RED,
            )
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Optimized Pose Graph with clouds",
            )
