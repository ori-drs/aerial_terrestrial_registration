from digiforest_registration.optimization.pose_graph import PoseGraph
import digiforest_registration.optimization.visualization as vis
import gtsam
import open3d as o3d


class PoseGraphOptimization:
    def __init__(self, pose_graph: PoseGraph, show_clouds=False, process_tiles=False):
        self.pose_graph = pose_graph
        self.show_clouds = show_clouds
        self.factor_graph = None
        self.process_tiles = process_tiles
        self.initialize_factor_graph()

    def initialize_factor_graph(self):
        # Create a factor graph container and add factors to it
        factor_graph = gtsam.NonlinearFactorGraph()

        # set prior on the root node
        # prior_noise = gtsam.noiseModel.Diagonal.Sigmas(0.000001 * np.ones(6))
        # factor_graph.add(
        #     gtsam.PriorFactorPose3(
        #         self.pose_graph.root_id,
        #         self.pose_graph.get_node_pose(self.pose_graph.root_id),
        #         prior_noise,
        #     )
        # )

        # add the other edges
        for e in self.pose_graph.edges:
            if e["type"] == "in-between":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                # TODO check if the direction of the edge is correct
                # ie child -> parent of parent -> child

                if self.process_tiles:
                    # TODO it's for legacy reasons only, there should be no difference
                    # between tiles and payloads. The in-between transform for the tiles is not
                    # saved correctly in the pose graph file ( the inverse is saved instead )
                    factor_graph.add(
                        gtsam.BetweenFactorPose3(
                            e["child_id"], e["parent_id"], e["pose"], noise
                        )
                    )
                else:
                    # payload clouds
                    factor_graph.add(
                        gtsam.BetweenFactorPose3(
                            e["parent_id"], e["child_id"], e["pose"], noise
                        )
                    )
            elif e["type"] == "aerial":
                # if e["parent_id"] == self.pose_graph.root_id:
                #     # the root node is fixed by a prior constraint already
                #     continue

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
            print(f"Node {id}:")
            print(f"  pose: {node['pose']}")

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

        if self.show_clouds:
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

        self.visualize(initial_estimate)

        # Setup optimizer
        parameters = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(
            self.factor_graph, initial_estimate, parameters
        )

        # Optimize
        print("Optimizing factor graph...")
        result = optimizer.optimize()

        for id, _ in self.pose_graph.nodes.items():
            self.pose_graph.set_node_pose(id, result.atPose3(id))

        # Display results
        print("********")
        for id, node in self.pose_graph.nodes.items():
            print(f"Node {id}:")
            print(f"  pose: {node['pose']}")

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

        if self.show_clouds:
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
