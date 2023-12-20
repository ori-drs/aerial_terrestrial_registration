from digiforest_registration.optimization.pose_graph import PoseGraph
import digiforest_registration.optimization.visualization as vis
import gtsam
import open3d as o3d


class PoseGraphOptimization:
    def __init__(self, pose_graph: PoseGraph):
        self.pose_graph = pose_graph
        self.factor_graph = None
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
            if e["type"] == "center":
                noise = gtsam.noiseModel.Gaussian.Information(e["info"])
                # TODO check if the direction of the edge is correct
                # ie child -> parent of parent -> child
                factor_graph.add(
                    gtsam.BetweenFactorPose3(
                        e["child_id"],
                        e["parent_id"],
                        e["pose"],
                        noise
                        # e["parent_id"], e["child_id"], e["pose"], noise
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
        # optimized_pose_graph = copy.deepcopy(self.pose_graph)

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
